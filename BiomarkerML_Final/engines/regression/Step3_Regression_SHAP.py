from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

HERE = Path(__file__).resolve().parent
SCRIPT_ROOT = HERE.parents[0]
for path in (HERE, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core.publication_utils import (
    DEFAULT_RANDOM_STATE,
    assert_protein_level_features,
    build_output_prefix,
    configure_publication_plots,
    estimator_importance_with_source,
    format_effect_value,
    finalize_shap_table,
    meaningful_display_mask,
    prediction_input_with_feature_names,
    selected_feature_names_from_pipeline,
    set_global_seed,
    stage_task_inputs,
    task_output_dir,
    task_run_context,
)
from engines.ppi.protein_network import publication_gene_symbol_map

configure_publication_plots()
warnings.filterwarnings("ignore")

MODEL_ALIASES = {
    "NN_reg": "Neural_Network_reg",
    "RF_reg": "Random_Forest_reg",
    "SVM_reg": "SVM_reg",
    "XGB_reg": "XGBoost_reg",
    "PLS_reg": "PLS_reg",
    "KNN_reg": "KNN_reg",
    "LGBM_reg": "LightGBM_reg",
    "VAE_reg": "VAE_MLP_reg",
    "MLPVAE_reg": "MLP_in_VAE_reg",
}
TREE_MODELS = {"Random_Forest_reg", "XGBoost_reg", "LightGBM_reg"}

# Independently calibrated bounded permutation-SHAP policy for regression VAE models.
REGRESSION_VAE_SHAP_EXPLANATION_COHORT_SIZE = 100
REGRESSION_VAE_SHAP_DEFAULT_BACKGROUND_SIZE = 50
REGRESSION_VAE_SHAP_PERMUTATION_CYCLES = 5
REGRESSION_MLPVAE_SHAP_PERMUTATION_CYCLES = 3
REGRESSION_VAE_SHAP_DEFAULT_BATCH_SIZE = 64


def _resolve_model_name(name: str) -> str:
    for alias, value in MODEL_ALIASES.items():
        if name.casefold() == alias.casefold() or name.casefold() == value.casefold():
            return value
    return name


def _prepare_features(model: Pipeline, X: pd.DataFrame):
    if not isinstance(model, Pipeline):
        names = list(X.columns)
        return prediction_input_with_feature_names(X, names, model), names, model, "original_proteins"
    preprocessor = model.named_steps.get("preprocessor")
    names = list(map(str, preprocessor.get_feature_names_out())) if preprocessor is not None else selected_feature_names_from_pipeline(model, list(X.columns))
    current: Any = X
    for _, step in model.steps[:-1]:
        current = step.transform(current)
    matrix = np.asarray(current)
    if len(names) != matrix.shape[1]:
        names = [f"Protein_{i+1}" for i in range(matrix.shape[1])]
    assert_protein_level_features(names, "Regression SHAP")
    source = "selected_biomarkers_and_covariates" if preprocessor is not None else ("selected_proteins" if "feature_selection" in model.named_steps else "original_proteins")
    final_estimator = model.steps[-1][1]
    return prediction_input_with_feature_names(matrix, names, final_estimator), names, final_estimator, source


def _save_official_shap_plots(
    prefix: str,
    model_name: str,
    values,
    X_sample: np.ndarray,
    feature_names: list[str],
    display_names: dict[str, str],
) -> None:
    try:
        import shap

        array = np.asarray(values.values if hasattr(values, "values") else values)
        if array.ndim != 2 or array.shape[1] != len(feature_names):
            raise ValueError("Unexpected regression SHAP array shape.")
        keep = meaningful_display_mask(np.mean(np.abs(array), axis=0))
        if not keep.any():
            return
        filtered_names = [
            display_names.get(str(feature), str(feature))
            for feature, selected in zip(feature_names, keep)
            if selected
        ]
        filtered_values = array[:, keep]
        sample_df = pd.DataFrame(np.asarray(X_sample)[:, keep], columns=filtered_names)
        max_display = min(20, len(filtered_names))
        shap.summary_plot(
            filtered_values,
            sample_df,
            feature_names=filtered_names,
            max_display=max_display,
            show=False,
        )
        plt.savefig(f"{prefix}_{model_name}_shap_summary_beeswarm.png")
        plt.close()

        shap.summary_plot(
            filtered_values,
            sample_df,
            feature_names=filtered_names,
            plot_type="bar",
            max_display=max_display,
            show=False,
        )
        plt.savefig(f"{prefix}_{model_name}_shap_summary_bar.png")
        plt.close()
    except Exception as exc:
        plt.close()
        print(f"Official SHAP plots skipped for {model_name}: {exc}")


def _take_row_positions(values, positions):
    """Select positional rows without treating DataFrame positions as column labels."""
    return values.iloc[positions] if isinstance(values, pd.DataFrame) else values[positions]


def _configured_positive_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)
    return value if value > 0 else int(default)


def _proportional_group_positions(groups, limit: int, rng: np.random.RandomState) -> np.ndarray:
    group_ids, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    targets = counts.astype(float) * (limit / int(counts.sum()))
    allocation = np.floor(targets).astype(int)
    if limit >= len(group_ids):
        allocation = np.maximum(allocation, 1)
    allocation = np.minimum(allocation, counts)
    fractions = targets - np.floor(targets)
    remaining = limit - int(allocation.sum())
    while remaining > 0:
        candidates = np.flatnonzero(allocation < counts)
        if not len(candidates):
            break
        chosen = candidates[np.argmax(fractions[candidates] + rng.uniform(0.0, 1e-12, len(candidates)))]
        allocation[chosen] += 1
        fractions[chosen] = -1.0
        remaining -= 1
    while remaining < 0:
        candidates = np.flatnonzero(allocation > 1)
        if not len(candidates):
            candidates = np.flatnonzero(allocation > 0)
        chosen = candidates[np.argmin(fractions[candidates])]
        allocation[chosen] -= 1
        remaining += 1
    selected = []
    for group_index, count in enumerate(allocation):
        positions = np.flatnonzero(inverse == group_index)
        if count:
            selected.extend(rng.choice(positions, size=int(count), replace=False).tolist())
    return np.sort(np.asarray(selected, dtype=int))


def _regression_distribution_positions(y, n_samples: int, limit: int, rng: np.random.RandomState) -> np.ndarray:
    """Sample regression rows proportionally across deterministic target quantiles."""
    if n_samples <= 0:
        return np.empty(0, dtype=int)
    limit = min(max(1, int(limit)), n_samples)
    if limit == n_samples:
        return np.arange(n_samples, dtype=int)
    if y is None:
        return np.sort(rng.choice(n_samples, size=limit, replace=False))
    targets = np.asarray(y, dtype=float).reshape(-1)
    if targets.shape[0] != n_samples or not np.isfinite(targets).all():
        return np.sort(rng.choice(n_samples, size=limit, replace=False))
    unique_targets = np.unique(targets)
    bin_count = min(10, limit, len(unique_targets))
    if bin_count < 2:
        return np.sort(rng.choice(n_samples, size=limit, replace=False))
    edges = np.unique(np.quantile(targets, np.linspace(0.0, 1.0, bin_count + 1)))
    if len(edges) < 3:
        return np.sort(rng.choice(n_samples, size=limit, replace=False))
    bins = np.searchsorted(edges[1:-1], targets, side="right")
    return _proportional_group_positions(bins, limit, rng)


def _regression_vae_shap_configuration(n_samples: int, n_features: int, model_name: str) -> dict[str, int]:
    cycles = (
        REGRESSION_MLPVAE_SHAP_PERMUTATION_CYCLES
        if model_name == "MLP_in_VAE_reg"
        else REGRESSION_VAE_SHAP_PERMUTATION_CYCLES
    )
    return {
        "explanation_size": min(REGRESSION_VAE_SHAP_EXPLANATION_COHORT_SIZE, n_samples),
        "background_size": min(
            _configured_positive_int("BIOMARKERML_SHAP_BACKGROUND_SIZE", REGRESSION_VAE_SHAP_DEFAULT_BACKGROUND_SIZE),
            n_samples,
        ),
        "permutation_cycles": cycles,
        "max_evals": cycles * (2 * n_features + 1),
        "batch_size": _configured_positive_int("BIOMARKERML_BATCH_SIZE", REGRESSION_VAE_SHAP_DEFAULT_BATCH_SIZE),
    }


def _regression_shap_matrix(values, n_features: int) -> np.ndarray:
    array = np.asarray(values.values if hasattr(values, "values") else values)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2 or array.shape[1] != n_features:
        raise ValueError("Unexpected scalar-regression SHAP array shape.")
    return array


def _compute_regression_vae_permutation_shap(
    final_estimator,
    X_model,
    y,
    model_name: str,
    configuration: dict[str, int] | None = None,
):
    """Compute genuine bounded permutation SHAP for scalar VAE-family regression."""
    import shap

    n_samples, n_features = X_model.shape
    config = dict(
        _regression_vae_shap_configuration(n_samples, n_features, model_name)
        if configuration is None
        else configuration
    )
    rng = np.random.RandomState(DEFAULT_RANDOM_STATE)
    sample_positions = _regression_distribution_positions(y, n_samples, config["explanation_size"], rng)
    background_positions = _regression_distribution_positions(y, n_samples, config["background_size"], rng)
    sample = _take_row_positions(X_model, sample_positions)
    background = _take_row_positions(X_model, background_positions)
    stats = {
        "explanation_size": int(len(sample_positions)),
        "background_size": int(len(background_positions)),
        "n_features": int(n_features),
        "permutation_cycles": int(config["permutation_cycles"]),
        "max_evals": int(config["max_evals"]),
        "batch_size": int(config["batch_size"]),
        "callback_invocations": 0,
        "prediction_rows": 0,
    }

    def _vae_predict(values):
        matrix = np.asarray(values)
        stats["callback_invocations"] += 1
        stats["prediction_rows"] += int(matrix.shape[0])
        predictions = np.asarray(
            final_estimator.predict(
                matrix,
                inference_batch_size=stats["batch_size"] if model_name == "MLP_in_VAE_reg" else None,
            )
        )
        if predictions.ndim == 2 and predictions.shape[1] == 1:
            predictions = predictions[:, 0]
        if predictions.ndim != 1 or predictions.shape[0] != matrix.shape[0]:
            raise ValueError("VAE regression SHAP requires one prediction per input row.")
        return predictions

    explainer = shap.PermutationExplainer(_vae_predict, background, seed=DEFAULT_RANDOM_STATE)
    explanation = explainer(
        sample,
        max_evals=stats["max_evals"],
        batch_size=stats["batch_size"],
        silent=True,
    )
    return _regression_shap_matrix(explanation, n_features), sample, background, stats, explanation


def _compute_importance(model_name: str, final_estimator, X_model: np.ndarray, y=None):
    n_features = X_model.shape[1]
    try:
        import shap
        if model_name in {"VAE_MLP_reg", "MLP_in_VAE_reg"}:
            values, sample, _, stats, _ = _compute_regression_vae_permutation_shap(
                final_estimator, X_model, y, model_name
            )
            arr = np.mean(np.abs(values), axis=0)
            print(
                "Regression VAE-family SHAP: cohort={}, background={}, features={}, cycles={}, max_evals={}, callback_rows={}.".format(
                    stats["explanation_size"],
                    stats["background_size"],
                    stats["n_features"],
                    stats["permutation_cycles"],
                    stats["max_evals"],
                    stats["prediction_rows"],
                )
            )
        else:
            sample = X_model[: min(120, X_model.shape[0])]
            if model_name in TREE_MODELS:
                values = shap.TreeExplainer(final_estimator).shap_values(sample)
                arr = np.asarray(values)
                if arr.ndim == 3:
                    arr = np.mean(np.abs(arr), axis=(0, 2))
                else:
                    arr = np.mean(np.abs(arr), axis=0)
            else:
                background = sample[: min(30, len(sample))]
                explainer = shap.PermutationExplainer(final_estimator.predict, background)
                values = explainer(sample, max_evals=max(2 * n_features + 1, 20)).values
                arr = values
                arr = np.mean(np.abs(np.asarray(arr)), axis=0)
        arr = np.ravel(arr)
        if len(arr) == n_features:
            return arr, "shap", values, sample
    except Exception as exc:
        print(f"SHAP explainer fallback for {model_name}: {exc}")
    print(f"Official SHAP plots skipped for {model_name}; estimator importance will be used for publication plots.")
    importance, source_type = estimator_importance_with_source(final_estimator, n_features)
    return np.abs(importance), source_type, None, None


def _plot_bar(
    prefix: str,
    model_name: str,
    shap_df: pd.DataFrame,
    top_n: int,
    display_names: dict[str, str],
) -> None:
    value_column = "ImportanceValue" if "ImportanceValue" in shap_df.columns else "MeanAbsSHAP"
    true_shap = bool(shap_df.get("IsTrueSHAP", pd.Series(False, index=shap_df.index)).fillna(False).all())
    plot_df = shap_df.loc[meaningful_display_mask(shap_df[value_column])].head(top_n).copy().iloc[::-1]
    if plot_df.empty:
        return
    plot_df["DisplayFeature"] = plot_df["Feature"].astype(str).map(
        lambda feature: display_names.get(feature, feature)
    )
    fig, ax = plt.subplots(figsize=(8.0, max(4.0, 0.28 * len(plot_df))))
    bars = ax.barh(plot_df["DisplayFeature"], plot_df[value_column], color="#4C78A8")
    ax.set_xlabel("Mean absolute SHAP" if true_shap else "Model-specific importance")
    ax.set_ylabel("")
    ax.set_title(f"{model_name.replace('_', ' ').title()} {'SHAP' if true_shap else 'Feature Importance'}")
    if len(plot_df) <= 30:
        max_value = float(plot_df[value_column].max()) if not plot_df.empty else 0.0
        offset = max(max_value * 0.01, 1e-9)
        for bar in bars:
            value = bar.get_width()
            ax.text(value + offset, bar.get_y() + bar.get_height() / 2, format_effect_value(value), va="center", fontsize=9)
        ax.set_xlim(0, max_value * 1.15 if max_value > 0 else 1.0)
    plt.tight_layout()
    plt.savefig(f"{prefix}_{model_name}_shap_bar.png")
    plt.close(fig)


def _plot_dot(prefix: str, model_name: str, shap_df: pd.DataFrame, top_n: int) -> None:
    plot_df = shap_df.head(top_n).copy()
    value_column = "ImportanceValue" if "ImportanceValue" in plot_df.columns else "MeanAbsSHAP"
    true_shap = bool(plot_df.get("IsTrueSHAP", pd.Series(False, index=plot_df.index)).fillna(False).all())
    fig, ax = plt.subplots(figsize=(8.0, max(4.0, 0.28 * len(plot_df))))
    ax.scatter(plot_df[value_column], np.arange(len(plot_df)), s=38, color="#4C78A8")
    ax.set_yticks(np.arange(len(plot_df)), plot_df["Feature"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean absolute SHAP" if true_shap else "Model-specific importance")
    ax.set_title(f"{model_name.replace('_', ' ').title()} {'SHAP' if true_shap else 'Importance'} Summary")
    plt.tight_layout()
    plt.savefig(f"{prefix}_{model_name}_shap_summary_dot.png")
    plt.close(fig)


def _load_vae_regression_artifact(output_prefix: str, model_name: str):
    if model_name == "MLP_in_VAE_reg":
        artifact_path = f"{output_prefix}_MLP_in_VAE_reg_final_artifact.pkl"
        if not os.path.exists(artifact_path):
            return None
        artifact = joblib.load(artifact_path)
        required = {
            "model", "preprocessor", "X_raw", "y", "model_feature_order",
            "feature_roles", "selected_biomarker_features",
            "final_hyperparameters", "architecture", "scope",
        }
        missing = sorted(required - set(artifact))
        if missing:
            raise ValueError(f"Incomplete MLPVAE regression final artifact: {missing}")
        if artifact["scope"] != "final_full_development_model":
            raise ValueError("MLPVAE regression SHAP requires the final full-development model artifact.")
        X_model = artifact["preprocessor"].transform(artifact["X_raw"])
        feature_names = list(artifact["model_feature_order"])
        if list(X_model.columns) != feature_names:
            raise ValueError("MLPVAE regression artifact feature order does not match fitted preprocessing.")
        return artifact["model"], X_model, np.asarray(artifact["y"]), feature_names
    if model_name != "VAE_MLP_reg":
        return None
    model_path = f"{output_prefix}_vae_reg_model.pkl"
    scaler_path = f"{output_prefix}_vae_reg_scaler.pkl"
    data_path = f"{output_prefix}_vae_reg_data.pkl"
    if not all(os.path.exists(path) for path in [model_path, scaler_path, data_path]):
        return None
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    data = joblib.load(data_path)
    X_raw = data[0]
    y = np.asarray(data[1])
    transformed = scaler.transform(X_raw)
    if isinstance(transformed, pd.DataFrame):
        X_model = transformed
        feature_names = list(map(str, X_model.columns))
    else:
        feature_names = list(data[2]) if len(data) >= 3 else [f"Protein_{i+1}" for i in range(np.asarray(X_raw).shape[1])]
        X_model = pd.DataFrame(transformed, columns=feature_names)
    return model, X_model, y, feature_names


def _publication_display_map(
    table: pd.DataFrame,
    num_features: int,
    *,
    species: int,
    offline: bool,
) -> dict[str, str]:
    candidates = (
        table.loc[meaningful_display_mask(table["ImportanceValue"] if "ImportanceValue" in table.columns else table["MeanAbsSHAP"])]
        .head(max(20, int(num_features)))["Feature"]
        .astype(str)
        .tolist()
    )
    return publication_gene_symbol_map(candidates, species=species, offline=offline)


def calculate_shap_values(
    model_name: str,
    num_features: int,
    output_prefix: str,
    *,
    species: int = 9606,
    offline: bool = False,
) -> None:
    model_name = _resolve_model_name(model_name)
    if model_name in {"VAE_MLP_reg", "MLP_in_VAE_reg"}:
        artifact = _load_vae_regression_artifact(output_prefix, model_name)
        if artifact is None:
            print(f"Skipping {model_name}: missing regression VAE artifacts.")
            return
        final_estimator, X_model, y, feature_names = artifact
        source = "original_proteins"
        importance, importance_source, raw_values, shap_sample = _compute_importance(model_name, final_estimator, X_model, y=y)
        is_true_shap = str(importance_source).casefold() == "shap"
        values = np.asarray(importance, dtype=float)
        df = pd.DataFrame({
            "Feature": feature_names,
            "ImportanceValue": values,
            "ImportanceType": "SHAP" if is_true_shap else str(importance_source),
            "IsTrueSHAP": is_true_shap,
            "MeanAbsSHAP": values if is_true_shap else np.nan,
            "Model": model_name,
            "Task": "regression",
            "FeatureSource": source,
            "Fold": "Full",
            "ClassName": "Global",
            "IsProteinLevel": True,
            "Importance_Source": importance_source,
        }).sort_values("ImportanceValue", ascending=False).reset_index(drop=True)
        # Covariate attributions remain in the final-model explanation but are not biomarker candidates.
        artifact_path = f"{output_prefix}_{model_name}_final_artifact.pkl"
        if os.path.exists(artifact_path):
            role_map = joblib.load(artifact_path)["preprocessor"].feature_roles_out_
            df["FeatureRole"] = df["Feature"].map(role_map).fillna("BIOMARKER")
        else:
            df["FeatureRole"] = "BIOMARKER"
        df.to_csv(f"{output_prefix}_{model_name}_shap_values.csv", index=False)
        top = df.loc[df["FeatureRole"] == "BIOMARKER"].head(min(num_features, len(df))).copy()
        top.insert(1, "Rank", np.arange(1, len(top) + 1))
        top.to_csv(f"{output_prefix}_{model_name}_top_shap_biomarkers.csv", index=False)
        display_names = _publication_display_map(
            df,
            num_features,
            species=species,
            offline=offline,
        )
        if importance_source == "shap":
            _save_official_shap_plots(output_prefix, model_name, raw_values, shap_sample, feature_names, display_names)
        _plot_bar(output_prefix, model_name, df, num_features, display_names)
        _plot_dot(output_prefix, model_name, df, num_features)
        print(f"Finished SHAP for {model_name}")
        return
    model_path = f"{output_prefix}_{model_name}_best_model.pkl"
    x_path = f"{output_prefix}_{model_name}_X.npy"
    feature_path = f"{output_prefix}_{model_name}_feature_names.npy"
    data_path = f"{output_prefix}_{model_name}_data.pkl"
    if not os.path.exists(model_path) or (not os.path.exists(data_path) and not all(os.path.exists(path) for path in [x_path, feature_path])):
        print(f"Skipping {model_name}: missing regression artifacts.")
        return
    model = joblib.load(model_path)
    if os.path.exists(data_path):
        X, _, roles = joblib.load(data_path)
    else:
        features = np.load(feature_path, allow_pickle=True).astype(str)
        X = pd.DataFrame(np.load(x_path, allow_pickle=True), columns=features)
        roles = None
    X_model, feature_names, final_estimator, source = _prepare_features(model, X)
    importance, importance_source, raw_values, shap_sample = _compute_importance(model_name, final_estimator, X_model)
    is_true_shap = str(importance_source).casefold() == "shap"
    values = np.asarray(importance, dtype=float)
    df = pd.DataFrame({
        "Feature": feature_names,
        "ImportanceValue": values,
        "ImportanceType": "SHAP" if is_true_shap else str(importance_source),
        "IsTrueSHAP": is_true_shap,
        "MeanAbsSHAP": values if is_true_shap else np.nan,
        "Model": model_name,
        "Task": "regression",
        "FeatureSource": source,
        "Fold": "Full",
        "ClassName": "Global",
        "IsProteinLevel": True,
        "Importance_Source": importance_source,
    }).sort_values("ImportanceValue", ascending=False).reset_index(drop=True)
    preprocessor = model.named_steps.get("preprocessor") if isinstance(model, Pipeline) else None
    role_map = preprocessor.feature_roles_out_ if preprocessor is not None else {}
    df["FeatureRole"] = df["Feature"].map(role_map).fillna("BIOMARKER")
    df.to_csv(f"{output_prefix}_{model_name}_shap_values.csv", index=False)
    top = df.loc[df["FeatureRole"] == "BIOMARKER"].head(min(num_features, len(df))).copy()
    top.insert(1, "Rank", np.arange(1, len(top) + 1))
    top.to_csv(f"{output_prefix}_{model_name}_top_shap_biomarkers.csv", index=False)
    display_names = _publication_display_map(
        df,
        num_features,
        species=species,
        offline=offline,
    )
    if importance_source == "shap":
        _save_official_shap_plots(output_prefix, model_name, raw_values, shap_sample, feature_names, display_names)
    _plot_bar(output_prefix, model_name, df, num_features, display_names)
    _plot_dot(output_prefix, model_name, df, num_features)
    print(f"Finished SHAP for {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Protein-level SHAP analysis for BiomarkerML regression models.")
    parser.add_argument("--input", default=None, help="Task input directory containing fitted model artifacts.")
    parser.add_argument("--p", "-p", default=None, dest="prefix")
    parser.add_argument("--outdir", default=".", help="Base output directory")
    parser.add_argument("--feature_selection", choices=["yes", "no"], default="no", help="Accepted for CLI consistency")
    parser.add_argument("--jobs", type=int, default=1, help="Accepted for CLI consistency")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--m", "-m", nargs="+", required=True, dest="models")
    parser.add_argument("--f", "-f", type=int, default=20, dest="num_features")
    args = parser.parse_args()
    prefix = build_output_prefix(args.input, args.prefix)
    set_global_seed(args.seed)
    outdir = task_output_dir(args.outdir, "regression_shap")
    stage_task_inputs(args.input, outdir, prefix)
    with task_run_context(args.outdir, "regression_shap"):
        for model_name in args.models:
            calculate_shap_values(model_name, args.num_features, prefix)
        try:
            from engines.reliability.biomarker_reliability import run_publication_biomarker_analysis

            run_publication_biomarker_analysis(prefix, task="regression")
        except Exception as exc:
            print(f"Publication post-processing skipped: {exc}")


if __name__ == "__main__":
    main()
