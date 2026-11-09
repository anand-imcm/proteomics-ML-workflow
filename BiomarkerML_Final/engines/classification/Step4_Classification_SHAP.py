from __future__ import annotations

import argparse
import os
import re
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
    plot_average_shap_per_class,
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
    "KNN": "knn",
    "RF": "random_forest",
    "NN": "neural_network",
    "SVM": "svm",
    "XGB": "xgboost",
    "PLSDA": "plsda",
    "VAE": "vae",
    "LGBM": "lightgbm",
    "LR": "logistic_regression",
    "MLPVAE": "vaemlp",
    "GNB": "gaussiannb",
}
TREE_MODELS = {"random_forest", "xgboost", "lightgbm"}
LINEAR_MODELS = {"logistic_regression"}

# Bounded, calibrated permutation-SHAP policy for costly Torch VAE classifiers.
VAE_SHAP_EXPLANATION_COHORT_SIZE = 100
VAE_SHAP_DEFAULT_BACKGROUND_SIZE = 50
VAE_SHAP_PERMUTATION_CYCLES = 5
MLPVAE_SHAP_PERMUTATION_CYCLES = 3
VAE_SHAP_DEFAULT_BATCH_SIZE = 64


def _resolve_model_name(name: str) -> str:
    for alias, value in MODEL_ALIASES.items():
        if name.casefold() == alias.casefold() or name.casefold() == value.casefold():
            return value
    return name


def _load_artifact(prefix: str, model_name: str):
    if model_name == "vaemlp":
        final_path = f"{prefix}_vaemlp_final_artifact.pkl"
        if os.path.exists(final_path):
            artifact = joblib.load(final_path)
            required = {
                "model", "preprocessor", "X_raw", "y", "label_encoder",
                "model_feature_order", "feature_roles", "selected_biomarker_features",
                "final_hyperparameters", "architecture", "scope",
            }
            missing = sorted(required - set(artifact))
            if missing:
                raise ValueError(f"Incomplete MLPVAE classification final artifact: {missing}")
            if artifact["scope"] != "final_full_development_model":
                raise ValueError("MLPVAE classification SHAP requires the final full-development model artifact.")
            X_raw = artifact["X_raw"]
            y = artifact["y"]
            encoder = artifact["label_encoder"]
            X_model = artifact["preprocessor"].transform(X_raw)
            if list(X_model.columns) != list(artifact["model_feature_order"]):
                raise ValueError("MLPVAE classification artifact feature order mismatch.")
            return artifact["model"], X_model, np.asarray(y), encoder
    model_path = f"{prefix}_{model_name}_model.pkl"
    data_path = f"{prefix}_{model_name}_data.pkl"
    if not os.path.exists(model_path) or not os.path.exists(data_path):
        print(f"Skipping {model_name}: missing model or data artifact.")
        return None
    model = joblib.load(model_path)
    data = joblib.load(data_path)
    X = data[0]
    y = np.asarray(data[1])
    encoder = data[2] if len(data) >= 3 else None
    feature_names = data[3] if len(data) >= 4 else None
    if not isinstance(X, pd.DataFrame):
        names = [str(name) for name in feature_names] if feature_names is not None else [f"Protein_{i+1}" for i in range(np.asarray(X).shape[1])]
        X = pd.DataFrame(np.asarray(X), columns=names)
    scaler_path = f"{prefix}_{model_name}_scaler.pkl"
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        transformed = scaler.transform(X)
        X = transformed if isinstance(transformed, pd.DataFrame) else pd.DataFrame(transformed, columns=list(X.columns))
    return model, X, y, encoder


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
    assert_protein_level_features(names, "Classification SHAP")
    source = "selected_biomarkers_and_covariates" if preprocessor is not None else ("selected_proteins" if "feature_selection" in model.named_steps else "original_proteins")
    final_estimator = model.steps[-1][1]
    return prediction_input_with_feature_names(matrix, names, final_estimator), names, final_estimator, source


def _class_names(encoder, final_estimator, n_classes: int) -> list[str]:
    if encoder is not None and hasattr(encoder, "classes_"):
        return [str(x) for x in encoder.classes_]
    if hasattr(final_estimator, "classes_"):
        return [str(x) for x in final_estimator.classes_]
    return [str(i) for i in range(n_classes)]


def _mean_abs_from_shap_values(values, n_features: int, class_names: list[str]):
    if isinstance(values, list):
        arrays = [np.asarray(v).reshape(-1, n_features) for v in values]
        class_means = {f"Mean SHAP Value for {class_names[i] if i < len(class_names) else i}": np.mean(np.abs(arr), axis=0) for i, arr in enumerate(arrays)}
        global_mean = np.mean(np.vstack([np.abs(arr) for arr in arrays]), axis=0)
        return class_means, global_mean
    arr = np.asarray(values)
    if arr.ndim == 3:
        if arr.shape[1] == n_features:
            class_axis = 2
            arrays = [arr[:, :, i] for i in range(arr.shape[class_axis])]
        else:
            arrays = [arr[:, i, :] for i in range(arr.shape[1])]
        class_means = {f"Mean SHAP Value for {class_names[i] if i < len(class_names) else i}": np.mean(np.abs(a), axis=0) for i, a in enumerate(arrays)}
        global_mean = np.mean(np.stack([np.abs(a) for a in arrays]), axis=(0, 1))
        return class_means, global_mean
    if arr.ndim == 2 and arr.shape[1] == n_features:
        global_mean = np.mean(np.abs(arr), axis=0)
        return {"Mean SHAP Value for Global": global_mean}, global_mean
    raise ValueError("Unexpected SHAP array shape.")


def _summary_plot_values(values, n_features: int):
    if isinstance(values, list):
        return [np.asarray(v).reshape(-1, n_features) for v in values]
    arr = np.asarray(values)
    if arr.ndim == 3:
        if arr.shape[1] == n_features:
            return [arr[:, :, i] for i in range(arr.shape[2])]
        if arr.shape[2] == n_features:
            return [arr[:, i, :] for i in range(arr.shape[1])]
    return arr


def _signed_class_arrays_from_shap_values(values, n_features: int, class_names: list[str]) -> list[tuple[str, np.ndarray]]:
    if isinstance(values, list):
        arrays = [np.asarray(v).reshape(-1, n_features) for v in values]
    else:
        arr = np.asarray(values)
        if arr.ndim == 3:
            if arr.shape[1] == n_features:
                arrays = [arr[:, :, i] for i in range(arr.shape[2])]
            elif arr.shape[2] == n_features:
                arrays = [arr[:, i, :] for i in range(arr.shape[1])]
            else:
                return []
        elif arr.ndim == 2 and arr.shape[1] == n_features:
            arrays = [arr]
        else:
            return []

    if len(arrays) == 1 and len(class_names) == 2:
        arrays = [-arrays[0], arrays[0]]
    if len(arrays) != len(class_names):
        return []
    return [(class_names[i], arrays[i]) for i in range(len(class_names))]


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

        plot_values = _summary_plot_values(values, len(feature_names))
        arrays = plot_values if isinstance(plot_values, list) else [np.asarray(plot_values)]
        mean_abs = np.mean(
            np.stack([np.mean(np.abs(np.asarray(array)), axis=0) for array in arrays]),
            axis=0,
        )
        keep = meaningful_display_mask(mean_abs)
        if not keep.any():
            return
        filtered_names = [
            display_names.get(str(feature), str(feature))
            for feature, selected in zip(feature_names, keep)
            if selected
        ]
        filtered_values = [np.asarray(array)[:, keep] for array in arrays]
        beeswarm_values = filtered_values[-1]
        sample_df = pd.DataFrame(np.asarray(X_sample)[:, keep], columns=filtered_names)
        max_display = min(20, len(filtered_names))
        shap.summary_plot(
            beeswarm_values,
            sample_df,
            feature_names=filtered_names,
            max_display=max_display,
            show=False,
        )
        plt.savefig(f"{prefix}_{model_name}_shap_summary_beeswarm.png")
        plt.close()

        bar_values = filtered_values if isinstance(plot_values, list) else filtered_values[0]
        shap.summary_plot(
            bar_values,
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


def _representative_row_positions(y, n_samples: int, limit: int, rng: np.random.RandomState) -> np.ndarray:
    """Select a deterministic empirical-distribution cohort by row position."""
    if n_samples <= 0:
        return np.empty(0, dtype=int)
    limit = min(max(1, int(limit)), n_samples)
    if limit == n_samples:
        return np.arange(n_samples, dtype=int)
    if y is None:
        return np.sort(rng.choice(n_samples, size=limit, replace=False))

    labels = np.asarray(y).reshape(-1)
    if labels.shape[0] != n_samples:
        raise ValueError("SHAP labels must align with the explanation matrix.")
    classes, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    targets = counts.astype(float) * (limit / n_samples)
    allocation = np.floor(targets).astype(int)
    if limit >= len(classes):
        allocation = np.maximum(allocation, 1)
    allocation = np.minimum(allocation, counts)
    remaining = limit - int(allocation.sum())
    fractions = targets - np.floor(targets)
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
    for class_index, count in enumerate(allocation):
        positions = np.flatnonzero(inverse == class_index)
        if count:
            selected.extend(rng.choice(positions, size=int(count), replace=False).tolist())
    return np.sort(np.asarray(selected, dtype=int))


def _vae_shap_configuration(n_samples: int, n_features: int, model_name: str = "vae") -> dict[str, int]:
    background_size = min(
        _configured_positive_int("BIOMARKERML_SHAP_BACKGROUND_SIZE", VAE_SHAP_DEFAULT_BACKGROUND_SIZE),
        n_samples,
    )
    explanation_size = min(VAE_SHAP_EXPLANATION_COHORT_SIZE, n_samples)
    cycles = MLPVAE_SHAP_PERMUTATION_CYCLES if model_name == "vaemlp" else VAE_SHAP_PERMUTATION_CYCLES
    return {
        "explanation_size": explanation_size,
        "background_size": background_size,
        "permutation_cycles": cycles,
        "max_evals": cycles * (2 * n_features + 1),
        "batch_size": _configured_positive_int("BIOMARKERML_BATCH_SIZE", VAE_SHAP_DEFAULT_BATCH_SIZE),
    }


def _compute_vae_permutation_shap(
    final_estimator,
    X_model,
    y,
    configuration: dict[str, int] | None = None,
    *,
    model_name: str = "vae",
):
    """Compute genuine bounded permutation SHAP for the fitted VAE-family predictor."""
    import shap

    n_samples, n_features = X_model.shape
    config = dict(_vae_shap_configuration(n_samples, n_features, model_name) if configuration is None else configuration)
    rng = np.random.RandomState(DEFAULT_RANDOM_STATE)
    sample_positions = _representative_row_positions(y, n_samples, config["explanation_size"], rng)
    background_positions = _representative_row_positions(y, n_samples, config["background_size"], rng)
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

    def _vae_predict_proba(values):
        # SHAP keeps DataFrame metadata; the Torch wrapper consumes its numerical matrix.
        matrix = np.asarray(values)
        stats["callback_invocations"] += 1
        stats["prediction_rows"] += int(matrix.shape[0])
        return final_estimator.predict_proba(
            matrix,
            inference_batch_size=stats["batch_size"] if model_name == "vaemlp" else None,
        )

    explainer = shap.PermutationExplainer(
        _vae_predict_proba,
        background,
        seed=DEFAULT_RANDOM_STATE,
    )
    explanation = explainer(
        sample,
        max_evals=stats["max_evals"],
        batch_size=stats["batch_size"],
        silent=True,
    )
    return explanation.values, sample, background, stats, explanation


def _compute_importance(model_name: str, final_estimator, X_model: np.ndarray, class_names: list[str], y=None):
    n_features = X_model.shape[1]
    try:
        import shap

        if model_name in {"vae", "vaemlp"}:
            values, sample, _, stats, _ = _compute_vae_permutation_shap(final_estimator, X_model, y, model_name=model_name)
            print(
                "VAE-family SHAP: cohort={}, background={}, features={}, cycles={}, max_evals={}, callback_rows={}.".format(
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
            elif model_name in LINEAR_MODELS:
                values = shap.LinearExplainer(final_estimator, sample).shap_values(sample)
            else:
                background = sample[: min(30, len(sample))]
                explainer = shap.PermutationExplainer(final_estimator.predict_proba, background)
                values = explainer(sample, max_evals=max(2 * n_features + 1, 20)).values
        class_means, global_mean = _mean_abs_from_shap_values(values, n_features, class_names)
        return class_means, global_mean, "shap", values, sample
    except Exception as exc:
        print(f"SHAP explainer fallback for {model_name}: {exc}")
        print(f"Official SHAP plots skipped for {model_name}; estimator importance will be used for publication bar plots.")
        importance, source_type = estimator_importance_with_source(final_estimator, n_features)
        importance = np.abs(importance)
        return {"Model-specific importance": importance}, importance, source_type, None, None


def _write_shap_outputs(prefix: str, model_name: str, feature_names: list[str], class_means: dict[str, np.ndarray], global_mean: np.ndarray, source: str, importance_source: str, top_n: int, feature_roles: dict[str, str] | None = None) -> pd.DataFrame:
    assert_protein_level_features(feature_names, f"{model_name} SHAP")
    is_true_shap = str(importance_source).casefold() == "shap"
    values = np.asarray(global_mean, dtype=float)
    df = pd.DataFrame({
        "Feature": feature_names,
        "ImportanceValue": values,
        "ImportanceType": "SHAP" if is_true_shap else str(importance_source),
        "IsTrueSHAP": is_true_shap,
        "MeanAbsSHAP": values if is_true_shap else np.nan,
        "Model": model_name,
        "Task": "classification",
        "FeatureSource": source,
        "Fold": "Full",
        "ClassName": "Global",
        "IsProteinLevel": True,
        "Importance_Source": importance_source,
    }).sort_values("ImportanceValue", ascending=False).reset_index(drop=True)
    df["FeatureRole"] = df["Feature"].map(feature_roles or {}).fillna("BIOMARKER")
    df.to_csv(f"{prefix}_{model_name}_shap_values.csv", index=False)
    top = df.loc[df["FeatureRole"] == "BIOMARKER"].head(min(top_n, len(df))).copy()
    top.insert(1, "Rank", np.arange(1, len(top) + 1))
    top.to_csv(f"{prefix}_{model_name}_top_shap_biomarkers.csv", index=False)
    return df


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


def _plot_radar(prefix: str, model_name: str, shap_df: pd.DataFrame, top_n: int) -> None:
    plot_df = shap_df.head(min(top_n, 12)).copy()
    if plot_df.empty:
        return
    labels = plot_df["Feature"].astype(str).tolist()
    value_column = "ImportanceValue" if "ImportanceValue" in plot_df.columns else "MeanAbsSHAP"
    true_shap = bool(plot_df.get("IsTrueSHAP", pd.Series(False, index=plot_df.index)).fillna(False).all())
    values = plot_df[value_column].to_numpy(dtype=float)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values = np.concatenate([values, values[:1]])
    angles = angles + angles[:1]
    fig, ax = plt.subplots(figsize=(7.0, 6.2), subplot_kw={"polar": True})
    ax.plot(angles, values, lw=2, color="#4C78A8")
    ax.fill(angles, values, color="#4C78A8", alpha=0.22)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(f"{model_name.replace('_', ' ').title()} {'SHAP' if true_shap else 'Importance'} Radar")
    plt.tight_layout()
    plt.savefig(f"{prefix}_{model_name}_shap_radar.png")
    plt.close(fig)


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return cleaned or "class"


def _direction(value: float) -> str:
    if value > 1e-12:
        return "Positive"
    if value < -1e-12:
        return "Negative"
    return "Neutral"


def _global_shap_rank_lookup(prefix: str, model_name: str) -> dict[str, dict[str, float]]:
    path = Path(f"{prefix}_{model_name}_shap_values.csv")
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if "Feature" not in df.columns or "MeanAbsSHAP" not in df.columns:
        return {}
    global_df = df[["Feature", "MeanAbsSHAP"]].copy()
    global_df["Feature"] = global_df["Feature"].astype(str)
    global_df["GlobalRank"] = np.arange(1, len(global_df) + 1)
    global_df = global_df.rename(columns={"MeanAbsSHAP": "GlobalMeanAbsSHAP"})
    return global_df.drop_duplicates("Feature").set_index("Feature")[["GlobalRank", "GlobalMeanAbsSHAP"]].to_dict("index")


def _write_per_class_shap_outputs(
    prefix: str,
    model_name: str,
    feature_names: list[str],
    class_names: list[str],
    raw_values,
    source: str,
    importance_source: str,
    top_n: int,
    display_names: dict[str, str],
) -> None:
    class_arrays = _signed_class_arrays_from_shap_values(raw_values, len(feature_names), class_names)
    if not class_arrays:
        print(f"Per-class SHAP skipped for {model_name}: class-specific SHAP values were unavailable.")
        return

    global_lookup = _global_shap_rank_lookup(prefix, model_name)
    rows = []
    for class_name, values in class_arrays:
        mean_shap = np.mean(values, axis=0)
        mean_abs = np.mean(np.abs(values), axis=0)
        order = np.argsort(-mean_abs)
        ranks = np.empty(len(order), dtype=int)
        ranks[order] = np.arange(1, len(order) + 1)
        for feature, signed_value, abs_value, rank in zip(feature_names, mean_shap, mean_abs, ranks):
            global_info = global_lookup.get(str(feature), {})
            rows.append(
                {
                    "Feature": feature,
                    "GlobalRank": global_info.get("GlobalRank", pd.NA),
                    "GlobalMeanAbsSHAP": global_info.get("GlobalMeanAbsSHAP", np.nan),
                    "ClassName": class_name,
                    "RankWithinClass": int(rank),
                    "MeanSHAP": float(signed_value),
                    "MeanAbsSHAP": float(abs_value),
                    "Direction": _direction(float(signed_value)),
                    "ImportanceSource": importance_source,
                    "FeatureSource": source,
                }
            )
    columns = [
        "Feature",
        "GlobalRank",
        "GlobalMeanAbsSHAP",
        "ClassName",
        "RankWithinClass",
        "MeanSHAP",
        "MeanAbsSHAP",
        "Direction",
        "ImportanceSource",
        "FeatureSource",
    ]
    df = pd.DataFrame(rows)[columns]
    df = df.sort_values(["ClassName", "RankWithinClass"]).reset_index(drop=True)
    df["DisplayFeature"] = df["Feature"].astype(str).map(
        lambda feature: display_names.get(feature, feature)
    )
    df.to_csv(f"{prefix}_{model_name}_per_class_shap.csv", index=False)

    for class_name in class_names:
        class_df = df[df["ClassName"] == class_name].sort_values("RankWithinClass")
        plot_average_shap_per_class(
            class_df,
            class_name,
            f"{prefix}_{model_name}_per_class_shap_{_safe_name(class_name)}.png",
            top_n=top_n,
        )


def _publication_display_map(
    shap_df: pd.DataFrame,
    feature_names: list[str],
    class_names: list[str],
    raw_values,
    top_n: int,
    *,
    species: int,
    offline: bool,
) -> dict[str, str]:
    candidates = (
        shap_df.loc[meaningful_display_mask(shap_df["ImportanceValue"] if "ImportanceValue" in shap_df.columns else shap_df["MeanAbsSHAP"])]
        .head(max(20, int(top_n)))["Feature"]
        .astype(str)
        .tolist()
    )
    if raw_values is not None:
        for _, values in _signed_class_arrays_from_shap_values(raw_values, len(feature_names), class_names):
            mean_abs = np.mean(np.abs(values), axis=0)
            valid = np.flatnonzero(meaningful_display_mask(mean_abs))
            ranked = valid[np.argsort(-mean_abs[valid])[: max(20, int(top_n))]]
            candidates.extend(str(feature_names[index]) for index in ranked)
    return publication_gene_symbol_map(
        candidates,
        species=species,
        offline=offline,
    )


def process_model(
    prefix: str,
    model_name: str,
    top_n: int,
    *,
    species: int = 9606,
    offline: bool = False,
) -> None:
    model_name = _resolve_model_name(model_name)
    artifact = _load_artifact(prefix, model_name)
    if artifact is None:
        return
    model, X, y, encoder = artifact
    X_model, feature_names, final_estimator, source = _prepare_features(model, X)
    class_names = _class_names(encoder, final_estimator, len(np.unique(y)))
    class_means, global_mean, importance_source, raw_values, shap_sample = _compute_importance(model_name, final_estimator, X_model, class_names, y=y)
    preprocessor = model.named_steps.get("preprocessor") if isinstance(model, Pipeline) else None
    feature_roles = preprocessor.feature_roles_out_ if preprocessor is not None else None
    artifact_path = f"{prefix}_{model_name}_final_artifact.pkl"
    if model_name in {"vae", "vaemlp"} and os.path.exists(artifact_path):
        feature_roles = joblib.load(artifact_path)["preprocessor"].feature_roles_out_
    shap_df = _write_shap_outputs(prefix, model_name, feature_names, class_means, global_mean, source, importance_source, top_n, feature_roles)
    display_names = _publication_display_map(
        shap_df,
        feature_names,
        class_names,
        raw_values,
        top_n,
        species=species,
        offline=offline,
    )
    if importance_source == "shap":
        _save_official_shap_plots(prefix, model_name, raw_values, shap_sample, feature_names, display_names)
        _write_per_class_shap_outputs(
            prefix,
            model_name,
            feature_names,
            class_names,
            raw_values,
            source,
            importance_source,
            top_n,
            display_names,
        )
    _plot_bar(prefix, model_name, shap_df, top_n, display_names)
    _plot_radar(prefix, model_name, shap_df, top_n)
    print(f"Finished SHAP for {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Protein-level SHAP analysis for BiomarkerML classification models.")
    parser.add_argument("-m", "--models", nargs="+", required=True)
    parser.add_argument("--input", default=None, help="Task input directory containing fitted model artifacts.")
    parser.add_argument("-p", "--prefix", default=None)
    parser.add_argument("--outdir", default=".", help="Base output directory")
    parser.add_argument("--feature_selection", choices=["yes", "no"], default="no", help="Accepted for CLI consistency")
    parser.add_argument("--jobs", type=int, default=1, help="Accepted for CLI consistency")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("-f", "--num_features", type=int, default=20)
    args = parser.parse_args()
    prefix = build_output_prefix(args.input, args.prefix)
    set_global_seed(args.seed)
    outdir = task_output_dir(args.outdir, "classification_shap")
    stage_task_inputs(args.input, outdir, prefix)
    with task_run_context(args.outdir, "classification_shap"):
        for model in args.models:
            process_model(prefix, model, args.num_features)
        try:
            from engines.reliability.biomarker_reliability import run_publication_biomarker_analysis

            run_publication_biomarker_analysis(prefix, task="classification")
        except Exception as exc:
            print(f"Publication post-processing skipped: {exc}")


if __name__ == "__main__":
    main()
