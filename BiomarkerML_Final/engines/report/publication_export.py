from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from core.publication_utils import (
    configure_publication_plots,
    format_effect_value,
    meaningful_display_mask,
    style_publication_axis,
)
from engines.ppi.protein_network import publication_gene_symbol_map
from engines.reliability.biomarker_reliability import (
    recommendation_score_definition,
    recommendation_score_mathtext,
)


MODEL_EXPORT_NAMES = {
    "RF": "RandomForest",
    "XGB": "XGBoost",
    "LGBM": "LightGBM",
    "LR": "LogisticRegression",
    "SVM": "SVM",
    "KNN": "KNN",
    "GNB": "GaussianNaiveBayes",
    "NN": "NeuralNetwork",
    "PLSDA": "PLSDA",
    "VAE": "VAE",
    "MLPVAE": "MLPInVAE",
}

MODEL_STABILITY_KEYS = {
    "RF": "random_forest",
    "XGB": "xgboost",
    "LGBM": "lightgbm",
    "LR": "logistic_regression",
    "SVM": "svm",
    "KNN": "knn",
    "GNB": "gaussiannb",
    "NN": "neural_network",
    "PLSDA": "plsda",
    "VAE": "vae",
    "MLPVAE": "vaemlp",
    "RF_reg": "random_forest_reg",
    "XGB_reg": "xgboost_reg",
    "LGBM_reg": "lightgbm_reg",
    "SVM_reg": "svm_reg",
    "KNN_reg": "knn_reg",
    "NN_reg": "neural_network_reg",
    "PLS_reg": "pls_reg",
    "VAE_reg": "vae_reg",
    "MLPVAE_reg": "vaemlp_reg",
}

FORBIDDEN_FINAL_PATTERNS = (
    r"optuna",
    r"(?:^|[/_])fold\d+",
    r"(?:^|/)cache(?:/|$)",
    r"(?:^|/)temp(?:/|$)",
    r"\.log$",
    r"training[_ -]?history",
    r"epoch",
    r"\.pkl$",
    r"\.npy$",
    r"intermediate",
)


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _source_file(directory: Path, prefix: str, suffixes: tuple[str, ...]) -> Path | None:
    if not directory.exists():
        return None
    candidates: list[Path] = []
    for suffix in suffixes:
        candidates.extend(directory.glob(f"*{suffix}"))
    if not candidates:
        return None
    return sorted(
        set(candidates),
        key=lambda path: (
            0 if path.name.lower().startswith(prefix.lower() + "_") else 1,
            len(path.name),
            path.name.lower(),
        ),
    )[0]


def _export(source: Path | None, destination: Path) -> bool:
    if source is None or not source.is_file():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _meaningful_rows(table: pd.DataFrame, value_columns: tuple[str, ...]) -> pd.DataFrame:
    if table.empty:
        return table.copy()
    available = [column for column in value_columns if column in table.columns]
    if not available:
        return table.copy()
    keep = np.zeros(len(table), dtype=bool)
    for column in available:
        keep |= meaningful_display_mask(table[column])
    return table.loc[keep].copy()


def audit_diagnostic_separation(model_dir: Path, prefix: str, target: Path | None = None) -> None:
    """Reject publication diagnostics that reuse one source or one rendered figure."""
    learning_csv = _source_file(model_dir, prefix, ("_learning_curve.csv",))
    training_csv = _source_file(model_dir, prefix, ("_training_size_curve.csv",))
    learning_png = _source_file(model_dir, prefix, ("_learning_curve.png",))
    training_png = _source_file(model_dir, prefix, ("_training_size_curve.png",))

    pairs = [
        ("CSV", learning_csv, training_csv),
        ("figure", learning_png, training_png),
    ]
    for artifact_type, learning, training in pairs:
        if learning is None or training is None:
            continue
        if learning.resolve() == training.resolve():
            raise RuntimeError(
                f"{model_dir.name}: Learning Curve and Training Size {artifact_type} share one source."
            )
        if _file_digest(learning) == _file_digest(training):
            raise RuntimeError(
                f"{model_dir.name}: Learning Curve and Training Size {artifact_type} are byte-identical."
            )

    if learning_csv is not None and training_csv is not None:
        learning = _read_csv(learning_csv)
        training = _read_csv(training_csv)
        learning_columns = {"ProgressionStep", "ProgressionValue", "TrainingLoss", "ValidationLoss"}
        training_schemas = (
            {"Training_Size", "Training_Score_Loss", "Validation_Score_Loss"},
            {"TrainingFraction", "TrainingSamples", "TrainingScore", "ValidationScore"},
        )
        if not learning_columns.issubset(learning.columns):
            raise RuntimeError(f"{model_dir.name}: Learning Curve is missing progression-specific data.")
        if not any(schema.issubset(training.columns) for schema in training_schemas):
            raise RuntimeError(f"{model_dir.name}: Training Size is missing sample-size data.")

    if target is not None:
        learning_final = target / "Learning_Curve.png"
        training_final = target / "Training_Size_Diagnostic.png"
        if learning_final.exists() and training_final.exists():
            if _file_digest(learning_final) == _file_digest(training_final):
                raise RuntimeError(
                    f"{model_dir.name}: Final Learning Curve and Training Size figures are identical."
                )


def _remove_plot_sidecars(output: Path) -> None:
    for suffix in (".pdf", ".tiff"):
        sidecar = output.with_suffix(suffix)
        if sidecar.exists():
            sidecar.unlink()


def _model_export_name(model_dir: Path, task: str) -> str:
    base = MODEL_EXPORT_NAMES.get(model_dir.name, model_dir.name.replace("_", ""))
    return f"{base}Regression" if task == "regression" and not base.lower().endswith("regression") else base


def _model_sources(run_root: Path, task: str) -> list[Path]:
    directory = run_root / task
    if not directory.exists():
        return []
    return sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_dir() and not path.name.startswith(".") and any(path.glob("*.png"))
        ],
        key=lambda path: path.name,
    )


def _biomarker_display_labels(
    table: pd.DataFrame,
    display_names: dict[str, str],
    protein_column: str = "Protein",
) -> list[str]:
    """Return canonical publication labels without altering source identifiers."""
    if protein_column not in table.columns:
        return []
    return [
        display_names.get(str(protein), str(protein))
        for protein in table[protein_column].astype(str)
    ]


def _plot_publication_biomarker_bars(
    table: pd.DataFrame,
    value_column: str,
    output: Path,
    display_names: dict[str, str],
    *,
    title: str,
    subtitle: str,
    x_label: str,
    top: int = 30,
    annotation_columns: tuple[str, str] | None = None,
) -> None:
    """Render existing biomarker scores with publication-only display names."""
    if table.empty or not {"Protein", value_column}.issubset(table.columns):
        return
    configure_publication_plots()
    import matplotlib.pyplot as plt

    view = table.copy()
    view[value_column] = pd.to_numeric(view[value_column], errors="coerce")
    view = view.dropna(subset=[value_column]).sort_values(value_column, ascending=False).head(top)
    if view.empty:
        return
    view = view.iloc[::-1]
    labels = _biomarker_display_labels(view, display_names)
    values = view[value_column].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.4, max(4.4, 0.30 * len(view) + 1.8)))
    bars = ax.barh(labels, values, color="#4C78A8", height=0.68)
    max_value = float(np.nanmax(np.abs(values))) if len(values) else 0.0
    offset = max(max_value * 0.012, 1e-9)
    for index, (bar, value) in enumerate(zip(bars, values)):
        annotation = format_effect_value(value)
        if annotation_columns is not None:
            count_column, frequency_column = annotation_columns
            count_value = pd.to_numeric(
                view.iloc[index].get(count_column, 0),
                errors="coerce",
            )
            frequency_value = pd.to_numeric(
                view.iloc[index].get(frequency_column, 0.0),
                errors="coerce",
            )
            count = 0 if pd.isna(count_value) else int(count_value)
            frequency = 0.0 if pd.isna(frequency_value) else float(frequency_value)
            annotation = f"{annotation} | {count} models ({frequency:.0%})"
        ax.text(
            value + offset,
            bar.get_y() + bar.get_height() / 2,
            annotation,
            va="center",
            fontsize=8.2,
            color="#374151",
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel("")
    ax.set_xlim(0, max_value * (1.42 if annotation_columns else 1.20) if max_value > 0 else 1.0)
    fig.suptitle(title, y=0.985, fontsize=14, fontweight="bold")
    fig.text(0.125, 0.935, subtitle, ha="left", va="top", fontsize=9)
    style_publication_axis(ax)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(output)
    plt.close(fig)
    _remove_plot_sidecars(output)


def _model_stability_source(run_root: Path, prefix: str, model_dir: Path) -> Path | None:
    key = MODEL_STABILITY_KEYS.get(model_dir.name, model_dir.name)
    return _source_file(
        run_root / "biomarker_reliability",
        prefix,
        (f"_{key}_top_marker_stability.csv",),
    )


def _export_model(
    run_root: Path,
    final_root: Path,
    prefix: str,
    task: str,
    model_dir: Path,
    display_names: dict[str, str],
) -> None:
    target = final_root / _model_export_name(model_dir, task)
    shap_root = run_root / f"{task}_shap" / model_dir.name
    audit_diagnostic_separation(model_dir, prefix)
    common = {
        "Nested_Cross_Validation.png": (
            "_nested_cv_f1_auc.png",
            "_metrics_line_plot.png",
            "_metrics_over_folds.png",
        ),
        "Learning_Curve.png": ("_learning_curve.png",),
        "Training_Size_Diagnostic.png": ("_training_size_curve.png",),
    }
    task_specific = (
        {
            "ROC.png": ("_roc_curve.png",),
            "Confusion_Matrix.png": ("_confusion_matrix.png",),
        }
        if task == "classification"
        else {
            "Observed_vs_Predicted.png": ("_overall_prediction.png", "_predictions.png"),
            "Residuals.png": ("_overall_residuals.png", "_residuals.png"),
        }
    )
    for destination, suffixes in {**common, **task_specific}.items():
        _export(_source_file(model_dir, prefix, suffixes), target / destination)
    publication_metrics_png = _source_file(model_dir, prefix, ("_publication_metrics.png",))
    publication_metrics_csv = _source_file(model_dir, prefix, ("_publication_metrics.csv",))
    _export(
        publication_metrics_png
        or _source_file(model_dir, prefix, ("_metrics.png", "_average_metrics.png")),
        target / "Evaluation_Metrics.png",
    )
    _export(
        publication_metrics_csv
        or _source_file(
            model_dir,
            prefix,
            ("_final_validation_metrics.csv", "_metrics.csv", "_average_metrics.csv"),
        ),
        target / "Evaluation_Metrics.csv",
    )
    audit_diagnostic_separation(model_dir, prefix, target)

    _export(
        _source_file(shap_root, prefix, ("_shap_summary_beeswarm.png",)),
        target / "Official_SHAP_Beeswarm.png",
    )
    top_source = _source_file(shap_root, prefix, ("_top_shap_biomarkers.csv",))
    top = _meaningful_rows(_read_csv(top_source), ("ImportanceValue", "MeanAbsSHAP", "MeanSHAP", "Importance"))
    if not top.empty:
        target.mkdir(parents=True, exist_ok=True)
        stem = "Top_SHAP_Biomarkers" if _table_is_true_shap(top) else "Top_Biomarker_Importance"
        top.to_csv(target / f"{stem}.csv", index=False)
        _plot_top_shap(top, target / f"{stem}.png", task, display_names)
    stability = _read_csv(_model_stability_source(run_root, prefix, model_dir))
    if not stability.empty:
        target.mkdir(parents=True, exist_ok=True)
        key = MODEL_STABILITY_KEYS.get(model_dir.name, model_dir.name)
        _plot_publication_biomarker_bars(
            stability,
            "TopMarkerStabilityScore",
            target / f"{prefix}_{key}_top_marker_stability.png",
            display_names,
            title=f"{_model_export_name(model_dir, task)} Top-marker Stability",
            subtitle="Outer-fold support and rank stability for this model's top markers.",
            x_label="Top-marker stability score",
            top=20,
        )
    if target.exists() and not any(target.iterdir()):
        target.rmdir()


def _read_csv(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _best_model_shap_dir(run_root: Path, prefix: str, task: str) -> Path | None:
    benchmark = run_root / "overall_benchmark"
    ranking_name = f"{task}_model_ranking.csv"
    ranking = _read_csv(_source_file(benchmark, prefix, (ranking_name,)))
    best_model = str(ranking.iloc[0]["Model"]) if not ranking.empty and "Model" in ranking.columns else ""
    shap_root = run_root / f"{task}_shap"
    candidates = [path for path in shap_root.iterdir() if path.is_dir()] if shap_root.exists() else []
    if best_model:
        best_key = _normalized(best_model)
        for candidate in candidates:
            aliases = {
                _normalized(candidate.name),
                _normalized(MODEL_EXPORT_NAMES.get(candidate.name, candidate.name)),
            }
            if best_key in aliases:
                return candidate
    return sorted(candidates, key=lambda path: path.name)[0] if candidates else None


def _classification_signed_shap(shap_dir: Path | None, prefix: str) -> pd.DataFrame:
    table = _read_csv(_source_file(shap_dir or Path("__missing__"), prefix, ("_per_class_shap.csv",)))
    required = {"Feature", "ClassName", "MeanSHAP"}
    if table.empty or not required.issubset(table.columns):
        return pd.DataFrame()
    table = _meaningful_rows(table, ("MeanSHAP",))
    if "GlobalRank" in table.columns:
        table = table.sort_values(["GlobalRank", "ClassName"])
    return table.copy()


def _plot_top_shap(
    table: pd.DataFrame,
    output: Path,
    task: str,
    display_names: dict[str, str] | None = None,
) -> None:
    if table.empty:
        return
    configure_publication_plots()
    import matplotlib.pyplot as plt

    if task == "classification" and {"Feature", "ClassName", "MeanSHAP"}.issubset(table.columns):
        rank_column = "GlobalRank" if "GlobalRank" in table.columns else "RankWithinClass"
        top_features = (
            table.groupby("Feature", as_index=False)[rank_column]
            .min()
            .sort_values(rank_column)
            .head(20)["Feature"]
            .astype(str)
            .tolist()
        )
        view = _meaningful_rows(
            table[table["Feature"].astype(str).isin(top_features)].copy(),
            ("MeanSHAP",),
        )
        if view.empty:
            return
        pivot = view.pivot_table(index="Feature", columns="ClassName", values="MeanSHAP", aggfunc="first")
        pivot = pivot.reindex([feature for feature in top_features[::-1] if feature in pivot.index])
        if pivot.empty:
            return
        pivot.index = [display_names.get(str(feature), str(feature)) if display_names else str(feature) for feature in pivot.index]
        fig_height = max(5.8, 0.30 * len(pivot) + 2.0)
        fig, ax = plt.subplots(figsize=(8.2, fig_height))
        pivot.plot(kind="barh", ax=ax, width=0.78)
        ax.axvline(0.0, color="#111827", linewidth=1.0)
        ax.set_xlabel("Mean signed SHAP")
        ax.set_ylabel("Protein")
        ax.set_title("Top Signed SHAP Biomarkers by Class")
        ax.legend(title="Class", frameon=False, loc="best")
        style_publication_axis(ax)
    else:
        feature = "Feature" if "Feature" in table.columns else "Protein"
        value = next(
            (column for column in ("ImportanceValue", "MeanImportance", "MeanAbsSHAP", "MeanSHAP") if column in table.columns),
            None,
        )
        if value is None:
            return
        true_shap = bool(table.get("IsTrueSHAP", pd.Series(False, index=table.index)).fillna(False).all())
        view = _meaningful_rows(table, (value,)).head(30).sort_values(value)
        if view.empty:
            return
        fig_height = max(5.8, 0.24 * len(view) + 2.0)
        fig, ax = plt.subplots(figsize=(7.4, fig_height))
        labels = [
            display_names.get(str(protein), str(protein)) if display_names else str(protein)
            for protein in view[feature]
        ]
        ax.barh(labels, pd.to_numeric(view[value], errors="coerce"), color="#4C78A8")
        ax.set_xlabel("Mean absolute SHAP" if true_shap else "Model-specific importance")
        ax.set_ylabel("Protein")
        ax.set_title("Top SHAP Biomarkers" if true_shap else "Top Biomarker Importance")
        style_publication_axis(ax)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    _remove_plot_sidecars(output)


def _table_is_true_shap(table: pd.DataFrame) -> bool:
    if table.empty:
        return False
    if "IsTrueSHAP" in table.columns:
        return bool(table["IsTrueSHAP"].fillna(False).astype(bool).all())
    if "ImportanceType" in table.columns:
        return bool(table["ImportanceType"].fillna("").astype(str).str.casefold().eq("shap").all())
    return False


def _plot_integrated_biomarkers(
    suggested: pd.DataFrame,
    signed_shap: pd.DataFrame,
    output: Path,
    task: str,
    consensus_available: bool,
    display_names: dict[str, str] | None = None,
) -> None:
    if suggested.empty:
        return
    configure_publication_plots()
    import matplotlib.pyplot as plt

    protein_column = "Protein" if "Protein" in suggested.columns else "Feature"
    importance_column = next(
        (column for column in ("MeanImportance", "ImportanceValue", "MeanAbsSHAP", "MeanSHAP") if column in suggested.columns),
        None,
    )
    if importance_column is None:
        return
    top = _meaningful_rows(suggested, (importance_column,)).head(min(20, len(suggested))).copy()
    if top.empty:
        return
    top[protein_column] = top[protein_column].astype(str)
    has_consensus = consensus_available and "ConsensusScore" in top.columns
    recommendation_column = next(
        (
            column
            for column in ("RecommendedBiomarkerScore", "ConfidenceScore")
            if column in top.columns
        ),
        None,
    )
    signed = pd.DataFrame()
    if task == "classification" and not signed_shap.empty:
        signed = _meaningful_rows(
            signed_shap[signed_shap["Feature"].astype(str).isin(top[protein_column])].copy(),
            ("MeanSHAP",),
        )
    display = top.iloc[::-1].reset_index(drop=True)
    proteins = display[protein_column].astype(str).tolist()
    protein_labels = [
        display_names.get(protein, protein) if display_names else protein
        for protein in proteins
    ]
    positions = np.arange(len(display), dtype=float)
    panel_count = 4 if has_consensus else 3
    widths = [1.28, 1.0, 1.0, 1.08] if has_consensus else [1.28, 1.0, 1.08]
    fig_height = max(7.2, 0.34 * len(display) + 2.6)
    fig, axes = plt.subplots(
        1,
        panel_count,
        figsize=(14.2 if has_consensus else 10.8, fig_height),
        sharey=True,
        gridspec_kw={"width_ratios": widths, "wspace": 0.08},
    )
    axes = np.atleast_1d(axes)
    for axis in axes:
        for row in range(len(display)):
            if row % 2 == 0:
                axis.axhspan(row - 0.5, row + 0.5, color="#F6F7F9", zorder=0)
        axis.set_ylim(-0.65, len(display) - 0.35)

    importance_ax = axes[0]
    if task == "classification" and not signed.empty:
        pivot = signed.pivot_table(index="Feature", columns="ClassName", values="MeanSHAP", aggfunc="first")
        pivot = pivot.reindex(proteins)
        class_names = [str(column) for column in pivot.columns]
        colors = ("#4C78A8", "#F58518", "#59A14F", "#B279A2", "#E45756", "#72B7B2")
        offsets = np.linspace(-0.22, 0.22, max(1, len(class_names)))
        for class_index, class_name in enumerate(class_names):
            values = pd.to_numeric(pivot[class_name], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(values)
            if not valid.any():
                continue
            y = positions[valid] + offsets[class_index]
            importance_ax.hlines(
                y,
                0.0,
                values[valid],
                color=colors[class_index % len(colors)],
                linewidth=1.15,
                alpha=0.68,
                zorder=2,
            )
            importance_ax.scatter(
                values[valid],
                y,
                s=24,
                color=colors[class_index % len(colors)],
                edgecolor="white",
                linewidth=0.45,
                label=class_name,
                zorder=3,
            )
        importance_ax.axvline(0.0, color="#1F2937", linewidth=0.9)
        importance_ax.set_xlabel("Mean signed SHAP")
        importance_ax.legend(
            title="Class",
            frameon=False,
            fontsize=7.5,
            title_fontsize=8,
            loc="lower right",
        )
    else:
        importance_values = pd.to_numeric(display[importance_column], errors="coerce").fillna(0.0).to_numpy()
        importance_ax.barh(positions, importance_values, color="#4C78A8", height=0.56, zorder=2)
        for y, value in zip(positions, importance_values):
            importance_ax.text(
                value,
                y,
                f"  {format_effect_value(value)}",
                va="center",
                ha="left",
                fontsize=7.2,
                color="#374151",
            )
        is_true_shap = bool(display.get("IsTrueSHAP", pd.Series(False, index=display.index)).fillna(False).all())
        importance_ax.set_xlabel("Mean absolute SHAP" if is_true_shap else "Mean model importance")
    importance_ax.set_yticks(positions, labels=protein_labels)
    importance_ax.tick_params(axis="y", labelleft=True, labelsize=8.2, length=0)
    importance_ax.set_title("1  Model importance", loc="left")
    style_publication_axis(importance_ax)

    stability_column = "StabilityScore" if "StabilityScore" in top.columns else "BiomarkerStabilityScore"
    stability_ax = axes[1]
    stability_values = pd.to_numeric(display.get(stability_column, 0.0), errors="coerce").fillna(0.0).to_numpy()
    stability_ax.barh(positions, stability_values, color="#59A14F", height=0.56, zorder=2)
    for y, value in zip(positions, stability_values):
        stability_ax.text(
            value,
            y,
            f"  {format_effect_value(value)}",
            va="center",
            ha="left",
            fontsize=7.2,
            color="#374151",
        )
    stability_ax.set_xlabel("Stability score")
    stability_ax.set_title("2  Reproducibility", loc="left")
    stability_ax.tick_params(axis="y", labelleft=False, length=0)
    style_publication_axis(stability_ax)

    next_panel = 2
    if has_consensus:
        consensus_ax = axes[next_panel]
        consensus_values = pd.to_numeric(display["ConsensusScore"], errors="coerce").fillna(0.0).to_numpy()
        consensus_ax.barh(positions, consensus_values, color="#F28E2B", height=0.56, zorder=2)
        for y, value in zip(positions, consensus_values):
            consensus_ax.text(
                value,
                y,
                f"  {format_effect_value(value)}",
                va="center",
                ha="left",
                fontsize=7.2,
                color="#374151",
            )
        consensus_ax.set_xlabel("Consensus score")
        consensus_ax.set_title("3  Cross-model support", loc="left")
        consensus_ax.tick_params(axis="y", labelleft=False, length=0)
        style_publication_axis(consensus_ax)
        next_panel += 1

    recommendation_ax = axes[next_panel]
    rank_lookup = {
        str(protein): rank
        for rank, protein in enumerate(top[protein_column].astype(str), start=1)
    }
    ranks = np.asarray([rank_lookup[protein] for protein in proteins], dtype=int)
    if recommendation_column is not None:
        recommendation_values = (
            pd.to_numeric(display[recommendation_column], errors="coerce")
            .fillna(0.0)
            .to_numpy()
        )
        recommendation_ax.barh(
            positions,
            recommendation_values,
            color="#B279A2",
            height=0.56,
            zorder=2,
        )
        for y, value, rank in zip(positions, recommendation_values, ranks):
            recommendation_ax.text(
                value,
                y,
                f"  #{rank}  {format_effect_value(value)}",
                va="center",
                ha="left",
                fontsize=7.2,
                color="#374151",
            )
        recommendation_ax.set_xlabel(
            "Recommendation score"
            if recommendation_column == "RecommendedBiomarkerScore"
            else "Confidence score"
        )
    else:
        rank_values = len(display) + 1 - ranks
        recommendation_ax.scatter(
            rank_values,
            positions,
            s=28,
            color="#B279A2",
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        for y, value, rank in zip(positions, rank_values, ranks):
            recommendation_ax.text(
                value,
                y,
                f"  #{rank}",
                va="center",
                ha="left",
                fontsize=7.2,
                color="#374151",
            )
        recommendation_ax.set_xlabel("Final recommendation rank")
    recommendation_ax.set_title(
        f"{4 if has_consensus else 3}  Final recommendation",
        loc="left",
    )
    recommendation_ax.tick_params(axis="y", labelleft=False, length=0)
    style_publication_axis(recommendation_ax)

    for axis in axes:
        axis.margins(x=0.16)
        axis.grid(axis="y", visible=False)
    fig.suptitle("Suggested Biomarkers: Convergent Scientific Evidence", fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.955,
        (
            "Model importance  \u2192  reproducibility  \u2192  cross-model support  \u2192  final recommendation"
            if has_consensus
            else "Model importance  \u2192  reproducibility  \u2192  final recommendation"
        ),
        ha="center",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        color="#374151",
    )
    fig.text(
        0.5,
        0.932,
        (
            f"Top {len(display)} non-negligible biomarkers retain the existing BiomarkerML recommendation order. "
            "The final panel displays the existing recommendation score without recomputation."
        ),
        ha="center",
        va="top",
        fontsize=8.4,
        color="#4B5563",
    )
    fig.subplots_adjust(left=0.085, right=0.988, bottom=0.075, top=0.89, wspace=0.08)
    fig.savefig(output)
    plt.close(fig)
    _remove_plot_sidecars(output)


def _plot_suggested_biomarker_explanation(
    suggested: pd.DataFrame,
    output: Path,
    display_names: dict[str, str],
    consensus_available: bool,
) -> None:
    if suggested.empty:
        return
    configure_publication_plots()
    import matplotlib.pyplot as plt

    protein_column = "Protein" if "Protein" in suggested.columns else "Feature"
    score_column = next(
        (
            column
            for column in ("RecommendedBiomarkerScore", "ConfidenceScore")
            if column in suggested.columns
        ),
        None,
    )
    top = suggested.head(min(3, len(suggested))).copy()
    names = [
        display_names.get(str(protein), str(protein))
        for protein in top[protein_column].astype(str)
    ]
    score_text = ""
    if score_column is not None:
        scores = pd.to_numeric(top[score_column], errors="coerce").dropna()
        if not scores.empty:
            score_text = (
                f" Their existing recommendation scores span "
                f"{format_effect_value(float(scores.min()))}–{format_effect_value(float(scores.max()))}."
            )
    attribution = "SHAP importance" if _table_is_true_shap(suggested) else "model-specific attribution or importance"
    evidence = f"{attribution}, validation stability, and cross-model consensus" if consensus_available else f"{attribution} and validation stability"
    conclusion = (
        f"{', '.join(names)} rank highest in the production recommendation output because they combine {evidence}.{score_text} "
        "This convergent predictive evidence prioritizes these proteins for targeted experimental validation; it does not by itself establish causal or clinical utility."
    )
    fig = plt.figure(figsize=(12.8, 2.8))
    fig.text(0.05, 0.79, recommendation_score_mathtext(), fontsize=18, fontweight="bold")
    fig.text(
        0.05,
        0.55,
        recommendation_score_definition(),
        fontsize=9.2,
        color="#374151",
        wrap=True,
    )
    fig.text(
        0.05,
        0.29,
        conclusion,
        fontsize=10.0,
        color="#1F2937",
        wrap=True,
    )
    fig.text(
        0.05,
        0.08,
        "The equation and weights are generated from the same constants used by the production recommendation calculation.",
        fontsize=8.3,
        color="#6B7280",
    )
    fig.savefig(output)
    plt.close(fig)
    _remove_plot_sidecars(output)


def _publication_display_names(
    run_root: Path,
    proteins,
    *,
    offline: bool,
) -> dict[str, str]:
    ppi = run_root / "ppi"
    annotation_tables = []
    if ppi.exists():
        for path in sorted(ppi.glob("*_ppi*_network_nodes.csv")):
            table = _read_csv(path)
            if not table.empty and "Protein" in table.columns:
                annotation_tables.append(table)
    annotations = pd.concat(annotation_tables, ignore_index=True) if annotation_tables else None
    return publication_gene_symbol_map(
        proteins,
        annotations=annotations,
        offline=offline,
    )


def _export_shap_summary(
    run_root: Path,
    final_root: Path,
    prefix: str,
    task: str,
    display_names: dict[str, str],
) -> None:
    target = final_root / "SHAP"
    target.mkdir(parents=True, exist_ok=True)
    reliability = run_root / "biomarker_reliability"
    shap_dir = _best_model_shap_dir(run_root, prefix, task)
    signed = _classification_signed_shap(shap_dir, prefix) if task == "classification" else pd.DataFrame()

    if task == "classification" and not signed.empty:
        signed.to_csv(target / "Top_SHAP_Biomarkers.csv", index=False)
        _plot_top_shap(signed, target / "Top_SHAP_Biomarkers.png", task, display_names)
    else:
        top_source = _source_file(shap_dir or Path("__missing__"), prefix, ("_top_shap_biomarkers.csv",))
        top = _meaningful_rows(
            _read_csv(top_source),
            ("ImportanceValue", "MeanImportance", "MeanAbsSHAP", "MeanSHAP", "Importance"),
        )
        if not top.empty:
            stem = "Top_SHAP_Biomarkers" if _table_is_true_shap(top) else "Top_Biomarker_Importance"
            top.to_csv(target / f"{stem}.csv", index=False)
            _plot_top_shap(top, target / f"{stem}.png", task, display_names)

    _export(
        _source_file(reliability, prefix, ("_biomarker_frequency.csv",)),
        target / "Stability.csv",
    )
    stability_table = _read_csv(
        _source_file(reliability, prefix, ("_biomarker_frequency.csv",))
    )
    _plot_publication_biomarker_bars(
        stability_table,
        "BiomarkerStabilityScore",
        target / "Stability.png",
        display_names,
        title="Cross-model Biomarker Frequency",
        subtitle="Top biomarkers ranked by outer-fold stability score across all executed models.",
        x_label="Biomarker stability score",
    )
    consensus_csv = _source_file(reliability, prefix, ("_consensus_biomarkers.csv",))
    consensus_table = _read_csv(consensus_csv)
    consensus_available = not consensus_table.empty
    if consensus_available and _export(consensus_csv, target / "Consensus.csv"):
        _plot_publication_biomarker_bars(
            consensus_table,
            "ConsensusScore",
            target / "Consensus.png",
            display_names,
            title="Consensus Biomarker Ranking",
            subtitle="Consensus score with support count and percentage across contributing models.",
            x_label="Consensus score",
            annotation_columns=("NumberSupportingModels", "SupportFrequency"),
        )
        _plot_publication_biomarker_bars(
            consensus_table,
            "SupportFrequency",
            target / "Consensus_Support.png",
            display_names,
            title="Consensus Support Frequency",
            subtitle="Proportional support among models included in consensus analysis.",
            x_label="Supporting models (proportion)",
        )

    suggested_source = _source_file(reliability, prefix, ("_recommended_biomarkers.csv",))
    suggested = _meaningful_rows(
        _read_csv(suggested_source),
        ("MeanImportance", "ImportanceValue", "MeanAbsSHAP", "MeanSHAP"),
    )
    if not suggested.empty:
        target.mkdir(parents=True, exist_ok=True)
        suggested.to_csv(target / "Suggested_Biomarkers.csv", index=False)
        ranking = suggested.head(min(30, len(suggested))).copy()
        ranking.to_csv(target / "Integrated_Biomarker_Ranking.csv", index=False)
        summary = pd.DataFrame(
            [
                {"Metric": "Task", "Value": task},
                {"Metric": "Suggested Biomarkers", "Value": int(len(suggested))},
                {"Metric": "Top Suggested Biomarker", "Value": str(suggested.iloc[0].get("Protein", suggested.iloc[0].get("Feature", "")))},
                {"Metric": "Consensus Available", "Value": consensus_available},
                {"Metric": "Recommendation Source", "Value": "Existing BiomarkerML recommended_biomarkers output"},
            ]
        )
        summary.to_csv(target / "Integrated_Biomarker_Summary.csv", index=False)
        _plot_integrated_biomarkers(
            suggested,
            signed,
            target / "Suggested_Biomarkers.png",
            task,
            consensus_available=consensus_available,
            display_names=display_names,
        )
        _plot_suggested_biomarker_explanation(
            suggested,
            target / "Suggested_Biomarkers_Explanation.png",
            display_names,
            consensus_available,
        )


def _export_ppi(run_root: Path, final_root: Path, prefix: str) -> None:
    source = run_root / "ppi"
    if not source.exists():
        return
    target = final_root / "PPI"
    mappings = {
        "Recommended_Biomarkers_PPI.png": ("_ppi_network.png",),
        "Recommended_Biomarkers_Hub_Proteins.png": ("_ppi_network_hub_proteins.png",),
        "Expanded_PPI.png": ("_ppi_expanded_network.png",),
        "Expanded_PPI_Hub_Proteins.png": ("_ppi_expanded_network_hub_proteins.png",),
        "Recommended_Nodes.csv": ("_ppi_network_nodes.csv",),
        "Recommended_Edges.csv": ("_ppi_network_edges.csv",),
        "Expanded_Nodes.csv": ("_ppi_expanded_network_nodes.csv",),
        "Expanded_Edges.csv": ("_ppi_expanded_network_edges.csv",),
        "Hub_Proteins.csv": ("_hub_proteins.csv",),
        "Community_Summary.csv": ("_community_summary.csv",),
    }
    for destination, suffixes in mappings.items():
        _export(_source_file(source, prefix, suffixes), target / destination)
    report = _read_csv(_source_file(source, prefix, ("_ppi_report.csv",)))
    if not report.empty and {"Section", "Metric", "Value"}.issubset(report.columns):
        metric_names = {
            "Number of Nodes",
            "Number of Edges",
            "Connected Components",
            "Average Degree",
            "Graph Density",
            "Clustering Coefficient",
            "Largest Connected Component",
            "Average STRING Confidence",
            "Average Co-expression",
        }
        statistics = report[report["Metric"].astype(str).isin(metric_names)].copy()
        if not statistics.empty:
            target.mkdir(parents=True, exist_ok=True)
            statistics.to_csv(target / "Network_Statistics.csv", index=False)


def _manifest_metadata(path: Path) -> tuple[str, str, str, str]:
    name = path.stem.lower()
    category = path.parts[0] if len(path.parts) > 1 else "Index"
    if "roc" in name:
        return "Prediction", "Model discrimination", "How well does the model discriminate?", category
    if "confusion" in name:
        return "Prediction", "Classification error structure", "Which classes are correctly or incorrectly predicted?", category
    if "observed" in name or "residual" in name:
        return "Prediction", "Regression fit", "How closely do predictions match observed outcomes?", category
    if "evaluation" in name or "nested" in name:
        return "Prediction", "Model performance", "How well and reproducibly does the model perform?", category
    if "learning_curve" in name:
        return "Prediction", "Optimization progression", "How does the fitted model behave across its valid optimization progression?", category
    if "training_size" in name:
        return "Prediction", "Sample-size generalization", "How does model performance improve with additional training samples?", category
    if "shap" in name:
        return "SHAP", "Model interpretation", "Which proteins contribute most strongly to model predictions?", category
    if "stability" in name:
        return "Biomarker Stability", "Biomarker reproducibility", "Which biomarker signals are reproducible across validation?", category
    if "consensus" in name:
        return "Consensus", "Cross-model support", "Which biomarkers are supported by multiple predictive models?", category
    if "suggested" in name or "integrated" in name:
        return "Recommended Biomarkers", "Biomarker prioritization", "Which biomarkers should be experimentally validated?", category
    if "ppi" in name or "node" in name or "edge" in name or "hub" in name or "community" in name or "network" in name:
        return "PPI", "Biological network interpretation", "How are the recommended biomarkers biologically connected?", category
    return "Publication Export", "Scientific deliverable", "What scientific evidence does this deliverable provide?", category


def _generate_manifest(final_root: Path) -> pd.DataFrame:
    files = sorted(path for path in final_root.rglob("*") if path.is_file() and path.name != "Manifest.csv")
    rows = []
    for path in files:
        relative = path.relative_to(final_root)
        stage, purpose, question, category = _manifest_metadata(relative)
        csv_peer = path.with_suffix(".csv")
        figure_peer = path.with_suffix(".png")
        rows.append(
            {
                "File Name": relative.as_posix(),
                "Source Workflow Stage": stage,
                "Scientific Purpose": purpose,
                "Scientific Question": question,
                "Associated CSV": csv_peer.relative_to(final_root).as_posix() if csv_peer.exists() else "",
                "Associated Figure": figure_peer.relative_to(final_root).as_posix() if figure_peer.exists() else "",
                "Output Category": category,
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(final_root / "Manifest.csv", index=False)
    return manifest


def validate_final_directory(final_root: Path) -> None:
    errors = []
    for path in final_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(final_root).as_posix()
        lower = relative.lower()
        if path.suffix.lower() not in {".csv", ".png"}:
            errors.append(f"unsupported file type: {relative}")
        if any(re.search(pattern, lower) for pattern in FORBIDDEN_FINAL_PATTERNS):
            errors.append(f"developer artifact: {relative}")
    if errors:
        raise RuntimeError("Final publication export contains invalid files:\n" + "\n".join(errors))


def generate_final_export(
    run_root: str | Path,
    prefix: str,
    *,
    offline: bool = False,
) -> Path:
    """Create the curated scientific export from existing production artifacts."""
    root = Path(run_root).resolve()
    final_root = root / "Final"
    if final_root.exists():
        shutil.rmtree(final_root)
    final_root.mkdir(parents=True)

    task = "classification" if (root / "classification").exists() and _model_sources(root, "classification") else "regression"
    protein_sources = []
    for path in root.rglob("*_top_shap_biomarkers.csv"):
        table = _read_csv(path)
        protein_column = "Feature" if "Feature" in table.columns else "Protein" if "Protein" in table.columns else None
        if protein_column:
            protein_sources.extend(table[protein_column].dropna().astype(str).tolist())
    recommended = _read_csv(_source_file(root / "biomarker_reliability", prefix, ("_recommended_biomarkers.csv",)))
    recommended_column = "Protein" if "Protein" in recommended.columns else "Feature" if "Feature" in recommended.columns else None
    if recommended_column:
        protein_sources.extend(recommended[recommended_column].dropna().astype(str).tolist())
    display_names = _publication_display_names(root, protein_sources, offline=offline)
    for model_dir in _model_sources(root, task):
        _export_model(root, final_root, prefix, task, model_dir, display_names)
    _export_shap_summary(root, final_root, prefix, task, display_names)
    _export_ppi(root, final_root, prefix)
    for directory in sorted(
        (path for path in final_root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        if not any(directory.iterdir()):
            directory.rmdir()
    _generate_manifest(final_root)
    validate_final_directory(final_root)
    return final_root
