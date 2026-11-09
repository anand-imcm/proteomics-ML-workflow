from __future__ import annotations

import argparse
import glob
import os
import re
import zipfile
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
try:
    import seaborn as sns
except Exception:
    sns = None

from core.publication_utils import (
    DEFAULT_RANDOM_STATE,
    assert_protein_level_features,
    build_output_prefix,
    configure_publication_plots,
    format_effect_value,
    is_component_feature,
    is_true_shap_source,
    robust_minmax,
    set_global_seed,
    stage_task_inputs,
    task_output_dir,
    task_run_context,
)

configure_publication_plots()
import matplotlib.pyplot as plt

RECOMMENDATION_SCORE_COMPONENTS = (
    ("NormalizedImportance", "normalized protein-level model attribution or importance", 0.40),
    ("ConsensusScore", "cross-model consensus support", 0.35),
    ("StabilityScore", "biomarker reproducibility", 0.25),
)


def recommendation_score(normalized_importance, consensus_score, stability_score):
    """Apply the canonical recommendation-score equation."""
    values = {
        "NormalizedImportance": normalized_importance,
        "ConsensusScore": consensus_score,
        "StabilityScore": stability_score,
    }
    return sum(weight * values[key] for key, _, weight in RECOMMENDATION_SCORE_COMPONENTS)


def recommendation_score_mathtext() -> str:
    """Return a display equation generated from the production weights."""
    symbols = {
        "NormalizedImportance": r"S_{\mathrm{importance}}^{\mathrm{norm}}",
        "ConsensusScore": r"S_{\mathrm{consensus}}",
        "StabilityScore": r"S_{\mathrm{stability}}",
    }
    terms = [
        rf"{weight:.2f}\,{symbols[key]}"
        for key, _, weight in RECOMMENDATION_SCORE_COMPONENTS
    ]
    return r"$R_{\mathrm{recommendation}} = " + " + ".join(terms) + "$"


def recommendation_score_definition() -> str:
    """Describe the variables in the canonical recommendation-score equation."""
    descriptions = [
        f"{key}: {description} (weight {weight:.2f})"
        for key, description, weight in RECOMMENDATION_SCORE_COMPONENTS
    ]
    return "; ".join(descriptions) + "."


def _read_csv(path: str) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _protein_col(df: pd.DataFrame) -> str | None:
    if "Protein" in df.columns:
        return "Protein"
    if "Feature" in df.columns:
        return "Feature"
    return None


def _numeric(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series([default] * len(df), index=df.index, dtype=float)


def _model_from_fold_file(path: str, prefix: str) -> str:
    name = Path(path).name
    name = re.sub(r"^" + re.escape(Path(prefix).name) + r"_", "", name)
    name = re.sub(r"_fold\d+.*$", "", name)
    return name


def _standardize_fold_table(path: str, prefix: str, top_n: int, source_priority: str) -> pd.DataFrame:
    df = _read_csv(path)
    if df is None or df.empty:
        return pd.DataFrame()
    protein_col = _protein_col(df)
    if protein_col is None:
        return pd.DataFrame()
    fold_match = re.search(r"_fold(\d+)", Path(path).name)
    fold_column = "FoldID" if "FoldID" in df.columns else "Fold" if "Fold" in df.columns else None
    fold_values = pd.to_numeric(df[fold_column], errors="coerce") if fold_column else pd.Series(dtype=float)
    fold = int(fold_match.group(1)) if fold_match else int(fold_values.dropna().iloc[0]) if fold_values.notna().any() else 0
    importance_source = (
        df["ImportanceType"].astype(str)
        if "ImportanceType" in df.columns
        else df["Importance_Source"].astype(str)
        if "Importance_Source" in df.columns
        else source_priority
    )
    explicit_importance = _numeric(df, "ImportanceValue", np.nan)
    if explicit_importance.isna().all():
        explicit_importance = _numeric(df, "Importance", np.nan)
    explicit_shap = _numeric(df, "MeanAbsSHAP", np.nan)
    if explicit_shap.isna().all():
        explicit_shap = _numeric(df, "Global Mean", np.nan)
    if explicit_shap.isna().all():
        explicit_shap = _numeric(df, "Mean SHAP Value", np.nan)
    source_text = importance_source.map(_canonical_importance_type)
    true_shap = source_text.eq("SHAP")
    importance_value = explicit_importance.fillna(explicit_shap).fillna(0.0)
    out = pd.DataFrame({
        "Model": df["Model"].astype(str) if "Model" in df.columns else _model_from_fold_file(path, prefix),
        "Fold": fold,
        "Protein": df[protein_col].astype(str),
        "Rank": _numeric(df, "Rank", np.nan),
        "ImportanceValue": importance_value,
        "ImportanceType": source_text,
        "MeanSHAP": explicit_shap.where(true_shap),
        "Importance": importance_value,
        "Importance_Source": source_text,
        "FeatureRole": df["FeatureRole"].astype(str) if "FeatureRole" in df.columns else "BIOMARKER",
    })
    out = out.loc[out["FeatureRole"].eq("BIOMARKER")].copy()
    out["Rank"] = out["Rank"].fillna(out.groupby(["Model", "Fold"]).cumcount() + 1)
    out = out[~out["Protein"].map(is_component_feature)]
    if out.empty:
        return out
    assert_protein_level_features(out["Protein"].tolist(), "Fold biomarker stability")
    return out.sort_values(["Model", "Fold", "Rank"]).groupby(["Model", "Fold"], group_keys=False).head(top_n)


def _load_fold_biomarkers(prefix: str, top_n: int = 200) -> pd.DataFrame:
    paths = sorted(glob.glob(f"{prefix}_*_fold[0-9]*_biomarkers.csv"))
    frames = [_standardize_fold_table(path, prefix, top_n, "EstimatorImportance") for path in paths]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _fold_key(row: pd.Series) -> str:
    return f"{row['Model']}_fold{int(row['Fold'])}"


def _offdiag_mean(df: pd.DataFrame) -> float:
    if df.empty or df.shape[0] < 2:
        return 1.0 if df.shape[0] == 1 else 0.0
    arr = df.to_numpy(dtype=float)
    mask = ~np.eye(arr.shape[0], dtype=bool)
    return float(np.nanmean(arr[mask]))


def _plot_heatmap(df: pd.DataFrame, title: str, out: str, vmin: float = -1.0, vmax: float = 1.0) -> None:
    if df.empty or "Status" in df.columns:
        return
    fig, ax = plt.subplots(figsize=(max(5, 0.36 * df.shape[1]), max(4, 0.36 * df.shape[0])))
    if sns is not None:
        sns.heatmap(df, vmin=vmin, vmax=vmax, cmap="vlag", square=True, cbar_kws={"shrink": 0.75}, ax=ax)
    else:
        image = ax.imshow(df.to_numpy(dtype=float), vmin=vmin, vmax=vmax, cmap="coolwarm")
        ax.set_xticks(range(df.shape[1]), df.columns, rotation=90)
        ax.set_yticks(range(df.shape[0]), df.index)
        plt.colorbar(image, ax=ax, shrink=0.75)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)


def _plot_bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    out: str,
    top: int = 30,
    subtitle: str | None = None,
    x_label: str | None = None,
) -> None:
    if df.empty:
        return
    plot_df = df.head(top).copy().iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.0, max(4.0, 0.28 * len(plot_df))))
    bars = ax.barh(plot_df[y].astype(str), plot_df[x], color="#4C78A8")
    ax.set_xlabel(x_label or x)
    ax.set_ylabel("")
    fig.suptitle(title, y=0.985, fontsize=14, fontweight="bold")
    if subtitle:
        fig.text(0.125, 0.935, subtitle, ha="left", va="top", fontsize=9)
    values = pd.to_numeric(plot_df[x], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    max_value = float(np.nanmax(np.abs(values))) if len(values) else 0.0
    offset = max(max_value * 0.01, 1e-9)
    for bar, value in zip(bars, values):
        ax.text(value + offset, bar.get_y() + bar.get_height() / 2, format_effect_value(value), va="center", fontsize=8.5)
    ax.set_xlim(0, max_value * 1.20 if max_value > 0 else 1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(out)
    plt.close(fig)


def _plot_consensus_bar(df: pd.DataFrame, out: str, contributing_models: int) -> None:
    if df.empty:
        return
    plot_df = df.head(30).copy().iloc[::-1]
    values = pd.to_numeric(plot_df["ConsensusScore"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    support = pd.to_numeric(plot_df["NumberSupportingModels"], errors="coerce").fillna(0).to_numpy(dtype=int)
    frequency = pd.to_numeric(plot_df["SupportFrequency"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.4, max(4.2, 0.30 * len(plot_df))))
    bars = ax.barh(plot_df["Protein"].astype(str), values, color="#4C78A8")
    max_value = float(np.nanmax(values)) if len(values) else 0.0
    offset = max(max_value * 0.012, 1e-9)
    for bar, score, count, fraction in zip(bars, values, support, frequency):
        label = f"{format_effect_value(score)} | {count}/{contributing_models} ({fraction:.0%})"
        ax.text(score + offset, bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=8.2)
    ax.set_xlabel("Consensus score")
    ax.set_ylabel("")
    ax.set_xlim(0, max_value * 1.42 if max_value > 0 else 1.0)
    fig.suptitle("Consensus Biomarker Ranking", y=0.985, fontsize=14, fontweight="bold")
    fig.text(
        0.125,
        0.935,
        f"Support count and percentage across {contributing_models} contributing models.",
        ha="left",
        va="top",
        fontsize=9,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(out)
    plt.close(fig)


def _safe_model_name(value: object) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return cleaned or "model"


def _model_identity(value: object) -> str:
    normalized = re.sub(r"[^a-z0-9]", "", str(value).lower())
    aliases = {
        "rf": "randomforest",
        "randomforestreg": "randomforestreg",
        "xgb": "xgboost",
        "xgbreg": "xgboostreg",
        "lgbm": "lightgbm",
        "lgbmreg": "lightgbmreg",
        "lr": "logisticregression",
        "nn": "neuralnetwork",
        "nnreg": "neuralnetworkreg",
        "gnb": "gaussiannb",
        "mlpvae": "vaemlp",
        "mlpvaereg": "mlpinvaereg",
        "vaemlpreg": "vaereg",
    }
    return aliases.get(normalized, normalized)


def _plot_model_stability_bars(folds: pd.DataFrame, prefix: str, top: int = 20) -> None:
    if folds.empty or not {"Model", "FoldKey", "Protein"}.issubset(folds.columns):
        return
    for model, group in folds.groupby("Model"):
        fold_count = max(1, group["FoldKey"].nunique())
        model_freq = group.groupby("Protein").agg(
            Frequency=("FoldKey", "nunique"),
            MeanRank=("Rank", "mean"),
            MeanImportance=("ImportanceValue", "mean"),
            Importance_Source=("Importance_Source", lambda s: ";".join(sorted(set(map(str, s))))),
        ).reset_index()
        model_freq["SupportFrequency"] = model_freq["Frequency"] / fold_count
        model_freq["TopMarkerStabilityScore"] = (
            0.60 * model_freq["SupportFrequency"]
            + 0.25 * robust_minmax(model_freq["MeanImportance"])
            + 0.15 * (1 - robust_minmax(model_freq["MeanRank"]))
        )
        model_freq = model_freq.sort_values(["TopMarkerStabilityScore", "Frequency", "MeanImportance"], ascending=False)
        safe = _safe_model_name(model)
        model_freq.to_csv(f"{prefix}_{safe}_top_marker_stability.csv", index=False)
        _plot_bar(
            model_freq,
            "TopMarkerStabilityScore",
            "Protein",
            f"{str(model).replace('_', ' ').title()} Top-marker Stability",
            f"{prefix}_{safe}_top_marker_stability.png",
            top=top,
            subtitle="Outer-fold support and rank stability for this model's top markers.",
            x_label="Top-marker stability score",
        )


def run_stability(prefix: str, top_n: int = 200) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folds = _load_fold_biomarkers(prefix, top_n=top_n)
    if folds.empty:
        empty = pd.DataFrame([{"Status": "No outer-fold biomarker records were found."}])
        empty.to_csv(f"{prefix}_stability_matrix.csv", index=False)
        empty.to_csv(f"{prefix}_biomarker_frequency.csv", index=False)
        empty.to_csv(f"{prefix}_biomarker_stability_summary.csv", index=False)
        return empty, empty, empty

    folds["FoldKey"] = folds.apply(_fold_key, axis=1)
    fold_keys = sorted(folds["FoldKey"].unique())
    jaccard = pd.DataFrame(np.eye(len(fold_keys)), index=fold_keys, columns=fold_keys)
    spearman = pd.DataFrame(np.eye(len(fold_keys)), index=fold_keys, columns=fold_keys)
    kendall = pd.DataFrame(np.eye(len(fold_keys)), index=fold_keys, columns=fold_keys)
    rank_maps = {key: folds.loc[folds["FoldKey"] == key, ["Protein", "Rank"]].set_index("Protein")["Rank"].to_dict() for key in fold_keys}
    for i, a in enumerate(fold_keys):
        for b in fold_keys[i + 1:]:
            set_a = set(rank_maps[a])
            set_b = set(rank_maps[b])
            union = set_a | set_b
            inter = set_a & set_b
            jaccard.loc[a, b] = jaccard.loc[b, a] = len(inter) / len(union) if union else 0.0
            if len(inter) >= 2:
                ra = [rank_maps[a][p] for p in inter]
                rb = [rank_maps[b][p] for p in inter]
                s = spearmanr(ra, rb).correlation
                k = kendalltau(ra, rb).correlation
                spearman.loc[a, b] = spearman.loc[b, a] = 0.0 if pd.isna(s) else float(s)
                kendall.loc[a, b] = kendall.loc[b, a] = 0.0 if pd.isna(k) else float(k)
            else:
                spearman.loc[a, b] = spearman.loc[b, a] = 0.0
                kendall.loc[a, b] = kendall.loc[b, a] = 0.0

    freq = folds.groupby("Protein").agg(
        Frequency=("FoldKey", "nunique"),
        MeanRank=("Rank", "mean"),
        RankSD=("Rank", "std"),
        MeanImportance=("ImportanceValue", "mean"),
        NumberSupportingModels=("Model", "nunique"),
        SupportingModels=("Model", lambda s: ";".join(sorted(set(map(str, s))))),
        Importance_Source=("Importance_Source", lambda s: ";".join(sorted(set(map(str, s))))),
    ).reset_index()
    freq["Feature"] = freq["Protein"]
    freq["SupportFrequency"] = freq["Frequency"] / max(1, len(fold_keys))
    freq["SelectionFrequency"] = freq["SupportFrequency"]
    freq["Selection_Frequency"] = freq["SupportFrequency"]
    freq["BiomarkerStabilityScore"] = 0.60 * freq["SupportFrequency"] + 0.25 * robust_minmax(freq["MeanImportance"]) + 0.15 * (1 - robust_minmax(freq["MeanRank"]))
    freq["StabilityScore"] = freq["BiomarkerStabilityScore"]
    freq["Biomarker_Stability_Score"] = freq["BiomarkerStabilityScore"]
    freq["ImportanceSource"] = freq["Importance_Source"]
    freq["IsTrueSHAP"] = freq["ImportanceSource"].eq("SHAP")
    freq["MeanSHAP"] = freq["MeanImportance"].where(freq["IsTrueSHAP"])
    freq = freq.sort_values(["BiomarkerStabilityScore", "Frequency", "MeanImportance"], ascending=False)
    summary = pd.DataFrame(
        {
            "Metric": ["Mean Jaccard", "Mean Spearman", "Mean Kendall Tau", "Biomarker Stability Score", "Fold Records", "Unique Biomarkers"],
            "Value": [_offdiag_mean(jaccard), _offdiag_mean(spearman), _offdiag_mean(kendall), float(freq["BiomarkerStabilityScore"].mean()), int(len(folds)), int(len(freq))],
        }
    )
    jaccard.to_csv(f"{prefix}_stability_matrix.csv")
    jaccard.to_csv(f"{prefix}_jaccard_matrix.csv")
    spearman.to_csv(f"{prefix}_spearman_rank_correlation_matrix.csv")
    kendall.to_csv(f"{prefix}_kendall_tau_matrix.csv")
    freq.to_csv(f"{prefix}_biomarker_frequency.csv", index=False)
    summary.to_csv(f"{prefix}_biomarker_stability_summary.csv", index=False)
    _plot_heatmap(jaccard, "Stability Heatmap", f"{prefix}_stability_heatmap.png", vmin=0.0, vmax=1.0)
    _plot_heatmap(jaccard, "Jaccard Matrix", f"{prefix}_jaccard_matrix.png", vmin=0.0, vmax=1.0)
    _plot_heatmap(spearman, "Rank Correlation Matrix", f"{prefix}_rank_correlation_matrix.png")
    _plot_bar(freq, "BiomarkerStabilityScore", "Protein", "Biomarker Frequency", f"{prefix}_biomarker_frequency_plot.png", subtitle="Top biomarkers ranked by outer-fold stability score.")
    _plot_model_stability_bars(folds, prefix)
    return jaccard, freq, summary


def _classification_candidates(prefix: str) -> tuple[list[str], pd.DataFrame]:
    rows = []
    for path in sorted(glob.glob(f"{prefix}_*_outer_fold_metrics.csv")):
        df = _read_csv(path)
        if df is None or not {"Model", "AUC", "weighted_F1"}.issubset(df.columns):
            continue
        model = str(df["Model"].dropna().iloc[0]) if df["Model"].notna().any() else Path(path).name.replace(f"{Path(prefix).name}_", "").replace("_outer_fold_metrics.csv", "")
        rows.append(
            {
                "Model": model,
                "MeanAUC": pd.to_numeric(df["AUC"], errors="coerce").mean(),
                "SDAUC": pd.to_numeric(df["AUC"], errors="coerce").std(ddof=1),
                "MeanWeightedF1": pd.to_numeric(df["weighted_F1"], errors="coerce").mean(),
                "SDWeightedF1": pd.to_numeric(df["weighted_F1"], errors="coerce").std(ddof=1),
                "OuterFolds": int(len(df)),
            }
        )
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        return [], metrics
    best_auc_idx = metrics["MeanAUC"].idxmax()
    best_f1_idx = metrics["MeanWeightedF1"].idxmax()
    best_auc_sd = metrics.loc[best_auc_idx, "SDAUC"]
    best_f1_sd = metrics.loc[best_f1_idx, "SDWeightedF1"]
    has_outer_sd = pd.notna(best_auc_sd) and pd.notna(best_f1_sd)
    auc_threshold = float(metrics.loc[best_auc_idx, "MeanAUC"] - best_auc_sd) if has_outer_sd else np.nan
    f1_threshold = float(metrics.loc[best_f1_idx, "MeanWeightedF1"] - best_f1_sd) if has_outer_sd else np.nan
    metrics["BestAUCThreshold"] = auc_threshold
    metrics["BestWeightedF1Threshold"] = f1_threshold
    metrics["ConsensusSelectionWarning"] = "" if has_outer_sd else "Outer-fold SD is unavailable; consensus candidate selection requires at least two outer folds."
    if has_outer_sd:
        metrics["IsConsensusCandidate"] = (metrics["MeanAUC"] >= auc_threshold) & (metrics["MeanWeightedF1"] >= f1_threshold)
    else:
        metrics["IsConsensusCandidate"] = np.isclose(metrics["MeanAUC"], metrics["MeanAUC"].max()) & np.isclose(metrics["MeanWeightedF1"], metrics["MeanWeightedF1"].max())
    metrics.to_csv(f"{prefix}_consensus_model_candidates.csv", index=False)
    return metrics.loc[metrics["IsConsensusCandidate"], "Model"].astype(str).tolist(), metrics


def _regression_candidates(prefix: str) -> tuple[list[str], pd.DataFrame]:
    rows = []
    for path in sorted(glob.glob(f"{prefix}_*_outer_fold_metrics.csv")):
        df = _read_csv(path)
        if df is None or not {"Model", "RMSE", "MAE", "R2"}.issubset(df.columns):
            continue
        model = str(df["Model"].dropna().iloc[0]) if df["Model"].notna().any() else Path(path).name.replace(f"{Path(prefix).name}_", "").replace("_outer_fold_metrics.csv", "")
        rows.append(
            {
                "Model": model,
                "MeanRMSE": pd.to_numeric(df["RMSE"], errors="coerce").mean(),
                "SDRMSE": pd.to_numeric(df["RMSE"], errors="coerce").std(ddof=1),
                "MeanMAE": pd.to_numeric(df["MAE"], errors="coerce").mean(),
                "SDMAE": pd.to_numeric(df["MAE"], errors="coerce").std(ddof=1),
                "MeanR2": pd.to_numeric(df["R2"], errors="coerce").mean(),
                "SDR2": pd.to_numeric(df["R2"], errors="coerce").std(ddof=1),
                "OuterFolds": int(len(df)),
            }
        )
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        return [], metrics
    rmse_idx = metrics["MeanRMSE"].idxmin()
    mae_idx = metrics["MeanMAE"].idxmin()
    r2_idx = metrics["MeanR2"].idxmax()
    rmse_sd = metrics.loc[rmse_idx, "SDRMSE"]
    mae_sd = metrics.loc[mae_idx, "SDMAE"]
    r2_sd = metrics.loc[r2_idx, "SDR2"]
    has_outer_sd = pd.notna(rmse_sd) and pd.notna(mae_sd) and pd.notna(r2_sd)
    rmse_threshold = float(metrics.loc[rmse_idx, "MeanRMSE"] + rmse_sd) if has_outer_sd else np.nan
    mae_threshold = float(metrics.loc[mae_idx, "MeanMAE"] + mae_sd) if has_outer_sd else np.nan
    r2_threshold = float(metrics.loc[r2_idx, "MeanR2"] - r2_sd) if has_outer_sd else np.nan
    metrics["BestRMSEThreshold"] = rmse_threshold
    metrics["BestMAEThreshold"] = mae_threshold
    metrics["BestR2Threshold"] = r2_threshold
    metrics["ConsensusSelectionWarning"] = "" if has_outer_sd else "Outer-fold SD is unavailable; consensus candidate selection requires at least two outer folds."
    if has_outer_sd:
        metrics["IsConsensusCandidate"] = (metrics["MeanRMSE"] <= rmse_threshold) & (metrics["MeanMAE"] <= mae_threshold) & (metrics["MeanR2"] >= r2_threshold)
    else:
        metrics["IsConsensusCandidate"] = (
            np.isclose(metrics["MeanRMSE"], metrics["MeanRMSE"].min())
            & np.isclose(metrics["MeanMAE"], metrics["MeanMAE"].min())
            & np.isclose(metrics["MeanR2"], metrics["MeanR2"].max())
        )
    metrics.to_csv(f"{prefix}_consensus_model_candidates.csv", index=False)
    return metrics.loc[metrics["IsConsensusCandidate"], "Model"].astype(str).tolist(), metrics


def _canonical_importance_type(value: object) -> str:
    text = str(value or "").strip()
    lower = text.lower()
    if is_true_shap_source(text):
        return "SHAP"
    if "coef" in lower:
        return "CoefficientMagnitude"
    if "gain" in lower:
        return "GainImportance"
    if "featureimportance" in lower or "feature_importance" in lower:
        return "NativeFeatureImportance"
    if "native" in lower or "estimator" in lower:
        return "NativeEstimatorImportance"
    if "uniform" in lower:
        return "UniformFallback"
    return text or "ModelSpecificImportance"


def _standardize_global_importance(df: pd.DataFrame, model: str) -> pd.DataFrame:
    protein_col = _protein_col(df)
    if protein_col is None:
        return pd.DataFrame()
    if "ImportanceValue" in df.columns:
        values = _numeric(df, "ImportanceValue")
    elif "MeanImportance" in df.columns:
        values = _numeric(df, "MeanImportance")
    elif "MeanAbsSHAP" in df.columns:
        values = _numeric(df, "MeanAbsSHAP")
    elif "Global Mean" in df.columns:
        values = _numeric(df, "Global Mean")
    elif "Mean SHAP Value" in df.columns:
        values = _numeric(df, "Mean SHAP Value")
    else:
        numeric_cols = df.drop(columns=[protein_col], errors="ignore").select_dtypes(include="number")
        values = numeric_cols.abs().mean(axis=1) if not numeric_cols.empty else pd.Series(np.zeros(len(df)))
    source_col = next(
        (column for column in ("ImportanceType", "Importance_Source", "ImportanceSource") if column in df.columns),
        None,
    )
    has_explicit_shap_values = any(
        column in df.columns for column in ("MeanAbsSHAP", "Global Mean", "Mean SHAP Value")
    )
    default_source = "SHAP" if has_explicit_shap_values else "ModelSpecificImportance"
    source = (
        df[source_col].map(_canonical_importance_type)
        if source_col
        else pd.Series([default_source] * len(df), index=df.index)
    )
    true_shap = source.eq("SHAP")
    out = pd.DataFrame({
        "Model": model,
        "Protein": df[protein_col].astype(str),
        "MeanImportance": values,
        "ImportanceType": source,
        "ImportanceSource": source,
        "IsTrueSHAP": true_shap,
        "MeanSHAP": values.where(true_shap),
        "FeatureRole": df["FeatureRole"].astype(str) if "FeatureRole" in df.columns else "BIOMARKER",
    })
    out = out.loc[out["FeatureRole"].eq("BIOMARKER")].copy()
    out = out[~out["Protein"].map(is_component_feature)]
    if not out.empty:
        assert_protein_level_features(out["Protein"].tolist(), "Consensus importance")
    out = out.sort_values("MeanImportance", ascending=False).reset_index(drop=True)
    out["Rank"] = np.arange(1, len(out) + 1)
    return out


def _standardize_shap(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Compatibility wrapper; provenance is resolved by the generic implementation."""
    return _standardize_global_importance(df, model)


def _load_global_importance(prefix: str) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for path in sorted(glob.glob(f"{prefix}_*_shap_values.csv")):
        if "_fold" in Path(path).name:
            continue
        model = Path(path).name.replace(f"{Path(prefix).name}_", "").replace("_shap_values.csv", "")
        df = _read_csv(path)
        if df is None:
            continue
        standardized = _standardize_global_importance(df, model)
        if not standardized.empty:
            tables[model] = standardized
    if tables:
        return tables
    folds = _load_fold_biomarkers(prefix)
    if folds.empty:
        return tables
    for model, group in folds.groupby("Model"):
        tmp = group.groupby("Protein").agg(
            MeanImportance=("ImportanceValue", "mean"),
            Rank=("Rank", "mean"),
            ImportanceType=("ImportanceType", lambda values: ";".join(sorted(set(map(_canonical_importance_type, values))))),
        ).reset_index().sort_values("MeanImportance", ascending=False)
        tmp["ImportanceSource"] = tmp["ImportanceType"]
        tmp["IsTrueSHAP"] = tmp["ImportanceType"].eq("SHAP")
        tmp["MeanSHAP"] = tmp["MeanImportance"].where(tmp["IsTrueSHAP"])
        tmp["Model"] = str(model)
        tmp["Rank"] = np.arange(1, len(tmp) + 1)
        tables[str(model)] = tmp[[
            "Model", "Protein", "MeanImportance", "ImportanceType",
            "ImportanceSource", "IsTrueSHAP", "MeanSHAP", "Rank",
        ]]
    return tables


def _load_global_shap(prefix: str) -> dict[str, pd.DataFrame]:
    """Compatibility alias for callers using the historical function name."""
    return _load_global_importance(prefix)



def run_consensus(prefix: str, task: str = "classification") -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates, metrics = _regression_candidates(prefix) if task == "regression" else _classification_candidates(prefix)
    metrics.to_csv(f"{prefix}_consensus_candidate_model_metrics.csv", index=False)
    importance_tables = _load_global_importance(prefix)
    importance_lookup = {_model_identity(model): table for model, table in importance_tables.items()}
    resolved = [(model, importance_lookup[_model_identity(model)]) for model in candidates if _model_identity(model) in importance_lookup]
    candidates = [model for model, _ in resolved]
    if not candidates:
        empty = pd.DataFrame(columns=["Protein", "SupportingModels", "NumberSupportingModels", "SupportFrequency", "MeanImportance", "MeanSHAP", "MeanRank", "RankSD", "StabilityScore", "ConsensusScore", "ImportanceType", "ImportanceSource", "IsTrueSHAP", "RecommendedCandidate"])
        empty.to_csv(f"{prefix}_consensus_biomarkers.csv", index=False)
        empty.to_csv(f"{prefix}_consensus_biomarker_summary.csv", index=False)
        return empty, metrics
    records = []
    for model, table in resolved:
        tmp = table.copy()
        tmp["Model"] = model
        tmp["NormalizedImportance"] = robust_minmax(tmp["MeanImportance"])
        records.append(tmp)
    combined = pd.concat(records, ignore_index=True)
    stability = _read_csv(f"{prefix}_biomarker_frequency.csv")
    stability_cols = ["Protein", "StabilityScore"]
    if stability is not None and "Protein" in stability.columns and "StabilityScore" in stability.columns:
        stability = stability[stability_cols]
    else:
        stability = pd.DataFrame(columns=stability_cols)
    consensus = combined.groupby("Protein").agg(
        SupportingModels=("Model", lambda s: ";".join(sorted(set(map(str, s))))),
        NumberSupportingModels=("Model", "nunique"),
        MeanImportance=("MeanImportance", "mean"),
        MeanRank=("Rank", "mean"),
        RankSD=("Rank", "std"),
        MeanNormalizedImportance=("NormalizedImportance", "mean"),
        IsTrueSHAP=("IsTrueSHAP", "all"),
        ImportanceType=("ImportanceType", lambda s: ";".join(sorted(set(map(str, s))))),
        ImportanceSource=("ImportanceSource", lambda s: ";".join(sorted(set(map(str, s))))),
    ).reset_index()
    consensus["SupportFrequency"] = consensus["NumberSupportingModels"] / max(1, len(candidates))
    consensus = consensus.merge(stability, on="Protein", how="left")
    consensus["StabilityScore"] = consensus["StabilityScore"].fillna(0.0)
    consensus["MeanSHAP"] = consensus["MeanImportance"].where(consensus["IsTrueSHAP"])
    consensus["ConsensusScore"] = 0.45 * consensus["SupportFrequency"] + 0.35 * consensus["MeanNormalizedImportance"] + 0.20 * (1 - robust_minmax(consensus["MeanRank"]))
    consensus["RecommendedCandidate"] = consensus["ConsensusScore"] >= consensus["ConsensusScore"].quantile(0.75) if len(consensus) else False
    consensus["Feature"] = consensus["Protein"]
    consensus = consensus.sort_values(["ConsensusScore", "SupportFrequency", "MeanImportance"], ascending=False)
    summary = pd.DataFrame(
        {
            "Metric": ["Candidate Models", "Consensus Biomarkers", "Mean Consensus Score", "Top Consensus Protein"],
            "Value": [";".join(candidates), int(len(consensus)), float(consensus["ConsensusScore"].mean()) if len(consensus) else 0.0, consensus["Protein"].iloc[0] if len(consensus) else ""],
        }
    )
    consensus.to_csv(f"{prefix}_consensus_biomarkers.csv", index=False)
    summary.to_csv(f"{prefix}_consensus_biomarker_summary.csv", index=False)
    model_text = "Contributing models: " + (", ".join(candidates) if candidates else "none")
    _plot_consensus_bar(consensus, f"{prefix}_consensus_biomarker_ranking.png", len(candidates))
    _plot_bar(
        consensus.sort_values(["SupportFrequency", "ConsensusScore"], ascending=False),
        "SupportFrequency",
        "Protein",
        "Consensus Support Frequency",
        f"{prefix}_consensus_frequency_plot.png",
        subtitle=model_text,
        x_label=f"Supporting models (proportion of {len(candidates)})",
    )
    return consensus, metrics


def run_recommended(prefix: str) -> pd.DataFrame:
    consensus = _read_csv(f"{prefix}_consensus_biomarkers.csv")
    frequency = _read_csv(f"{prefix}_biomarker_frequency.csv")
    importance_tables = _load_global_importance(prefix)
    importance_df = pd.concat(importance_tables.values(), ignore_index=True) if importance_tables else pd.DataFrame(columns=["Protein", "MeanImportance", "Rank", "Model"])
    if not importance_df.empty:
        importance_summary = importance_df.groupby("Protein").agg(
            MeanImportance=("MeanImportance", "mean"),
            MeanRank=("Rank", "mean"),
            SupportingModels=("Model", lambda s: ";".join(sorted(set(map(str, s))))),
            IsTrueSHAP=("IsTrueSHAP", "all"),
            ImportanceType=("ImportanceType", lambda s: ";".join(sorted(set(map(str, s))))),
            ImportanceSource=("ImportanceSource", lambda s: ";".join(sorted(set(map(str, s))))),
        ).reset_index()
        importance_summary["MeanSHAP"] = importance_summary["MeanImportance"].where(importance_summary["IsTrueSHAP"])
    else:
        importance_summary = pd.DataFrame(columns=["Protein", "MeanImportance", "MeanSHAP", "MeanRank", "SupportingModels", "IsTrueSHAP", "ImportanceType", "ImportanceSource"])
    proteins = set(importance_summary.get("Protein", []))
    if consensus is not None and "Protein" in consensus.columns:
        proteins |= set(consensus["Protein"].astype(str))
    if frequency is not None and "Protein" in frequency.columns:
        proteins |= set(frequency["Protein"].astype(str))
    recommended = pd.DataFrame({"Protein": sorted(proteins)})
    recommended = recommended.merge(importance_summary, on="Protein", how="left")
    if consensus is not None and "Protein" in consensus.columns:
        keep = [c for c in ["Protein", "ConsensusScore", "SupportFrequency", "NumberSupportingModels", "SupportingModels"] if c in consensus.columns]
        recommended = recommended.merge(
            consensus[keep].rename(
                columns={
                    "SupportFrequency": "ConsensusSupportFrequency",
                    "NumberSupportingModels": "ConsensusNumberSupportingModels",
                    "SupportingModels": "ConsensusSupportingModels",
                }
            ),
            on="Protein",
            how="left",
        )
    if frequency is not None and "Protein" in frequency.columns:
        keep = [c for c in ["Protein", "StabilityScore", "BiomarkerStabilityScore", "SupportFrequency", "NumberSupportingModels"] if c in frequency.columns]
        recommended = recommended.merge(frequency[keep], on="Protein", how="left")
    recommended["ConsensusScore"] = _numeric_or_default(recommended, "ConsensusScore")
    recommended["StabilityScore"] = (
        _numeric_or_default(recommended, "StabilityScore")
        if "StabilityScore" in recommended.columns
        else _numeric_or_default(recommended, "BiomarkerStabilityScore")
    )
    recommended["MeanImportance"] = _numeric_or_default(recommended, "MeanImportance")
    recommended["IsTrueSHAP"] = recommended.get("IsTrueSHAP", pd.Series(False, index=recommended.index)).fillna(False).astype(bool)
    recommended["MeanSHAP"] = recommended["MeanImportance"].where(recommended["IsTrueSHAP"])
    recommended["MeanRank"] = _numeric_or_default(recommended, "MeanRank")
    stability_support = _numeric_or_default(recommended, "SupportFrequency", np.nan)
    consensus_support = _numeric_or_default(recommended, "ConsensusSupportFrequency", 0.0)
    recommended["SupportFrequency"] = stability_support.fillna(consensus_support).fillna(0.0)
    stability_models = _numeric_or_default(recommended, "NumberSupportingModels", np.nan)
    consensus_models = _numeric_or_default(recommended, "ConsensusNumberSupportingModels", 0.0)
    recommended["NumberSupportingModels"] = stability_models.fillna(consensus_models).fillna(0.0).astype(int)
    recommended["SupportingModels"] = _text_or_default(
        recommended,
        "ConsensusSupportingModels" if "ConsensusSupportingModels" in recommended.columns else "SupportingModels",
        "",
    )
    recommended["ImportanceType"] = _text_or_default(recommended, "ImportanceType", "ModelSpecificImportance")
    recommended["ImportanceSource"] = _text_or_default(recommended, "ImportanceSource", "ModelSpecificImportance")
    recommended["NormalizedImportance"] = robust_minmax(recommended["MeanImportance"])
    recommended["RecommendedBiomarkerScore"] = recommendation_score(
        recommended["NormalizedImportance"],
        recommended["ConsensusScore"],
        recommended["StabilityScore"],
    )
    recommended["ConfidenceScore"] = (100 * robust_minmax(recommended["RecommendedBiomarkerScore"])).round(2)
    threshold = recommended["RecommendedBiomarkerScore"].quantile(0.75) if len(recommended) else 0.0
    recommended["Recommended"] = recommended["RecommendedBiomarkerScore"] >= threshold
    recommended["Feature"] = recommended["Protein"]
    recommended = recommended.sort_values(["RecommendedBiomarkerScore", "ConfidenceScore"], ascending=False)
    recommended["FeatureRole"] = "BIOMARKER"
    columns = ["Protein", "ConfidenceScore", "ConsensusScore", "StabilityScore", "MeanImportance", "NormalizedImportance", "MeanSHAP", "MeanRank", "SupportFrequency", "NumberSupportingModels", "SupportingModels", "ImportanceType", "ImportanceSource", "IsTrueSHAP", "FeatureRole", "Recommended", "RecommendedBiomarkerScore", "Feature"]
    recommended = recommended[[c for c in columns if c in recommended.columns]]
    recommended.to_csv(f"{prefix}_recommended_biomarkers.csv", index=False)
    _plot_bar(recommended, "RecommendedBiomarkerScore", "Protein", "Recommended Biomarkers", f"{prefix}_recommended_biomarkers.png", subtitle="Integrated protein-level model attribution, stability, and consensus support.")
    return recommended


def _numeric_or_default(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series([default] * len(df), index=df.index, dtype=float)


def _text_or_default(df: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column in df.columns:
        return df[column].fillna(default).astype(str)
    return pd.Series([default] * len(df), index=df.index, dtype=str)


def _excel_col(idx: int) -> str:
    letters = ""
    idx += 1
    while idx:
        idx, rem = divmod(idx - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def _cell_xml(value, row_idx: int, col_idx: int) -> str:
    ref = f"{_excel_col(col_idx)}{row_idx}"
    if pd.isna(value):
        return f'<c r="{ref}"/>'
    if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
        return f'<c r="{ref}"><v>{float(value)}</v></c>'
    return f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'


def write_simple_xlsx(path: str, sheets: dict[str, pd.DataFrame]) -> None:
    safe_names = []
    for idx, name in enumerate(sheets, start=1):
        clean = re.sub(r"[\[\]\:\*\?\/\\]", "_", name)[:31] or f"Sheet{idx}"
        if clean in safe_names:
            clean = f"{clean[:27]}_{idx}"
        safe_names.append(clean)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "<?xml version='1.0' encoding='UTF-8'?><Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'><Default Extension='rels' ContentType='application/vnd.openxmlformats-package.relationships+xml'/><Default Extension='xml' ContentType='application/xml'/><Override PartName='/xl/workbook.xml' ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml'/>" + "".join(f"<Override PartName='/xl/worksheets/sheet{i}.xml' ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml'/>" for i in range(1, len(safe_names) + 1)) + "</Types>")
        zf.writestr("_rels/.rels", "<?xml version='1.0' encoding='UTF-8'?><Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'><Relationship Id='rId1' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument' Target='xl/workbook.xml'/></Relationships>")
        zf.writestr("xl/_rels/workbook.xml.rels", "<?xml version='1.0' encoding='UTF-8'?><Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'>" + "".join(f"<Relationship Id='rId{i}' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet' Target='worksheets/sheet{i}.xml'/>" for i in range(1, len(safe_names) + 1)) + "</Relationships>")
        zf.writestr("xl/workbook.xml", "<?xml version='1.0' encoding='UTF-8'?><workbook xmlns='http://schemas.openxmlformats.org/spreadsheetml/2006/main' xmlns:r='http://schemas.openxmlformats.org/officeDocument/2006/relationships'><sheets>" + "".join(f"<sheet name='{escape(name)}' sheetId='{i}' r:id='rId{i}'/>" for i, name in enumerate(safe_names, start=1)) + "</sheets></workbook>")
        for idx, ((_, df), _) in enumerate(zip(sheets.items(), safe_names), start=1):
            columns = list(df.columns)
            rows = ["<row r='1'>" + "".join(_cell_xml(col, 1, c) for c, col in enumerate(columns)) + "</row>"]
            for r, (_, row) in enumerate(df.iterrows(), start=2):
                rows.append(f"<row r='{r}'>" + "".join(_cell_xml(row[col], r, c) for c, col in enumerate(columns)) + "</row>")
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", "<?xml version='1.0' encoding='UTF-8'?><worksheet xmlns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'><sheetData>" + "".join(rows) + "</sheetData></worksheet>")


def write_publication_summary(prefix: str, task: str = "classification", input_csv: str | None = None) -> None:
    file_map = {
        "Model Candidates": f"{prefix}_consensus_model_candidates.csv",
        "Consensus Biomarkers": f"{prefix}_consensus_biomarkers.csv",
        "Consensus Summary": f"{prefix}_consensus_biomarker_summary.csv",
        "Biomarker Frequency": f"{prefix}_biomarker_frequency.csv",
        "Stability Summary": f"{prefix}_biomarker_stability_summary.csv",
        "Recommended Biomarkers": f"{prefix}_recommended_biomarkers.csv",
        "Hub Proteins": f"{prefix}_hub_proteins.csv",
        "Consensus Hub Proteins": f"{prefix}_consensus_hub_proteins.csv",
    }
    sheets = {name: df for name, path in file_map.items() if (df := _read_csv(path)) is not None and not df.empty}
    summary_rows = [{"Field": "Task", "Value": task}, {"Field": "Prefix", "Value": prefix}]
    if input_csv and os.path.exists(input_csv):
        df = pd.read_csv(input_csv)
        summary_rows += [
            {"Field": "Dataset", "Value": input_csv},
            {"Field": "Samples", "Value": len(df)},
            {"Field": "Features", "Value": len([c for c in df.columns if c not in {"SampleID", "Label"}])},
        ]
    rec = sheets.get("Recommended Biomarkers")
    if rec is not None and not rec.empty:
        summary_rows.append({"Field": "Top Recommended Biomarker", "Value": rec.iloc[0].get("Protein", "")})
    cand = sheets.get("Model Candidates")
    if cand is not None and not cand.empty:
        best = cand.sort_values([c for c in cand.columns if c.startswith("Mean")], ascending=False).iloc[0]
        summary_rows.append({"Field": "Best Model", "Value": best.get("Model", "")})
    summary = pd.DataFrame(summary_rows)
    sheets = {"Publication Summary": summary, **sheets}
    summary.to_csv(f"{prefix}_publication_summary.csv", index=False)
    if not sheets:
        return
    out = f"{prefix}_publication_summary.xlsx"
    try:
        with pd.ExcelWriter(out) as writer:
            for name, df in sheets.items():
                df.to_excel(writer, sheet_name=name[:31], index=False)
    except Exception:
        write_simple_xlsx(out, sheets)


def run_publication_biomarker_analysis(prefix: str, task: str = "classification", input_csv: str | None = None) -> None:
    run_stability(prefix)
    run_consensus(prefix, task=task)
    run_recommended(prefix)
    write_publication_summary(prefix, task=task, input_csv=input_csv)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate biomarker stability, consensus, and recommended biomarker outputs.")
    parser.add_argument("--input", default=None, help="Task input directory containing SHAP and model metric artifacts.")
    parser.add_argument("-p", "--prefix", default=None)
    parser.add_argument("--outdir", default=".", help="Base output directory")
    parser.add_argument("--feature_selection", choices=["yes", "no"], default="no", help="Accepted for CLI consistency")
    parser.add_argument("--jobs", type=int, default=1, help="Accepted for CLI consistency")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("-t", "--task", default="classification", choices=["classification", "regression"])
    parser.add_argument("-i", "--csv", default=None)
    args = parser.parse_args()
    prefix = build_output_prefix(args.input or args.csv, args.prefix)
    set_global_seed(args.seed)
    outdir = task_output_dir(args.outdir, "biomarker_reliability")
    stage_task_inputs(args.input, outdir, prefix)
    with task_run_context(args.outdir, "biomarker_reliability"):
        run_publication_biomarker_analysis(prefix, args.task, args.csv)


if __name__ == "__main__":
    main()
