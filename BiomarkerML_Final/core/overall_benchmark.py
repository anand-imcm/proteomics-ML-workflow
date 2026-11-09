from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_ROOT = Path(__file__).resolve().parent
ROOT = SCRIPT_ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.classification_metrics import classification_sensitivity_specificity

try:
    from .publication_utils import (
        DEFAULT_RANDOM_STATE,
        PUBLICATION_COLORS,
        add_bar_value_labels,
        add_common_cli_arguments,
        build_output_prefix,
        configure_publication_plots,
        plot_metric_bars,
        set_global_seed,
        style_publication_axis,
        task_output_dir,
        task_run_context,
        write_artifact_alias,
    )
except ImportError:
    from publication_utils import (
        DEFAULT_RANDOM_STATE,
        PUBLICATION_COLORS,
        add_bar_value_labels,
        add_common_cli_arguments,
        build_output_prefix,
        configure_publication_plots,
        plot_metric_bars,
        set_global_seed,
        style_publication_axis,
        task_output_dir,
        task_run_context,
        write_artifact_alias,
    )


def run_overall_benchmark_task(args) -> None:
    prefix = build_output_prefix(args.input, args.prefix)
    set_global_seed(getattr(args, "seed", DEFAULT_RANDOM_STATE))
    task_output_dir(args.outdir, "overall_benchmark")
    source = Path(args.input).resolve() if Path(str(args.input)).is_dir() else task_output_dir(args.outdir, args.task)
    with task_run_context(args.outdir, "overall_benchmark"):
        if args.task == "classification":
            run_classification_benchmark(prefix, source)
        else:
            run_regression_benchmark(prefix, source)


def run_classification_benchmark(prefix: str, source_dir: Path = Path(".")) -> None:
    rows = []
    roc_curves = []
    for model_dir in _model_directories(source_dir):
        metrics = _read_classification_model(model_dir, prefix)
        if metrics is None:
            continue
        rows.append(metrics)
        _write_classification_publication_metrics(model_dir, prefix, metrics)
        roc_curves.extend(_read_model_roc(model_dir, prefix, metrics["Model"]))

    comparison = pd.DataFrame(rows)
    if comparison.empty:
        raise RuntimeError("No completed classification model outputs were found for benchmarking.")

    comparison = comparison.sort_values(["OOF AUC", "Weighted F1", "Accuracy", "Model"], ascending=[False, False, False, True])
    comparison_out = comparison[
        ["Model", "Accuracy", "Weighted F1", "OOF AUC", "OOF Micro AUC", "MeanFold AUC", "SDFold AUC", "Sensitivity", "Specificity"]
    ].copy()
    ranking = comparison[["Model", "Accuracy", "Weighted F1", "OOF AUC", "OOF Micro AUC", "MeanFold AUC", "SDFold AUC"]].copy()
    ranking["Rank"] = np.arange(1, len(ranking) + 1)
    ranking = ranking[["Rank", "Model", "OOF AUC", "Weighted F1", "Accuracy", "OOF Micro AUC", "MeanFold AUC", "SDFold AUC"]]

    best = ranking.iloc[0].to_dict()
    summary = {
        "Task": "classification",
        "BestModel": best["Model"],
        "SelectionCriterion": "Highest pooled outer-fold OOF AUC; ties broken by Weighted F1 and then Accuracy.",
        "Reason": _classification_reason(best),
        "Metrics": _json_metrics(
            comparison.iloc[0].to_dict(),
            ["OOF AUC", "Weighted F1", "Accuracy", "OOF Micro AUC", "MeanFold AUC", "SDFold AUC", "Sensitivity", "Specificity"],
        ),
    }

    _write_table_outputs(prefix, "classification_model_comparison.csv", comparison_out)
    _write_table_outputs(prefix, "classification_model_ranking.csv", ranking)
    _write_json_outputs(prefix, "classification_best_model_summary.json", summary)
    _plot_all_model_roc(roc_curves, "All-model ROC Comparison", f"{prefix}_all_models_roc_comparison.png")
    _plot_ranking(comparison, "Accuracy", "Accuracy Ranking", f"{prefix}_accuracy_ranking.png", higher_is_better=True, ylim=(0.0, 1.12))
    _plot_ranking(comparison, "Weighted F1", "Weighted F1 Ranking", f"{prefix}_weighted_f1_ranking.png", higher_is_better=True, ylim=(0.0, 1.12))
    _plot_ranking(comparison, "OOF AUC", "Pooled OOF AUC Ranking", f"{prefix}_macro_auc_ranking.png", higher_is_better=True, ylim=(0.0, 1.12))
    _plot_overall_performance_summary(
        comparison,
        ["Accuracy", "Weighted F1", "OOF AUC", "OOF Micro AUC", "Sensitivity", "Specificity"],
        "Overall Classification Performance Summary",
        f"{prefix}_overall_performance_summary.png",
        higher_is_better={
            "Accuracy": True,
            "Weighted F1": True,
            "OOF AUC": True,
            "OOF Micro AUC": True,
            "Sensitivity": True,
            "Specificity": True,
        },
    )
    _alias_benchmark_outputs(prefix, [
        "all_models_roc_comparison.png",
        "accuracy_ranking.png",
        "weighted_f1_ranking.png",
        "macro_auc_ranking.png",
        "overall_performance_summary.png",
    ])


def run_regression_benchmark(prefix: str, source_dir: Path = Path(".")) -> None:
    rows = []
    predictions = []
    for model_dir in _model_directories(source_dir):
        metrics = _read_regression_model(model_dir, prefix)
        if metrics is None:
            continue
        rows.append(metrics)
        pred = _read_regression_predictions(model_dir, prefix, metrics["Model"])
        if pred is not None:
            predictions.append(pred)

    comparison = pd.DataFrame(rows)
    if comparison.empty:
        raise RuntimeError("No completed regression model outputs were found for benchmarking.")

    comparison = comparison.sort_values(["RMSE", "R2", "Model"], ascending=[True, False, True])
    comparison_out = comparison[["Model", "RMSE", "MAE", "R2"]].copy()
    ranking = comparison_out.copy()
    ranking["Rank"] = np.arange(1, len(ranking) + 1)
    ranking = ranking[["Rank", "Model", "RMSE", "MAE", "R2"]]

    best = ranking.iloc[0].to_dict()
    summary = {
        "Task": "regression",
        "BestModel": best["Model"],
        "SelectionCriterion": "Lowest RMSE; ties broken by highest R2.",
        "Reason": _regression_reason(best),
        "Metrics": _json_metrics(best, ["RMSE", "MAE", "R2"]),
    }

    _write_table_outputs(prefix, "regression_model_comparison.csv", comparison_out)
    _write_table_outputs(prefix, "regression_model_ranking.csv", ranking)
    _write_json_outputs(prefix, "regression_best_model_summary.json", summary)
    _plot_prediction_comparison(predictions, "Prediction Scatter Comparison", f"{prefix}_prediction_scatter_comparison.png")
    _plot_ranking(comparison, "RMSE", "RMSE Ranking", f"{prefix}_rmse_ranking.png", higher_is_better=False)
    _plot_ranking(comparison, "MAE", "MAE Ranking", f"{prefix}_mae_ranking.png", higher_is_better=False)
    _plot_ranking(comparison, "R2", "R2 Ranking", f"{prefix}_r2_ranking.png", higher_is_better=True)
    _plot_overall_performance_summary(
        comparison,
        ["RMSE", "MAE", "R2"],
        "Overall Regression Performance Summary",
        f"{prefix}_overall_performance_summary.png",
        higher_is_better={"RMSE": False, "MAE": False, "R2": True},
    )
    _alias_benchmark_outputs(prefix, [
        "prediction_scatter_comparison.png",
        "rmse_ranking.png",
        "mae_ranking.png",
        "r2_ranking.png",
        "overall_performance_summary.png",
    ])


def _model_directories(base: Path) -> list[Path]:
    return sorted([path for path in base.iterdir() if path.is_dir()], key=lambda path: path.name.lower())


def _read_classification_model(model_dir: Path, prefix: str) -> dict[str, Any] | None:
    model = model_dir.name
    final_metrics = _read_first_csv(model_dir.glob(f"{prefix}_*_final_validation_metrics.csv"))
    predictions = _read_pooled_prediction_csv(model_dir, prefix)
    fold_metrics = _read_first_csv(model_dir.glob(f"{prefix}_*_outer_fold_metrics.csv"))
    row = {
        "Model": model,
        "Accuracy": np.nan,
        "Weighted F1": np.nan,
        "OOF AUC": np.nan,
        "OOF Micro AUC": np.nan,
        "MeanFold AUC": _metric_mean(fold_metrics, ["Macro_AUC", "AUC"]),
        "SDFold AUC": float(_metric_series(fold_metrics, ["Macro_AUC", "AUC"]).std(ddof=1)) if len(_metric_series(fold_metrics, ["Macro_AUC", "AUC"])) > 1 else 0.0,
        "Sensitivity": np.nan,
        "Specificity": np.nan,
    }
    if predictions is not None:
        _fill_classification_from_predictions(row, predictions)
        row["OOF AUC"], row["OOF Micro AUC"] = _oof_auc_from_predictions(predictions)
    if not np.isfinite(row["OOF AUC"]):
        # Compatibility for legacy artifacts that explicitly identify pooled OOF metrics.
        scope = str(final_metrics.iloc[0].get("MetricScope", "")) if final_metrics is not None and not final_metrics.empty else ""
        if "out-of-fold" in scope.lower():
            row["OOF AUC"] = _metric_value(final_metrics, ["OOF_AUC", "Macro_AUC"])
            row["OOF Micro AUC"] = _metric_value(final_metrics, ["Micro_AUC"])
    if not _has_any_metric(row, ["Accuracy", "Weighted F1", "OOF AUC"]):
        return None
    return row


def _read_regression_model(model_dir: Path, prefix: str) -> dict[str, Any] | None:
    model = model_dir.name
    final_metrics = _read_first_csv(model_dir.glob(f"{prefix}_*_final_validation_metrics.csv"))
    predictions = _read_regression_predictions(model_dir, prefix, model)
    row = {"Model": model, "RMSE": np.nan, "MAE": np.nan, "R2": np.nan}
    if predictions is not None:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        y_true = predictions["Original Label"].to_numpy(dtype=float)
        y_pred = predictions["Predicted Label"].to_numpy(dtype=float)
        row.update({
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "R2": float(r2_score(y_true, y_pred)),
        })
    else:
        scope = str(final_metrics.iloc[0].get("MetricScope", "")) if final_metrics is not None and not final_metrics.empty else ""
        if "out-of-fold" in scope.lower():
            row.update({
                "RMSE": _metric_value(final_metrics, ["RMSE"]),
                "MAE": _metric_value(final_metrics, ["MAE"]),
                "R2": _metric_value(final_metrics, ["R2", "R²"]),
            })
    if not _has_any_metric(row, ["RMSE", "MAE", "R2"]):
        return None
    return row


def _read_first_csv(paths) -> pd.DataFrame | None:
    for path in sorted(paths, key=lambda item: str(item)):
        try:
            return pd.read_csv(path)
        except Exception:
            continue
    return None


def _read_pooled_prediction_csv(model_dir: Path, prefix: str) -> pd.DataFrame | None:
    """Read the aggregate OOF artifact, or reconstruct it from all fold artifacts."""
    paths = sorted(model_dir.glob(f"{prefix}_*_predictions.csv"), key=lambda item: str(item))
    aggregate = [path for path in paths if re.search(r"_fold\d+_predictions\.csv$", path.name) is None]
    if len(aggregate) > 1:
        raise RuntimeError(
            f"Ambiguous aggregate OOF prediction artifacts for {model_dir.name}: "
            + ", ".join(path.name for path in aggregate)
        )
    if aggregate:
        return _read_first_csv(aggregate)
    fold_paths = [path for path in paths if re.search(r"_fold\d+_predictions\.csv$", path.name)]
    fold_tables = [_read_first_csv([path]) for path in fold_paths]
    fold_tables = [table for table in fold_tables if table is not None]
    return pd.concat(fold_tables, ignore_index=True) if fold_tables else None


def _metric_value(df: pd.DataFrame | None, names: list[str]) -> float:
    if df is None or df.empty:
        return np.nan
    for name in names:
        if name in df.columns:
            value = pd.to_numeric(df[name], errors="coerce").dropna()
            if not value.empty:
                return float(value.iloc[0])
    if {"Metric", "Score"}.issubset(df.columns):
        lookup = {str(metric).strip().lower(): score for metric, score in zip(df["Metric"], df["Score"])}
        for name in names:
            value = pd.to_numeric(pd.Series([lookup.get(name.lower())]), errors="coerce").dropna()
            if not value.empty:
                return float(value.iloc[0])
    return np.nan


def _metric_mean(df: pd.DataFrame | None, names: list[str]) -> float:
    values = _metric_series(df, names)
    return float(values.mean()) if not values.empty else np.nan


def _metric_series(df: pd.DataFrame | None, names: list[str]) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").dropna()
    return pd.Series(dtype=float)


def _fill_classification_from_predictions(row: dict[str, Any], predictions: pd.DataFrame) -> None:
    if not {"Original Label", "Predicted Label"}.issubset(predictions.columns):
        return
    y_true = predictions["Original Label"].astype(str)
    y_pred = predictions["Predicted Label"].astype(str)
    if y_true.empty:
        return
    row["Accuracy"] = _coalesce(row["Accuracy"], float((y_true == y_pred).mean()))
    try:
        from sklearn.metrics import f1_score

        row["Weighted F1"] = _coalesce(row["Weighted F1"], float(f1_score(y_true, y_pred, average="weighted", zero_division=0)))
    except Exception:
        pass
    sensitivity, specificity = _classification_sensitivity_specificity(predictions)
    row["Sensitivity"] = _coalesce(row.get("Sensitivity", np.nan), sensitivity)
    row["Specificity"] = _coalesce(row.get("Specificity", np.nan), specificity)


def _oof_auc_from_predictions(predictions: pd.DataFrame) -> tuple[float, float]:
    from sklearn.metrics import roc_auc_score
    probability_columns = [column for column in predictions.columns if str(column).startswith("Probability_")]
    if not probability_columns or "Original Label" not in predictions.columns:
        return np.nan, np.nan
    labels = [str(column).removeprefix("Probability_") for column in probability_columns]
    y_text = predictions["Original Label"].astype(str)
    probabilities = predictions[probability_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if probabilities.shape[1] != len(labels) or not np.isfinite(probabilities).all():
        return np.nan, np.nan
    encoded = pd.Categorical(y_text, categories=labels).codes
    if (encoded < 0).any():
        return np.nan, np.nan
    try:
        if len(labels) == 2:
            value = float(roc_auc_score(encoded, probabilities[:, 1]))
            return value, value
        macro = float(roc_auc_score(encoded, probabilities, multi_class="ovr", average="macro"))
        micro = float(roc_auc_score(encoded, probabilities, multi_class="ovr", average="micro"))
        return macro, micro
    except Exception:
        return np.nan, np.nan


def _classification_sensitivity_specificity(predictions: pd.DataFrame) -> tuple[float, float]:
    """Derive display metrics from completed held-out predictions without refitting."""
    if not {"Original Label", "Predicted Label"}.issubset(predictions.columns):
        return np.nan, np.nan
    y_true = predictions["Original Label"].astype(str)
    y_pred = predictions["Predicted Label"].astype(str)
    observed = set(y_true) | set(y_pred)
    probability_order = [
        column.removeprefix("Probability_")
        for column in predictions.columns
        if str(column).startswith("Probability_")
    ]
    labels = list(dict.fromkeys(probability_order))
    labels.extend(sorted(observed - set(labels)))
    if len(labels) < 2:
        return np.nan, np.nan

    return classification_sensitivity_specificity(y_true, y_pred, labels)


def _write_classification_publication_metrics(
    model_dir: Path,
    prefix: str,
    metrics: dict[str, Any],
) -> None:
    """Write one report-facing six-metric artifact from existing evaluation outputs."""
    labels = ["Accuracy", "Weighted F1", "OOF AUC", "OOF Micro AUC", "Sensitivity", "Specificity"]
    values = [_finite_float(metrics.get(label, np.nan)) for label in labels]
    if not all(np.isfinite(value) for value in values):
        return
    stem = f"{prefix}_{model_dir.name.lower()}_publication_metrics"
    table = pd.DataFrame(
        [
            {
                "Model": metrics.get("Model", model_dir.name),
                "Accuracy": values[0],
                "weighted_F1": values[1],
                "OOF_AUC": values[2],
                "Macro_AUC": values[2],
                "Micro_AUC": values[3],
                "MetricScope": "Pooled outer-fold out-of-fold predictions",
                "Sensitivity": values[4],
                "Specificity": values[5],
            }
        ]
    )
    table.to_csv(model_dir / f"{stem}.csv", index=False)
    plot_metric_bars(
        labels,
        values,
        f"{str(metrics.get('Model', model_dir.name)).replace('_', ' ').title()} Evaluation Metrics",
        model_dir / f"{stem}.png",
        ylim=(0.0, 1.12),
    )


def _read_roc_auc_dict(model_dir: Path, prefix: str) -> dict[str, float]:
    path = next(iter(sorted(model_dir.glob(f"{prefix}_*_roc_data.npy"), key=lambda item: str(item))), None)
    if path is None:
        return {}
    try:
        data = np.load(path, allow_pickle=True).item()
    except Exception:
        return {}
    aucs = data.get("roc_auc", {}) if isinstance(data, dict) else {}
    numeric = [_finite_float(value) for key, value in aucs.items() if isinstance(key, (int, np.integer))]
    numeric = [value for value in numeric if np.isfinite(value)]
    single = numeric[0] if len(numeric) == 1 else float(np.nanmean(numeric)) if numeric else np.nan
    return {
        "macro": _finite_float(aucs.get("macro", np.nan)),
        "micro": _finite_float(aucs.get("micro", np.nan)),
        "single": single,
    }


def _read_model_roc(model_dir: Path, prefix: str, model: str) -> list[dict[str, Any]]:
    path = next(iter(sorted(model_dir.glob(f"{prefix}_*_roc_data.npy"), key=lambda item: str(item))), None)
    if path is None:
        return []
    try:
        data = np.load(path, allow_pickle=True).item()
    except Exception:
        return []
    fpr = data.get("fpr", {})
    tpr = data.get("tpr", {})
    aucs = data.get("roc_auc", {})
    if all(key in fpr and key in tpr for key in ("macro", "micro")):
        return [
            {
                "fpr": fpr[roc_type],
                "tpr": tpr[roc_type],
                "auc": _finite_float(aucs.get(roc_type, np.nan)),
                "label": f"{model} {roc_type.title()}",
                "model": model,
                "roc_type": roc_type,
            }
            for roc_type in ("macro", "micro")
        ]
    keys = [item for item in fpr.keys() if item in tpr]
    if not keys:
        return []
    key = keys[0]
    return [{"fpr": fpr[key], "tpr": tpr[key], "auc": _finite_float(aucs.get(key, np.nan)), "label": model, "model": model, "roc_type": "binary"}]


def _read_regression_predictions(model_dir: Path, prefix: str, model: str) -> pd.DataFrame | None:
    predictions = _read_pooled_prediction_csv(model_dir, prefix)
    if predictions is None:
        return None
    true_col = "Original Label" if "Original Label" in predictions.columns else "Actual Value" if "Actual Value" in predictions.columns else None
    pred_col = "Predicted Label" if "Predicted Label" in predictions.columns else "Predicted Value" if "Predicted Value" in predictions.columns else None
    if true_col is None or pred_col is None:
        return None
    out = predictions[[true_col, pred_col]].copy()
    out.columns = ["Original Label", "Predicted Label"]
    out["Model"] = model
    out["Original Label"] = pd.to_numeric(out["Original Label"], errors="coerce")
    out["Predicted Label"] = pd.to_numeric(out["Predicted Label"], errors="coerce")
    out = out.dropna()
    return out if not out.empty else None


def _plot_all_model_roc(curves: list[dict[str, Any]], title: str, output: str) -> None:
    if not curves:
        return
    configure_publication_plots()
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    dense_legend = len(curves) > 10
    fig, ax = plt.subplots(figsize=(8.6, 6.2) if dense_legend else (6.6, 5.4))
    models = list(dict.fromkeys(str(curve.get("model", curve["label"])) for curve in curves))
    palette = (
        list(PUBLICATION_COLORS)
        if len(models) <= len(PUBLICATION_COLORS)
        else [mpl.colormaps["tab20"].resampled(len(models))(index) for index in range(len(models))]
    )
    model_colors = {model: palette[index] for index, model in enumerate(models)}
    for curve in curves:
        label = str(curve["label"])
        auc_value = curve.get("auc")
        if np.isfinite(_finite_float(auc_value)):
            label = f"{label} (AUC = {_finite_float(auc_value):.2f})"
        model = str(curve.get("model", curve["label"]))
        roc_type = str(curve.get("roc_type", "binary")).lower()
        linestyle = "--" if roc_type == "macro" else ":" if roc_type == "micro" else "-"
        linewidth = 2.2 if roc_type in {"macro", "micro"} else 1.8
        ax.plot(
            curve["fpr"],
            curve["tpr"],
            label=label,
            color=model_colors[model],
            linestyle=linestyle,
            linewidth=linewidth,
        )
    ax.plot([0, 1], [0, 1], color="#6B7280", linestyle="--", linewidth=1.0)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    if dense_legend:
        ax.legend(frameon=False, fontsize=6.8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    else:
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    style_publication_axis(ax, grid_axis=None)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _plot_ranking(df: pd.DataFrame, metric: str, title: str, output: str, *, higher_is_better: bool, ylim: tuple[float, float] | None = None) -> None:
    values = pd.to_numeric(df[metric], errors="coerce")
    plot_df = df.assign(_metric=values).dropna(subset=["_metric"]).sort_values("_metric", ascending=not higher_is_better)
    if plot_df.empty:
        return
    configure_publication_plots()
    import matplotlib.pyplot as plt

    fig_width = max(6.4, 0.55 * len(plot_df) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_width, 4.6))
    bars = ax.bar(plot_df["Model"], plot_df["_metric"], color=[PUBLICATION_COLORS[i % len(PUBLICATION_COLORS)] for i in range(len(plot_df))], width=0.68)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    style_publication_axis(ax)
    if ylim is not None:
        ax.set_ylim(*ylim)
    add_bar_value_labels(ax, bars, plot_df["_metric"], adjust_limits=ylim is None)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _plot_overall_performance_summary(
    df: pd.DataFrame,
    metrics: list[str],
    title: str,
    output: str,
    *,
    higher_is_better: dict[str, bool],
) -> None:
    configure_publication_plots()
    import matplotlib.pyplot as plt

    panel_count = len(metrics)
    if panel_count == 6:
        nrows, ncols = 2, 3
    elif panel_count == 4:
        nrows, ncols = 2, 2
    else:
        nrows, ncols = 1, panel_count
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.0, 7.6 if nrows == 2 else 4.4))
    axes_array = np.asarray(axes).reshape(-1)
    for index, metric in enumerate(metrics):
        ax = axes_array[index]
        values = pd.to_numeric(df.get(metric), errors="coerce")
        plot_df = df.assign(_metric=values).dropna(subset=["_metric"])
        plot_df = plot_df.sort_values("_metric", ascending=not higher_is_better.get(metric, True))
        if plot_df.empty:
            ax.axis("off")
            continue
        bars = ax.bar(
            plot_df["Model"],
            plot_df["_metric"],
            color=PUBLICATION_COLORS[index % len(PUBLICATION_COLORS)],
            width=0.68,
        )
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        if metric in {"Accuracy", "Weighted F1", "OOF AUC", "OOF Micro AUC", "MeanFold AUC", "SDFold AUC", "Sensitivity", "Specificity"}:
            ax.set_ylim(0.0, 1.12)
            add_bar_value_labels(ax, bars, plot_df["_metric"], adjust_limits=False)
        else:
            add_bar_value_labels(ax, bars, plot_df["_metric"])
        style_publication_axis(ax)
    for ax in axes_array[panel_count:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output)
    plt.close(fig)


def _plot_prediction_comparison(predictions: list[pd.DataFrame], title: str, output: str) -> None:
    if not predictions:
        return
    configure_publication_plots()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 5.3))
    all_values = []
    for index, df in enumerate(predictions):
        all_values.extend(df["Original Label"].tolist())
        all_values.extend(df["Predicted Label"].tolist())
        ax.scatter(
            df["Original Label"],
            df["Predicted Label"],
            label=str(df["Model"].iloc[0]),
            s=28,
            alpha=0.72,
            color=PUBLICATION_COLORS[index % len(PUBLICATION_COLORS)],
        )
    finite = np.asarray([value for value in all_values if np.isfinite(value)], dtype=float)
    if finite.size:
        lo, hi = float(np.min(finite)), float(np.max(finite))
        padding = (hi - lo) * 0.05 if hi > lo else 1.0
        ax.plot([lo - padding, hi + padding], [lo - padding, hi + padding], color="#4B5563", linestyle="--", linewidth=1.1)
        ax.set_xlim(lo - padding, hi + padding)
        ax.set_ylim(lo - padding, hi + padding)
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, loc="best")
    style_publication_axis(ax)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _write_table_outputs(prefix: str, filename: str, df: pd.DataFrame) -> None:
    prefixed = f"{prefix}_{filename}"
    df.to_csv(prefixed, index=False)
    write_artifact_alias(prefixed, filename)


def _write_json_outputs(prefix: str, filename: str, payload: dict[str, Any]) -> None:
    prefixed = f"{prefix}_{filename}"
    Path(prefixed).write_text(json.dumps(_json_clean(payload), indent=2), encoding="utf-8")
    write_artifact_alias(prefixed, filename)


def _alias_benchmark_outputs(prefix: str, suffixes: list[str]) -> None:
    for suffix in suffixes:
        source = Path(f"{prefix}_{suffix}")
        write_artifact_alias(source, suffix)


def _classification_reason(best: dict[str, Any]) -> str:
    return (
        f"{best['Model']} had the strongest benchmark rank using pooled OOF AUC, Weighted F1, "
        f"and Accuracy from completed prediction outputs."
    )


def _regression_reason(best: dict[str, Any]) -> str:
    return f"{best['Model']} had the lowest RMSE among completed prediction outputs, with R2 used for tie-breaking."


def _has_any_metric(row: dict[str, Any], names: list[str]) -> bool:
    return any(np.isfinite(_finite_float(row.get(name, np.nan))) for name in names)


def _coalesce(*values) -> float:
    for value in values:
        number = _finite_float(value)
        if np.isfinite(number):
            return number
    return np.nan


def _finite_float(value) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else np.nan
    except Exception:
        return np.nan


def _json_metrics(row: dict[str, Any], names: list[str]) -> dict[str, float | None]:
    return {name: _json_clean(_finite_float(row.get(name, np.nan))) for name in names}


def _json_clean(value):
    if isinstance(value, dict):
        return {key: _json_clean(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_clean(val) for val in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BiomarkerML overall model benchmark outputs.")
    add_common_cli_arguments(parser, require_input=True)
    parser.add_argument("--task", required=True, choices=["classification", "regression"])
    args = parser.parse_args()
    run_overall_benchmark_task(args)


if __name__ == "__main__":
    main()
