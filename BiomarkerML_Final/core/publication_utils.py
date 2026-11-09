from __future__ import annotations

import glob
import inspect
import json
import os
import re
import shutil
import tempfile
import warnings
import filecmp
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

DEFAULT_RANDOM_STATE = 42
DEFAULT_DPI = 350
PLOT_FORMATS = ("png", "pdf", "tiff")
PUBLICATION_FONT_FALLBACK = ("Arial", "Liberation Sans", "DejaVu Sans", "sans-serif")
RUNTIME_CACHE_DIR = Path(os.environ.get("BIOMARKERML_CACHE_DIR", Path(tempfile.gettempdir()) / "biomarkerml_cache"))
RUNTIME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(RUNTIME_CACHE_DIR / "xdg"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
TRAINING_HISTORY_COLUMNS = [
    "Epoch",
    "TrainingLoss",
    "ValidationLoss",
    "LearningRate",
    "EarlyStoppingTriggered",
    "BestEpoch",
]
TASK_OUTPUT_DIRS = {
    "preprocess": "preprocess",
    "classification": "classification",
    "regression": "regression",
    "overall_benchmark": "overall_benchmark",
    "classification_shap": "classification_shap",
    "regression_shap": "regression_shap",
    "biomarker_reliability": "biomarker_reliability",
    "ppi": "ppi",
    "pdf_report": "pdf_report",
    "resource_report": "resource_report",
    "workflow": "workflow",
    "paper_benchmark": "benchmark",
    "figure_builder": "figures",
    "supplementary": "supplementary",
    "paper": "paper",
}
WORKFLOW_RUN_DIRS = (
    "workflow",
    "preprocess",
    "classification",
    "regression",
    "overall_benchmark",
    "classification_shap",
    "regression_shap",
    "biomarker_reliability",
    "ppi",
    "pdf_report",
    "resource_report",
    "cache",
    "temp",
)
LEGACY_REDUCTION_METHODS = {"pca", "kpca", "umap", "tsne", "pls"}
LEGACY_SELECTION_METHODS = {"elasticnet", "boruta"}
NO_SELECTION_VALUES = {None, "", "none", "no", "false", "0", "off"}
YES_SELECTION_VALUES = {"yes", "true", "1", "on", "auto"}
TRUE_SHAP_SOURCE_NAMES = {
    "shap",
    "shapvalue",
    "shapvalues",
    "shapattribution",
    "shapattributions",
}
TREE_MODELS = {
    "rf",
    "random_forest",
    "random_forest_reg",
    "xgb",
    "xgboost",
    "xgboost_reg",
    "lgbm",
    "lightgbm",
    "lightgbm_reg",
}
STANDALONE_MODELS = {
    "plsda",
    "pls_reg",
    "vae",
    "vae_reg",
    "vae_mlp_reg",
    "mlpvae",
    "mlpvae_reg",
    "mlp_in_vae_reg",
}
MLP_HIDDEN_LAYER_ARCHITECTURES = {
    "single_32": (32,),
    "single_64": (64,),
    "single_128": (128,),
    "stacked_64_32": (64, 32),
}
COMPONENT_PREFIXES = (
    "PCA_Component",
    "KPCA_Component",
    "UMAP_Component",
    "TSNE_Component",
    "tSNE_Component",
    "PLS_Component",
    "Component_",
)


def decode_mlp_hidden_layer_sizes(value) -> tuple[int, ...]:
    """Decode Optuna's storage-safe MLP architecture label without changing the search set."""
    if isinstance(value, str):
        try:
            return MLP_HIDDEN_LAYER_ARCHITECTURES[value]
        except KeyError as exc:
            raise ValueError(f"Unknown MLP hidden-layer architecture: {value}") from exc
    if isinstance(value, (tuple, list, np.ndarray)):
        return tuple(int(width) for width in value)
    raise TypeError(f"Unsupported MLP hidden-layer architecture value: {value!r}")


def configure_publication_plots() -> None:
    import matplotlib as mpl
    import matplotlib.figure

    font_family = resolve_publication_font()
    mpl.rcParams.update(
        {
            "figure.dpi": DEFAULT_DPI,
            "savefig.dpi": DEFAULT_DPI,
            "savefig.bbox": "tight",
            "font.family": font_family,
            "font.sans-serif": list(PUBLICATION_FONT_FALLBACK),
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.linewidth": 0.9,
            "lines.linewidth": 1.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    if getattr(matplotlib.figure.Figure.savefig, "_biomarkerml_patched", False):
        return

    original_savefig = matplotlib.figure.Figure.savefig

    def savefig_all_formats(self, fname, *args, **kwargs):
        path = Path(str(fname))
        suffix = path.suffix.lower().lstrip(".")
        save_kwargs = dict(kwargs)
        if suffix in {"png", "tif", "tiff"} and save_kwargs.get("dpi", DEFAULT_DPI) in (None, 300, "figure"):
            save_kwargs["dpi"] = DEFAULT_DPI
        original_savefig(self, fname, *args, **save_kwargs)

        if suffix == "png":
            for fmt in ("pdf", "tiff"):
                alt = path.with_suffix(f".{fmt}")
                alt_kwargs = dict(save_kwargs)
                if fmt == "pdf":
                    alt_kwargs.pop("dpi", None)
                else:
                    alt_kwargs["dpi"] = DEFAULT_DPI
                original_savefig(self, alt, *args, **alt_kwargs)

    savefig_all_formats._biomarkerml_patched = True
    matplotlib.figure.Figure.savefig = savefig_all_formats


def resolve_publication_font() -> str:
    import matplotlib.font_manager as font_manager

    for family in PUBLICATION_FONT_FALLBACK:
        if family == "sans-serif":
            return family
        try:
            font_manager.findfont(family, fallback_to_default=False)
            return family
        except Exception:
            continue
    return "sans-serif"


def set_global_seed(seed: int = DEFAULT_RANDOM_STATE) -> int:
    import random

    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    except Exception:
        pass
    return seed


def build_output_prefix(input_path: str | os.PathLike[str] | None, prefix: str | None = None) -> str:
    if prefix:
        return str(prefix)
    if input_path:
        return Path(input_path).stem
    return "biomarkerml"


def resolve_workflow_run_directory(
    outdir: str | os.PathLike[str] | None,
    prefix: str | None = None,
) -> tuple[str, Path]:
    base = Path(outdir or "results").resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    requested_prefix = str(prefix).strip() if prefix else ""
    run_prefix = requested_prefix or f"Result_{timestamp}"
    run_dir = base / run_prefix

    if requested_prefix and run_dir.exists():
        run_prefix = f"{requested_prefix}_{timestamp}"
        run_dir = base / run_prefix

    if run_dir.exists():
        stem = run_prefix
        index = 2
        while run_dir.exists():
            run_prefix = f"{stem}_{index}"
            run_dir = base / run_prefix
            index += 1

    for subdir in WORKFLOW_RUN_DIRS:
        (run_dir / subdir).mkdir(parents=True, exist_ok=False)
    return run_prefix, run_dir


def task_output_dir(outdir: str | os.PathLike[str] | None, task: str) -> Path:
    base = Path(outdir or ".").resolve()
    subdir = TASK_OUTPUT_DIRS.get(task, task)
    path = base / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_output_dir(outdir: str | os.PathLike[str] | None, task: str, model: str) -> Path:
    path = task_output_dir(outdir, task) / str(model)
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_path(outdir: str | os.PathLike[str] | None, task: str, filename: str | os.PathLike[str]) -> Path:
    path = task_output_dir(outdir, task) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def task_run_context(outdir: str | os.PathLike[str] | None, task: str):
    path = task_output_dir(outdir, task)
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(previous)


def stage_task_inputs(
    input_path: str | os.PathLike[str] | None,
    output_dir: str | os.PathLike[str],
    prefix: str | None = None,
    *,
    patterns: Sequence[str] | None = None,
) -> None:
    """Stage run-owned artifacts without recursive first-file-wins discovery.

    Task outputs are stored either directly in a task directory or in one model
    subdirectory. Deeper paths are not part of this contract. Flattening is
    retained for legacy consumers, but ambiguous basenames fail explicitly.
    """
    if not input_path:
        return
    source_input = Path(input_path)
    if source_input.is_symlink() and not source_input.exists():
        raise RuntimeError(f"Cannot stage broken source symlink '{source_input}'.")
    source = source_input.resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if source.is_file():
        _stage_one_file(source, destination / source.name)
        return
    if not source.is_dir():
        return
    artifact_patterns = tuple(patterns or (f"{prefix}*" if prefix else "*",))
    candidates: dict[str, Path] = {}
    search_roots = [source, *(path for path in source.iterdir() if path.is_dir())]
    for search_root in search_roots:
        for pattern in artifact_patterns:
            for path in search_root.glob(pattern):
                if (
                    path.name.endswith(".biomarkerml-stage.json")
                    or not path.is_file()
                    or path.resolve().parent == destination
                ):
                    continue
                existing = candidates.get(path.name)
                if existing is not None and existing.resolve() != path.resolve():
                    raise RuntimeError(
                        "Ambiguous staged artifact basename "
                        f"'{path.name}' from '{existing}' and '{path}'."
                    )
                candidates[path.name] = path
    for name in sorted(candidates):
        destination_path = destination / name
        _stage_one_file(candidates[name], destination_path)


def _stage_one_file(source: Path, destination: Path) -> None:
    source = source.resolve(strict=True)
    provenance = destination.parent / f".{destination.name}.biomarkerml-stage.json"

    if destination.is_symlink():
        if not destination.exists():
            raise RuntimeError(f"Staged destination is a broken symlink: '{destination}'.")
        if destination.resolve(strict=True) == source:
            return
        raise RuntimeError(
            f"Staged destination '{destination}' points to a different source than '{source}'."
        )

    if destination.exists():
        if destination.is_file() and os.path.samefile(source, destination):
            return
        if destination.is_file() and provenance.is_file():
            recorded = json.loads(provenance.read_text(encoding="utf-8"))
            source_stat = source.stat()
            if (
                recorded.get("source") == str(source)
                and recorded.get("device") == source_stat.st_dev
                and recorded.get("inode") == source_stat.st_ino
                and filecmp.cmp(source, destination, shallow=False)
            ):
                return
        raise RuntimeError(
            f"Staged destination '{destination}' already exists but does not represent "
            f"the requested source '{source}'."
        )

    if provenance.exists() or provenance.is_symlink():
        raise RuntimeError(f"Stale staging provenance exists for absent destination '{destination}'.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination == source:
        return
    try:
        destination.symlink_to(source)
    except OSError:
        copy_created = False
        try:
            with source.open("rb") as source_handle, destination.open("xb") as destination_handle:
                copy_created = True
                shutil.copyfileobj(source_handle, destination_handle)
            shutil.copystat(source, destination)
        except Exception:
            if copy_created and destination.exists() and not provenance.exists():
                destination.unlink()
            raise
        source_stat = source.stat()
        record = {
            "source": str(source),
            "device": source_stat.st_dev,
            "inode": source_stat.st_ino,
        }
        provenance.write_text(json.dumps(record, indent=2), encoding="utf-8")


def write_artifact_alias(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
    source_path = Path(source)
    target_path = Path(target)
    if not source_path.exists() or source_path.resolve() == target_path.resolve():
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    if source_path.suffix.lower() == ".png" and target_path.suffix.lower() == ".png":
        for suffix in (".pdf", ".tiff"):
            sibling = source_path.with_suffix(suffix)
            if sibling.exists():
                shutil.copy2(sibling, target_path.with_suffix(suffix))


PUBLICATION_COLORS = ("#4C78A8", "#F58518", "#59A14F", "#B279A2", "#E45756", "#72B7B2")


def format_effect_value(value: float, *, digits: int = 2) -> str:
    try:
        number = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(number):
        return ""
    if number == 0:
        return "0.00"
    magnitude = abs(number)
    if magnitude < 0.001:
        return f"{number:.2e}"
    if magnitude < 0.1:
        return f"{number:.4g}"
    return f"{number:.3f}"


def meaningful_display_mask(
    values,
    *,
    absolute_epsilon: float = 1e-12,
    relative_epsilon: float = 1e-6,
) -> np.ndarray:
    """Select finite, non-negligible values for publication display only."""
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(numeric)
    if not finite.any():
        return np.zeros(len(numeric), dtype=bool)
    maximum = float(np.max(np.abs(numeric[finite])))
    threshold = max(float(absolute_epsilon), maximum * float(relative_epsilon))
    return finite & (np.abs(numeric) > threshold)


def style_publication_axis(ax, *, grid_axis: str | None = "y") -> None:
    if grid_axis:
        ax.grid(axis=grid_axis, color="#E5E7EB", linewidth=0.7, alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_average_shap_per_class(
    class_df: pd.DataFrame,
    class_name: str,
    output: str | os.PathLike[str],
    *,
    top_n: int = 20,
) -> None:
    required = {"Feature", "MeanSHAP"}
    if class_df.empty or not required.issubset(class_df.columns):
        return

    plot_df = class_df.copy()
    plot_df["MeanSHAP"] = pd.to_numeric(plot_df["MeanSHAP"], errors="coerce")
    plot_df = plot_df.loc[meaningful_display_mask(plot_df["MeanSHAP"])].copy()
    if plot_df.empty:
        output_path = Path(output)
        for path in (output_path, output_path.with_suffix(".pdf"), output_path.with_suffix(".tiff")):
            if path.exists():
                path.unlink()
        return
    if "RankWithinClass" in plot_df.columns:
        plot_df = plot_df.sort_values(["RankWithinClass", "Feature"], kind="stable")
    else:
        plot_df = plot_df.assign(_magnitude=plot_df["MeanSHAP"].abs()).sort_values(
            ["_magnitude", "Feature"], ascending=[False, True], kind="stable"
        )
    plot_df = plot_df.head(int(top_n)).sort_values(["MeanSHAP", "Feature"], kind="stable")
    if plot_df.empty:
        return

    configure_publication_plots()
    import matplotlib.pyplot as plt

    values = plot_df["MeanSHAP"].to_numpy(dtype=float)
    colors = np.where(values >= 0, "#B2182B", "#2166AC")
    fig, ax = plt.subplots(figsize=(8.6, max(4.8, 0.34 * len(plot_df))))
    display_column = "DisplayFeature" if "DisplayFeature" in plot_df.columns else "Feature"
    bars = ax.barh(plot_df[display_column].astype(str), values, color=colors, alpha=0.92, height=0.72)
    ax.axvline(0, color="#1F2937", linewidth=1.0)
    ax.set_xlabel("Mean signed SHAP value")
    ax.set_ylabel("")
    ax.set_title(f"Average SHAP per Class\n{class_name}\nTop {len(plot_df)} Biomarkers")
    limit = float(np.nanmax(np.abs(values))) if values.size else 0.0
    limit = limit * 1.25 if limit > 0 else 1.0
    ax.set_xlim(-limit, limit)
    style_publication_axis(ax, grid_axis="x")

    if len(plot_df) <= 30:
        offset = limit * 0.025
        for bar, value in zip(bars, values):
            x = value + offset if value >= 0 else value - offset
            ax.text(
                x,
                bar.get_y() + bar.get_height() / 2,
                format_effect_value(value),
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=9,
            )
    ax.text(
        0.0,
        -0.13,
        "Positive values increase prediction toward this class; negative values decrease prediction toward this class.",
        transform=ax.transAxes,
        fontsize=9,
        ha="left",
        va="top",
    )
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def add_bar_value_labels(ax, bars, values: Sequence[float] | None = None, fmt: str = "{:.2f}", *, adjust_limits: bool = True) -> None:
    heights = [float(bar.get_height()) for bar in bars]
    values = heights if values is None else [float(v) for v in values]
    if not heights:
        return
    low = min(0.0, min(heights))
    high = max(0.0, max(heights))
    span = high - low if high > low else 1.0
    offset = span * 0.025
    for bar, value, height in zip(bars, values, heights):
        y = height + offset if height >= 0 else height - offset
        va = "bottom" if height >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            fmt.format(value),
            ha="center",
            va=va,
            fontsize=9,
        )
    if adjust_limits:
        ax.set_ylim(low - span * 0.12, high + span * 0.18)


def plot_metric_bars(
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    output: str | os.PathLike[str],
    *,
    ylabel: str = "Score",
    ylim: tuple[float, float] | None = None,
    colors: Sequence[str] | None = None,
    fmt: str = "{:.2f}",
) -> None:
    configure_publication_plots()
    import matplotlib.pyplot as plt

    numeric = [0.0 if pd.isna(v) or not np.isfinite(float(v)) else float(v) for v in values]
    palette = list(colors or PUBLICATION_COLORS)
    fig_width = max(6.4, 2.0 + 1.05 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 4.4))
    bars = ax.bar(labels, numeric, color=[palette[i % len(palette)] for i in range(len(labels))], width=0.68)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    style_publication_axis(ax)
    if ylim is not None:
        ax.set_ylim(*ylim)
    add_bar_value_labels(ax, bars, numeric, fmt=fmt, adjust_limits=ylim is None)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, labels: Sequence[object], title: str, output: str | os.PathLike[str]) -> None:
    configure_publication_plots()
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    matrix = np.asarray(cm)
    if np.allclose(matrix, np.round(matrix)):
        matrix = np.round(matrix).astype(int)
        values_format = "d"
    else:
        values_format = ".2f"

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    disp = ConfusionMatrixDisplay(matrix, display_labels=[str(label) for label in labels])
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format=values_format)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_roc_curves(curves: Sequence[dict], title: str, output: str | os.PathLike[str]) -> None:
    configure_publication_plots()
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    dense_legend = len(curves) > 6
    fig, ax = plt.subplots(figsize=(7.8, 5.4) if dense_legend else (6.2, 5.2))
    class_curve_count = sum("macro" not in str(curve.get("label", "")).lower() and "micro" not in str(curve.get("label", "")).lower() for curve in curves)
    class_colors = (
        list(PUBLICATION_COLORS)
        if class_curve_count <= len(PUBLICATION_COLORS)
        else [mpl.colormaps["tab20"].resampled(class_curve_count)(index) for index in range(class_curve_count)]
    )
    class_index = 0
    for i, curve in enumerate(curves):
        label = curve.get("label", f"Curve {i + 1}")
        label_key = str(label).lower()
        auc_value = curve.get("auc")
        if auc_value is not None and np.isfinite(float(auc_value)):
            label = f"{label} (AUC = {float(auc_value):.2f})"
        if "macro" in label_key:
            color, linestyle, linewidth = "#111827", "--", 2.4
        elif "micro" in label_key:
            color, linestyle, linewidth = "#111827", ":", 2.4
        else:
            color = curve.get("color", class_colors[class_index % len(class_colors)])
            linestyle = curve.get("linestyle", "-")
            linewidth = float(curve.get("linewidth", 1.8))
            class_index += 1
        ax.plot(curve["fpr"], curve["tpr"], lw=linewidth, color=color, linestyle=linestyle, label=label)
    ax.plot([0, 1], [0, 1], color="#6B7280", lw=1, linestyle="--")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    if dense_legend:
        ax.legend(frameon=False, fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
    else:
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    style_publication_axis(ax, grid_axis=None)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_classification_nested_cv(
    folds: Sequence[int],
    f1_values: Sequence[float],
    auc_values: Sequence[float],
    title: str,
    output: str | os.PathLike[str],
    *,
    f1_label: str = "Weighted F1",
) -> None:
    configure_publication_plots()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(folds, f1_values, marker="o", label=f1_label, color=PUBLICATION_COLORS[0])
    ax.plot(folds, auc_values, marker="s", label="AUC", color=PUBLICATION_COLORS[1])
    ax.set_xlabel("Outer fold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(frameon=False)
    style_publication_axis(ax)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_regression_nested_cv(fold_metrics: pd.DataFrame, title: str, output: str | os.PathLike[str]) -> None:
    configure_publication_plots()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(fold_metrics["Fold"], fold_metrics["RMSE"], marker="o", label="RMSE", color=PUBLICATION_COLORS[0])
    ax.plot(fold_metrics["Fold"], fold_metrics["MAE"], marker="s", label="MAE", color=PUBLICATION_COLORS[1])
    ax.set_xlabel("Outer fold")
    ax.set_ylabel("Error")
    ax2 = ax.twinx()
    ax2.plot(fold_metrics["Fold"], fold_metrics["R2"], marker="^", label="R2", color=PUBLICATION_COLORS[2])
    ax2.set_ylabel("R2")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, loc="best")
    ax.set_title(title)
    style_publication_axis(ax)
    ax2.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_observed_predicted(y_true, y_pred, title: str, output: str | os.PathLike[str]) -> None:
    configure_publication_plots()
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    ax.scatter(y_true, y_pred, s=34, color=PUBLICATION_COLORS[0], alpha=0.85)
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], color="#6B7280", linestyle="--", lw=1)
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    style_publication_axis(ax)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_residuals(y_true, y_pred, title: str, output: str | os.PathLike[str]) -> None:
    configure_publication_plots()
    import matplotlib.pyplot as plt

    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    ax.scatter(y_pred, residuals, s=34, color=PUBLICATION_COLORS[2], alpha=0.85)
    ax.axhline(0, color="#6B7280", linestyle="--", lw=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.set_title(title)
    style_publication_axis(ax)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def write_training_history(rows: Iterable[dict], prefix: str, model_key: str) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    if df.empty:
        df = pd.DataFrame(columns=TRAINING_HISTORY_COLUMNS)
    for column in TRAINING_HISTORY_COLUMNS:
        if column not in df.columns:
            df[column] = False if column == "EarlyStoppingTriggered" else np.nan
    df = df[TRAINING_HISTORY_COLUMNS]
    df.to_csv(f"{prefix}_{model_key}_training_history.csv", index=False)
    if not Path("training_history.csv").exists():
        df.to_csv("training_history.csv", index=False)
    plot_training_history(df, f"{model_key.replace('_', ' ').title()} Training History", f"{prefix}_{model_key}_training_history_curve.png")
    learning = df.rename(columns={"Epoch": "ProgressionStep"}).copy()
    learning["ProgressionValue"] = learning["ProgressionStep"]
    learning["ProgressionLabel"] = "Epoch"
    learning["DiagnosticType"] = "Epoch-wise optimization"
    write_learning_progression(learning, prefix, model_key)
    return df


def plot_training_history(df: pd.DataFrame, title: str, output: str | os.PathLike[str]) -> None:
    if df.empty or "Epoch" not in df.columns:
        return
    plot_df = df.copy()
    for column in ["Epoch", "TrainingLoss", "ValidationLoss", "LearningRate", "BestEpoch"]:
        if column in plot_df.columns:
            plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    plot_df = plot_df.dropna(subset=["Epoch"])
    if plot_df.empty:
        return
    grouped = plot_df.groupby("Epoch", as_index=False)[["TrainingLoss", "ValidationLoss", "LearningRate", "BestEpoch"]].mean(numeric_only=True)
    if grouped[["TrainingLoss", "ValidationLoss"]].dropna(how="all").empty:
        return
    configure_publication_plots()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    if "TrainingLoss" in grouped.columns and grouped["TrainingLoss"].notna().any():
        ax.plot(grouped["Epoch"], grouped["TrainingLoss"], color=PUBLICATION_COLORS[0], marker="o", markersize=3.0, label="Training loss")
    if "ValidationLoss" in grouped.columns and grouped["ValidationLoss"].notna().any():
        ax.plot(grouped["Epoch"], grouped["ValidationLoss"], color=PUBLICATION_COLORS[1], marker="s", markersize=3.0, label="Validation loss")
    best_epochs = pd.to_numeric(plot_df.get("BestEpoch", pd.Series(dtype=float)), errors="coerce").dropna()
    if not best_epochs.empty:
        best_epoch = float(best_epochs.iloc[-1])
        if np.isfinite(best_epoch):
            ax.axvline(best_epoch, color="#6B7280", linestyle="--", linewidth=1.0, label="Best epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(frameon=False, loc="best")
    style_publication_axis(ax)
    if "LearningRate" in grouped.columns and grouped["LearningRate"].notna().any() and grouped["LearningRate"].nunique(dropna=True) > 1:
        ax_lr = ax.twinx()
        ax_lr.plot(grouped["Epoch"], grouped["LearningRate"], color=PUBLICATION_COLORS[2], linestyle=":", linewidth=1.3, label="Learning rate")
        ax_lr.set_ylabel("Learning rate")
        ax_lr.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def write_learning_progression(
    rows: Iterable[dict] | pd.DataFrame,
    prefix: str,
    model_key: str,
) -> pd.DataFrame:
    """Write the publication learning curve independently of training-size data."""
    columns = [
        "ProgressionStep",
        "ProgressionValue",
        "TrainingLoss",
        "ValidationLoss",
        "ProgressionLabel",
        "DiagnosticType",
    ]
    df = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    if df.empty:
        return df
    for column in columns:
        if column not in df.columns:
            df[column] = np.nan
    df = df[columns].copy()
    df.to_csv(f"{prefix}_{model_key}_learning_curve.csv", index=False)
    title = _learning_progression_title(df, model_key.replace("_", " ").title())
    plot_learning_progression(
        df,
        title,
        f"{prefix}_{model_key}_learning_curve.png",
    )
    return df


def _learning_progression_title(df: pd.DataFrame, model_label: str) -> str:
    diagnostic = (
        str(df["DiagnosticType"].dropna().iloc[0]).lower()
        if "DiagnosticType" in df.columns and df["DiagnosticType"].notna().any()
        else ""
    )
    progression = (
        str(df["ProgressionLabel"].dropna().iloc[0]).lower()
        if "ProgressionLabel" in df.columns and df["ProgressionLabel"].notna().any()
        else ""
    )
    if "epoch" in diagnostic or "epoch" in progression:
        suffix = "Learning Curve"
    elif "regular" in progression or "regular" in diagnostic:
        suffix = "Regularization Validation"
    else:
        suffix = "Model Complexity Validation"
    return f"{model_label} {suffix}"


def plot_learning_progression(
    df: pd.DataFrame,
    title: str,
    output: str | os.PathLike[str],
) -> None:
    """Render optimization or complexity progression without training-size inputs."""
    required = {"ProgressionStep", "TrainingLoss", "ValidationLoss"}
    if df.empty or not required.issubset(df.columns):
        return
    plot_df = df.copy()
    for column in ["ProgressionStep", "ProgressionValue", "TrainingLoss", "ValidationLoss"]:
        if column in plot_df.columns:
            plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    plot_df = plot_df.dropna(subset=["ProgressionStep", "TrainingLoss", "ValidationLoss"]).sort_values("ProgressionStep")
    if plot_df.empty:
        return

    configure_publication_plots()
    import matplotlib.pyplot as plt

    x = plot_df["ProgressionStep"].to_numpy(dtype=float)
    training = plot_df["TrainingLoss"].to_numpy(dtype=float)
    validation = plot_df["ValidationLoss"].to_numpy(dtype=float)
    x_label = str(plot_df["ProgressionLabel"].dropna().iloc[0]) if plot_df["ProgressionLabel"].notna().any() else "Optimization progression"
    diagnostic_type = str(plot_df["DiagnosticType"].dropna().iloc[0]) if plot_df["DiagnosticType"].notna().any() else ""
    epoch_trained = "epoch" in diagnostic_type.lower() or "epoch" in x_label.lower()
    training_label = "Training loss" if epoch_trained else "Training-set loss"
    validation_label = "Validation loss" if epoch_trained else "Validation-set loss"

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(x, training, marker="o", color=PUBLICATION_COLORS[0], linewidth=1.8, label=training_label)
    ax.plot(x, validation, marker="s", color=PUBLICATION_COLORS[1], linewidth=1.8, label=validation_label)
    best_index = int(np.nanargmin(validation))
    ax.axvline(x[best_index], color="#6B7280", linestyle="--", linewidth=1.0, label="Best observed validation point")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Loss")
    positive_x = x[np.isfinite(x) & (x > 0)]
    if positive_x.size and float(np.max(positive_x) / np.min(positive_x)) >= 100.0:
        ax.set_xscale("log")
    ax.set_title(title)
    if diagnostic_type:
        ax.text(0.01, 0.99, diagnostic_type, transform=ax.transAxes, va="top", ha="left", fontsize=8, color="#4B5563")
    ax.legend(frameon=False, loc="best")
    style_publication_axis(ax)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def ensure_publication_report_alias(canonical_report: str | os.PathLike[str]) -> Path:
    """Normalize the report name and remove a redundant lowercase duplicate."""
    canonical = Path(canonical_report)
    matching_entries = [
        path
        for path in canonical.parent.iterdir()
        if path.name.lower() == canonical.name.lower()
    ]
    actual = matching_entries[0] if matching_entries else canonical
    if actual.exists() and actual.name != canonical.name:
        temporary = canonical.parent / ".publication_report_case_normalization.pdf"
        actual.replace(temporary)
        temporary.replace(canonical)
    for path in canonical.parent.iterdir():
        if path.name.lower() == canonical.name.lower() and path.name != canonical.name:
            path.unlink()
    return canonical


def generate_training_size_curve(
    X,
    y,
    model_factory,
    prefix: str,
    model_key: str,
    *,
    task: str,
    fractions: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 1.0),
    X_validation=None,
    y_validation=None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    from sklearn.metrics import f1_score, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X)
    y = np.asarray(y)
    if X.shape[0] < 4:
        return pd.DataFrame()

    if X_validation is None or y_validation is None:
        validation_fraction = 0.25 if X.shape[0] >= 8 else 0.5
        stratify = _safe_stratify_labels(y) if task == "classification" else None
        try:
            train_idx, val_idx = train_test_split(
                np.arange(X.shape[0]),
                test_size=validation_fraction,
                random_state=random_state,
                stratify=stratify,
            )
        except ValueError:
            train_idx, val_idx = train_test_split(
                np.arange(X.shape[0]),
                test_size=validation_fraction,
                random_state=random_state,
            )
        X_train_pool, y_train_pool = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
    else:
        X_train_pool, y_train_pool = X, y
        X_val, y_val = np.asarray(X_validation), np.asarray(y_validation)

    if X_train_pool.shape[0] < 2 or X_val.shape[0] == 0:
        return pd.DataFrame()

    rows = []
    metric_label = "Weighted F1" if task == "classification" else "R2"
    for fraction in fractions:
        n_subset = int(np.ceil(float(fraction) * X_train_pool.shape[0]))
        n_subset = max(2, min(n_subset, X_train_pool.shape[0]))
        subset_idx = _deterministic_training_subset(y_train_pool, n_subset, task, random_state)
        if len(subset_idx) < 2:
            continue
        X_subset = X_train_pool[subset_idx]
        y_subset = y_train_pool[subset_idx]
        if task == "classification" and np.unique(y_subset).size < 2:
            continue
        try:
            scaler = StandardScaler()
            X_subset_scaled = scaler.fit_transform(X_subset)
            X_val_scaled = scaler.transform(X_val)
            model = model_factory()
            if hasattr(model, "batch_size"):
                try:
                    model.batch_size = max(2, min(int(model.batch_size), len(subset_idx)))
                except Exception:
                    pass
            with SuppressOutput():
                model.fit(X_subset_scaled, y_subset)
            train_pred = model.predict(X_subset_scaled)
            val_pred = model.predict(X_val_scaled)
            if task == "classification":
                train_score = f1_score(y_subset, train_pred, average="weighted", zero_division=0)
                val_score = f1_score(y_val, val_pred, average="weighted", zero_division=0)
            else:
                train_score = r2_score(y_subset, train_pred) if len(y_subset) > 1 else np.nan
                val_score = r2_score(y_val, val_pred) if len(y_val) > 1 else np.nan
            rows.append(
                {
                    "TrainingFraction": float(fraction),
                    "TrainingSamples": int(len(subset_idx)),
                    "TrainingScore": float(train_score),
                    "ValidationScore": float(val_score),
                    "Metric": metric_label,
                }
            )
        except Exception as exc:
            warnings.warn(f"Training-size diagnostic skipped at fraction {fraction:.2f}: {exc}", RuntimeWarning)
            continue

    return write_training_size_curve(rows, prefix, model_key, metric_label)


def write_training_size_curve(rows: Iterable[dict], prefix: str, model_key: str, metric_label: str) -> pd.DataFrame:
    columns = ["TrainingFraction", "TrainingSamples", "TrainingScore", "ValidationScore", "Metric"]
    df = pd.DataFrame(list(rows))
    if df.empty:
        return df
    for column in columns:
        if column not in df.columns:
            df[column] = np.nan
    df = df[columns]
    csv_path = f"{prefix}_{model_key}_training_size_curve.csv"
    png_path = f"{prefix}_{model_key}_training_size_curve.png"
    df.to_csv(csv_path, index=False)
    plot_training_size_curve(df, f"{model_key.replace('_', ' ').title()} Training Size Diagnostic", metric_label, png_path)
    return df


def plot_training_size_curve(df: pd.DataFrame, title: str, metric_label: str, output: str | os.PathLike[str]) -> None:
    if df.empty:
        return
    plot_df = df.copy()
    if {"TrainingFraction", "TrainingScore", "ValidationScore"}.issubset(plot_df.columns):
        x_column = "TrainingFraction"
        training_column = "TrainingScore"
        validation_column = "ValidationScore"
        x_label = "Training data used (%)"
        scale = 100.0
    elif {"Training_Size", "Training_Score_Loss", "Validation_Score_Loss"}.issubset(plot_df.columns):
        x_column = "Training_Size"
        training_column = "Training_Score_Loss"
        validation_column = "Validation_Score_Loss"
        x_label = "Training samples"
        scale = 1.0
    else:
        return
    for column in [x_column, training_column, validation_column]:
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    plot_df = plot_df.dropna(subset=[x_column, training_column, validation_column]).sort_values(x_column)
    if plot_df.empty:
        return
    configure_publication_plots()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    x_values = plot_df[x_column].to_numpy(dtype=float) * scale
    training = plot_df[training_column].to_numpy(dtype=float)
    validation = plot_df[validation_column].to_numpy(dtype=float)
    ax.plot(x_values, training, marker="o", color=PUBLICATION_COLORS[0], linewidth=1.8, label="Training")
    ax.plot(x_values, validation, marker="s", color=PUBLICATION_COLORS[1], linewidth=1.8, label="Validation")
    ax.fill_between(x_values, training, validation, color="#9CA3AF", alpha=0.18, label="Generalization gap")
    ax.set_xlabel(x_label)
    ax.set_ylabel(metric_label)
    ax.set_title(title)
    ax.legend(frameon=False, loc="best")
    style_publication_axis(ax)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _safe_stratify_labels(y: np.ndarray) -> np.ndarray | None:
    _, counts = np.unique(y, return_counts=True)
    return y if counts.size > 1 and np.all(counts >= 2) else None


def _deterministic_training_subset(y: np.ndarray, n_subset: int, task: str, random_state: int) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    all_idx = np.arange(len(y))
    if task != "classification":
        return np.sort(rng.choice(all_idx, size=min(n_subset, len(all_idx)), replace=False))

    selected = []
    for cls in np.sort(np.unique(y)):
        cls_idx = all_idx[y == cls]
        if cls_idx.size:
            selected.append(int(cls_idx[0]))
    remaining_slots = max(0, min(n_subset, len(all_idx)) - len(selected))
    if remaining_slots:
        remaining = np.setdiff1d(all_idx, np.asarray(selected, dtype=int), assume_unique=False)
        if remaining.size:
            selected.extend(rng.choice(remaining, size=min(remaining_slots, remaining.size), replace=False).astype(int).tolist())
    return np.sort(np.asarray(selected[: min(n_subset, len(all_idx))], dtype=int))


def add_common_cli_arguments(parser, *, require_input: bool = True, default_feature_selection: str = "no") -> None:
    try:
        from .execution_profile import PROFILE_CHOICES
    except ImportError:
        from execution_profile import PROFILE_CHOICES

    parser.add_argument("--input", required=require_input, help="Input CSV file or task input directory.")
    parser.add_argument("--prefix", default=None, help="Output prefix.")
    parser.add_argument("--outdir", default=".", help="Base output directory.")
    parser.add_argument("--feature_selection", default=default_feature_selection, choices=["yes", "no"], help="Use automatic protein feature selection.")
    parser.add_argument("--jobs", type=int, default=None, help="Parallel workers. Defaults to the selected execution profile.")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE, help="Random seed.")
    parser.add_argument("--profile", choices=PROFILE_CHOICES, default="balanced", help="Execution profile for runtime resource management.")
    parser.add_argument("--imputation", choices=["median", "knn"], default="median", help="Fold-local biomarker imputation method.")
    parser.add_argument("--covariates", nargs="*", default=None, metavar="COLUMN", help="Optional adjustment predictors that bypass biomarker selection.")
    parser.add_argument("--categorical-covariates", nargs="*", default=None, metavar="COLUMN", help="Covariates to encode categorically, including numeric-coded categories.")


class SuppressOutput:
    def __enter__(self):
        import sys

        self._stdout = sys.stdout
        self._stderr = sys.stderr
        self._null_out = open(os.devnull, "w")
        self._null_err = open(os.devnull, "w")
        sys.stdout = self._null_out
        sys.stderr = self._null_err
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        import sys

        sys.stdout = self._stdout
        sys.stderr = self._stderr
        self._null_out.close()
        self._null_err.close()


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min", verbose: bool = False):
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.best = None
        self.best_loss = None
        self.best_epoch = np.nan
        self.n_calls = 0
        self.counter = 0
        self.early_stop = False
        self._checkpoint = None

    def __call__(
        self,
        value: float,
        epoch: int | None = None,
        checkpoint_modules: Sequence[object] | None = None,
    ) -> bool:
        self.n_calls += 1
        current_epoch = self.n_calls if epoch is None else epoch
        if self.best is None:
            self.best = value
            self.best_loss = value
            self.best_epoch = current_epoch
            self._capture_checkpoint(checkpoint_modules)
            return False
        improved = value < self.best - self.min_delta if self.mode == "min" else value > self.best + self.min_delta
        if improved:
            self.best = value
            self.best_loss = value
            self.best_epoch = current_epoch
            self.counter = 0
            self._capture_checkpoint(checkpoint_modules)
            return False
        self.counter += 1
        if self.verbose:
            print(f"EarlyStopping counter: {self.counter} / {self.patience}")
        self.early_stop = self.counter >= self.patience
        return self.early_stop

    def _capture_checkpoint(self, modules: Sequence[object] | None) -> None:
        if modules is None:
            return
        self._checkpoint = tuple(deepcopy(module.state_dict()) for module in modules)

    def restore_checkpoint(self, *modules: object) -> bool:
        """Restore the validation-best module states captured during early stopping."""
        if self._checkpoint is None:
            return False
        if len(modules) != len(self._checkpoint):
            raise ValueError("Early-stopping checkpoint does not match the supplied modules.")
        for module, state in zip(modules, self._checkpoint):
            module.load_state_dict(state)
        return True


def normalize_feature_selection_flag(value: str | None) -> str:
    if value is None:
        return "none"
    value = str(value).strip().lower()
    if value in NO_SELECTION_VALUES:
        return "none"
    if value in YES_SELECTION_VALUES or value in LEGACY_SELECTION_METHODS:
        return "auto"
    if value in LEGACY_REDUCTION_METHODS:
        warnings.warn(
            f"'{value}' is Exploratory Analysis only and will not be used as a model input; using automatic protein-level feature selection.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "auto"
    raise ValueError("feature_selection must be yes/no. Reducer names are accepted only for older command invocations.")


def resolve_model_feature_selection(value: str | None, model_name: str, task: str = "classification") -> str:
    flag = normalize_feature_selection_flag(value)
    if flag == "none":
        return "none"

    key = str(model_name).strip().lower()
    if key in STANDALONE_MODELS or "vae" in key or key == "plsda" or key == "pls":
        return "none"
    if key in TREE_MODELS:
        return "boruta"
    return "elasticnet"


def is_component_feature(name: object) -> bool:
    text = str(name)
    return text.startswith(COMPONENT_PREFIXES)


def is_true_shap_source(value: object) -> bool:
    """Return true only for an explicit, recognized SHAP provenance label."""
    normalized = re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())
    return normalized in TRUE_SHAP_SOURCE_NAMES


def assert_protein_level_features(feature_names: Sequence[object], context: str = "SHAP") -> None:
    bad = [str(name) for name in feature_names if is_component_feature(name)]
    if bad:
        preview = ", ".join(bad[:5])
        raise ValueError(f"{context} requires protein-level features; component features detected: {preview}")


def finalize_shap_table(
    df: pd.DataFrame,
    model: str,
    task: str,
    feature_source: str = "original_proteins",
    fold: str | int = "Full",
    class_name: str = "Global",
) -> pd.DataFrame:
    if "Feature" not in df.columns:
        raise ValueError("SHAP table must contain a Feature column.")
    out = df.copy()
    out["Feature"] = out["Feature"].astype(str)
    assert_protein_level_features(out["Feature"].tolist(), f"{model} SHAP")
    if "MeanAbsSHAP" in out.columns:
        values = pd.to_numeric(out["MeanAbsSHAP"], errors="coerce").fillna(0.0).abs()
    elif "Global Mean" in out.columns:
        values = pd.to_numeric(out["Global Mean"], errors="coerce").fillna(0.0).abs()
    elif "Mean SHAP Value" in out.columns:
        values = pd.to_numeric(out["Mean SHAP Value"], errors="coerce").fillna(0.0).abs()
    else:
        numeric = out.drop(columns=["Feature"], errors="ignore").select_dtypes(include="number")
        values = numeric.abs().mean(axis=1) if not numeric.empty else pd.Series(np.zeros(len(out)), index=out.index)
    source = out["Importance_Source"].astype(str) if "Importance_Source" in out.columns else pd.Series(["shap"] * len(out), index=out.index)
    true_shap = source.map(is_true_shap_source)
    final = pd.DataFrame({
        "Feature": out["Feature"],
        "ImportanceValue": values.astype(float),
        "ImportanceType": np.where(true_shap, "SHAP", source),
        "IsTrueSHAP": true_shap,
        "MeanAbsSHAP": values.astype(float).where(true_shap),
        "Model": model,
        "Task": task,
        "FeatureSource": feature_source,
        "Fold": fold,
        "ClassName": class_name,
        "IsProteinLevel": True,
        "Importance_Source": source,
    })
    return final.sort_values("ImportanceValue", ascending=False).reset_index(drop=True)


def robust_minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    low = float(s.min())
    high = float(s.max())
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - low) / (high - low)


class BorutaProteinSelector:
    def __init__(self, task: str = "classification", random_state: int = DEFAULT_RANDOM_STATE, max_iter: int = 100, n_estimators: int = 500):
        self.task = task
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.support_ = None
        self.ranking_ = None
        self.feature_names_in_ = None
        self.selected_feature_names_ = None
        self.estimator_ = None
        self._boruta = None

    def get_params(self, deep: bool = True):
        return {
            "task": self.task,
            "random_state": self.random_state,
            "max_iter": self.max_iter,
            "n_estimators": self.n_estimators,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.feature_selection import SelectFromModel

        if y is None:
            raise ValueError("BorutaProteinSelector requires y for supervised feature selection.")
        X_arr = np.asarray(X)
        self.feature_names_in_ = list(getattr(X, "columns", [f"Feature_{i+1}" for i in range(X_arr.shape[1])]))
        if self.task == "regression":
            estimator = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=5,
                max_features="sqrt",
                min_samples_leaf=max(1, min(5, X_arr.shape[0] // 20)),
                random_state=self.random_state,
                n_jobs=-1,
            )
        else:
            estimator = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=5,
                max_features="sqrt",
                min_samples_leaf=max(1, min(5, X_arr.shape[0] // 20)),
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
            )
        self.estimator_ = estimator
        try:
            from boruta import BorutaPy

            boruta = BorutaPy(
                estimator,
                n_estimators="auto",
                max_iter=self.max_iter,
                random_state=self.random_state,
                verbose=0,
            )
            boruta.fit(X_arr, np.asarray(y))
            self._boruta = boruta
            support = np.asarray(boruta.support_, dtype=bool)
            ranking = np.asarray(boruta.ranking_, dtype=float)
        except Exception:
            estimator.fit(X_arr, np.asarray(y))
            selector = SelectFromModel(estimator, threshold="median", prefit=True)
            support = np.asarray(selector.get_support(), dtype=bool)
            if not support.any():
                top_k = max(1, min(50, X_arr.shape[1]))
                order = np.argsort(getattr(estimator, "feature_importances_", np.ones(X_arr.shape[1])))[::-1]
                support[order[:top_k]] = True
            ranks = np.argsort(np.argsort(-getattr(estimator, "feature_importances_", np.ones(X_arr.shape[1]))))
            ranking = ranks.astype(float) + 1.0

        # The fallback path already retains the strongest ranked features when an
        # estimator rejects every feature. Apply that established policy to a
        # completed BorutaPy run as well so downstream supervised estimators never
        # receive a zero-column matrix.
        if not support.any():
            top_k = max(1, min(50, X_arr.shape[1]))
            order = np.argsort(ranking)
            support[order[:top_k]] = True

        self.support_ = support
        self.ranking_ = ranking
        self.selected_feature_names_ = list(np.asarray(self.feature_names_in_, dtype=object)[support])
        return self

    def transform(self, X):
        if self.support_ is None:
            raise ValueError("BorutaProteinSelector is not fitted.")
        if isinstance(X, pd.DataFrame):
            selected = [str(name) for name in self.selected_feature_names_]
            column_lookup = {str(column): column for column in X.columns}
            if all(name in column_lookup for name in selected):
                output = X.loc[:, [column_lookup[name] for name in selected]].copy()
                output.columns = selected
                return output
        X_arr = np.asarray(X)
        return X_arr[:, self.support_]

    def get_support(self):
        if self.support_ is None:
            raise ValueError("BorutaProteinSelector is not fitted.")
        return self.support_


def make_elasticnet_logistic_regression(**kwargs):
    """Build equivalent elastic-net logistic regression across sklearn APIs."""
    from sklearn.linear_model import LogisticRegression

    parameters = dict(kwargs)
    parameters.pop("n_jobs", None)
    penalty_parameter = inspect.signature(LogisticRegression).parameters.get("penalty")
    if penalty_parameter is not None and penalty_parameter.default != "deprecated":
        parameters["penalty"] = "elasticnet"
    return LogisticRegression(**parameters)


def make_elasticnet_selector(task: str = "classification", C: float = 1.0, alpha: float = 0.01, l1_ratio: float = 0.5):
    from sklearn.feature_selection import SelectFromModel
    from sklearn.linear_model import ElasticNet

    if task == "regression":
        estimator = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=50000,
            tol=1e-4,
            random_state=DEFAULT_RANDOM_STATE,
            selection="random",
        )
    else:
        estimator = make_elasticnet_logistic_regression(
            solver="saga",
            C=C,
            l1_ratio=l1_ratio,
            max_iter=20000,
            tol=1e-4,
            class_weight="balanced",
            random_state=DEFAULT_RANDOM_STATE,
        )
    return SelectFromModel(estimator, threshold="median")


def selected_feature_names_from_pipeline(pipeline, original_features: Sequence[str]) -> list[str]:
    if not hasattr(pipeline, "named_steps"):
        return list(original_features)
    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor is not None and hasattr(preprocessor, "selected_biomarker_names_"):
        return list(map(str, preprocessor.selected_biomarker_names_))
    selector = pipeline.named_steps.get("feature_selection")
    if selector is None:
        return list(original_features)
    if hasattr(selector, "get_support"):
        support = np.asarray(selector.get_support(), dtype=bool)
    elif hasattr(selector, "selector") and hasattr(selector.selector, "get_support"):
        support = np.asarray(selector.selector.get_support(), dtype=bool)
    else:
        return list(original_features)
    if len(support) != len(original_features):
        return list(original_features)
    return list(np.asarray(original_features, dtype=object)[support])


def prediction_input_with_feature_names(values, feature_names: Sequence[str], estimator):
    """Retain fitted LightGBM feature names for prediction and explanation calls."""
    module_name = str(getattr(estimator.__class__, "__module__", ""))
    if not module_name.startswith("lightgbm"):
        return values
    fitted_names = getattr(estimator, "feature_name_", None)
    if fitted_names is None:
        fitted_names = getattr(estimator, "feature_names_in_", None)
    names = list(fitted_names) if fitted_names is not None else list(feature_names)
    if len(names) != np.asarray(values).shape[1]:
        raise ValueError(
            "LightGBM inference received a feature matrix whose width does not match "
            "the fitted feature-name sequence."
        )
    string_names = [str(name) for name in names]
    if isinstance(values, pd.DataFrame):
        column_lookup = {str(column): column for column in values.columns}
        if all(name in column_lookup for name in string_names):
            prepared = values.loc[:, [column_lookup[name] for name in string_names]].copy()
            prepared.columns = string_names
            return prepared
    index = values.index if isinstance(values, pd.DataFrame) else None
    return pd.DataFrame(
        np.asarray(values),
        columns=string_names,
        index=index,
    )


def estimator_importance(estimator, n_features: int) -> np.ndarray:
    values, _ = estimator_importance_with_source(estimator, n_features)
    return values


def _coefficient_importance(values, n_features: int) -> np.ndarray:
    array = np.abs(np.asarray(values, dtype=float))
    if array.ndim <= 1:
        return array.reshape(-1)
    if array.shape[-1] == n_features:
        axes = tuple(range(array.ndim - 1))
        return np.mean(array, axis=axes)
    if array.shape[0] == n_features:
        axes = tuple(range(1, array.ndim))
        return np.mean(array, axis=axes)
    return array.reshape(-1)


def estimator_importance_with_source(estimator, n_features: int) -> tuple[np.ndarray, str]:
    calibrated_estimators = getattr(estimator, "calibrated_classifiers_", None)
    if calibrated_estimators:
        component_values = []
        component_sources = []
        for calibrated in calibrated_estimators:
            base_estimator = getattr(calibrated, "estimator", None)
            if base_estimator is None:
                continue
            values, source = estimator_importance_with_source(base_estimator, n_features)
            component_values.append(values)
            component_sources.append(source)
        if component_values:
            return np.mean(np.vstack(component_values), axis=0), component_sources[0]
    if hasattr(estimator, "feature_importances_"):
        vals = np.asarray(estimator.feature_importances_, dtype=float)
        source = "NativeFeatureImportance"
    elif hasattr(estimator, "coef_"):
        vals = _coefficient_importance(estimator.coef_, n_features)
        source = "CoefficientMagnitude"
    elif hasattr(estimator, "model_") and hasattr(estimator.model_, "coef_"):
        vals = _coefficient_importance(estimator.model_.coef_, n_features)
        source = "CoefficientMagnitude"
    elif hasattr(estimator, "pls_") and hasattr(estimator.pls_, "coef_"):
        vals = _coefficient_importance(estimator.pls_.coef_, n_features)
        source = "CoefficientMagnitude"
    elif hasattr(estimator, "coefs_") and estimator.coefs_:
        vals = np.mean(np.abs(np.asarray(estimator.coefs_[0], dtype=float)), axis=1)
        source = "ModelSpecificImportance"
    elif hasattr(estimator, "theta_") and hasattr(estimator, "var_"):
        means = np.asarray(estimator.theta_, dtype=float)
        variances = np.asarray(estimator.var_, dtype=float)
        centered = means - np.mean(means, axis=0, keepdims=True)
        vals = np.sqrt(np.mean(centered**2, axis=0)) / np.sqrt(np.mean(variances, axis=0) + 1e-12)
        source = "ModelSpecificImportance"
    elif hasattr(estimator, "support_vectors_") and hasattr(estimator, "dual_coef_"):
        support = np.abs(np.asarray(estimator.support_vectors_, dtype=float))
        weights = np.sum(np.abs(np.asarray(estimator.dual_coef_, dtype=float)), axis=0)
        vals = np.average(support, axis=0, weights=weights) if np.sum(weights) > 0 else np.mean(support, axis=0)
        source = "ModelSpecificImportance"
    elif hasattr(estimator, "_predictors"):
        vals = np.zeros(n_features, dtype=float)
        for iteration in estimator._predictors:
            for predictor in iteration:
                nodes = getattr(predictor, "nodes", None)
                if nodes is None or "feature_idx" not in nodes.dtype.names:
                    continue
                split_nodes = nodes[nodes["is_leaf"] == 0] if "is_leaf" in nodes.dtype.names else nodes
                indices = np.asarray(split_nodes["feature_idx"], dtype=int)
                indices = indices[(indices >= 0) & (indices < n_features)]
                vals += np.bincount(indices, minlength=n_features)[:n_features]
        source = "ModelSpecificImportance"
    elif hasattr(estimator, "_fit_X") and hasattr(estimator, "_y"):
        fit_x = np.asarray(estimator._fit_X, dtype=float)
        fit_y = np.asarray(estimator._y)
        if fit_x.ndim == 2 and fit_x.shape[1] == n_features and np.unique(fit_y).size < max(10, len(fit_y) // 3):
            overall = np.mean(fit_x, axis=0)
            vals = np.zeros(n_features, dtype=float)
            for label in np.unique(fit_y):
                group = fit_x[fit_y == label]
                if len(group):
                    vals += len(group) * (np.mean(group, axis=0) - overall) ** 2
            vals /= max(1, len(fit_y))
        elif fit_x.ndim == 2 and fit_x.shape[1] == n_features:
            centered_x = fit_x - np.mean(fit_x, axis=0, keepdims=True)
            centered_y = fit_y.astype(float) - np.mean(fit_y.astype(float))
            vals = np.abs(centered_x.T @ centered_y) / max(1, len(fit_y) - 1)
        else:
            vals = np.ones(n_features, dtype=float)
        source = "ModelSpecificImportance"
    else:
        vals = _torch_input_importance(estimator, n_features)
        source = "ModelSpecificImportance" if vals is not None else "UniformFallback"
        if vals is None:
            vals = np.ones(n_features, dtype=float)
    vals = np.ravel(vals)
    if len(vals) != n_features:
        vals = np.ones(n_features, dtype=float)
        source = "UniformFallback"
    vals = np.nan_to_num(np.abs(vals), nan=0.0, posinf=0.0, neginf=0.0)
    if not np.any(vals > 0):
        vals = np.ones(n_features, dtype=float)
        source = "UniformFallback"
    return vals, source


def _torch_input_importance(estimator, n_features: int) -> np.ndarray | None:
    candidates = [estimator]
    for name in ("vae", "encoder", "model", "network"):
        value = getattr(estimator, name, None)
        if value is not None:
            candidates.append(value)
    for candidate in candidates:
        modules = getattr(candidate, "modules", None)
        if not callable(modules):
            continue
        try:
            for module in modules():
                weight = getattr(module, "weight", None)
                if weight is None:
                    continue
                array = weight.detach().cpu().numpy()
                if array.ndim == 2 and array.shape[1] == n_features:
                    return np.mean(np.abs(array), axis=0)
        except Exception:
            continue
    return None


def export_outer_fold_biomarkers(
    prefix: str,
    model_name: str,
    fold: int,
    estimator,
    original_features: Sequence[str],
    *,
    selection_method: str | None = None,
    shap_importance: Sequence[float] | None = None,
    importance_values: Sequence[float] | None = None,
    importance_type: str | None = None,
    top_n: int = 200,
) -> pd.DataFrame:
    feature_names = list(map(str, original_features))
    final_estimator = estimator
    preprocessor = None
    if hasattr(estimator, "steps") and estimator.steps:
        preprocessor = getattr(estimator, "named_steps", {}).get("preprocessor")
        feature_names = selected_feature_names_from_pipeline(estimator, feature_names)
        final_estimator = estimator.steps[-1][1]
    assert_protein_level_features(feature_names, "Outer-fold biomarker export")

    if shap_importance is not None:
        importance = np.asarray(shap_importance, dtype=float).reshape(-1)
        importance_type = "SHAP"
    elif importance_values is not None:
        importance = np.asarray(importance_values, dtype=float).reshape(-1)
        importance_type = importance_type or "ModelSpecificImportance"
    else:
        fitted_width = len(getattr(preprocessor, "feature_names_out_", feature_names))
        importance, importance_type = estimator_importance_with_source(final_estimator, fitted_width)
        if preprocessor is not None and hasattr(preprocessor, "feature_names_out_"):
            all_names = list(map(str, preprocessor.feature_names_out_))
            biomarker_positions = [all_names.index(name) for name in feature_names]
            importance = np.asarray(importance)[biomarker_positions]
    if len(importance) != len(feature_names):
        raise ValueError(
            f"Outer-fold biomarker export for {model_name} fold {fold} received "
            f"{len(importance)} importance values for {len(feature_names)} proteins."
        )

    if selection_method is None:
        selector = getattr(estimator, "named_steps", {}).get("feature_selection") if hasattr(estimator, "named_steps") else None
        if selector is None and preprocessor is not None:
            selector = getattr(preprocessor, "selector_", None)
        selector_name = type(selector).__name__.lower() if selector is not None else ""
        if "boruta" in selector_name:
            selection_method = "Boruta"
        elif "elastic" in selector_name or (
            selector is not None
            and type(getattr(selector, "estimator", None)).__name__ == "LogisticRegression"
            and 0.0 < float(getattr(getattr(selector, "estimator", None), "l1_ratio", 0.0) or 0.0) < 1.0
        ):
            selection_method = "ElasticNet"
        else:
            selection_method = "OriginalFeatures"

    importance = np.nan_to_num(np.abs(importance), nan=0.0, posinf=0.0, neginf=0.0)
    names = np.asarray(feature_names, dtype=object)
    order = np.lexsort((names.astype(str), -importance))[: min(int(top_n), len(names))]
    table = pd.DataFrame(
        {
            "Protein": names[order].astype(str),
            "Rank": np.arange(1, len(order) + 1, dtype=int),
            "Importance": importance[order],
            "ImportanceValue": importance[order],
            "ImportanceType": importance_type,
            "FeatureRole": "BIOMARKER",
            "FoldID": int(fold),
            "Model": str(model_name),
            "SelectionMethod": str(selection_method),
        }
    )
    table.to_csv(f"{prefix}_{model_name}_fold{int(fold)}_biomarkers.csv", index=False)
    return table


def find_prefixed_files(prefix: str, pattern: str) -> list[str]:
    return sorted(glob.glob(f"{prefix}{pattern}"))


def canonical_model_name_from_shap(path: str, prefix: str) -> str:
    name = Path(path).name
    name = re.sub(r"^" + re.escape(Path(prefix).name) + r"_", "", name)
    name = re.sub(r"_shap_values\.csv$", "", name)
    return name


def save_json(path: str | os.PathLike[str], payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
