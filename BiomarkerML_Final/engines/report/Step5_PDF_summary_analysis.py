from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from fpdf import FPDF
except Exception:
    FPDF = None

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from core.publication_utils import (
    DEFAULT_RANDOM_STATE,
    build_output_prefix,
    configure_publication_plots,
    format_effect_value,
    meaningful_display_mask,
    set_global_seed,
    task_output_dir,
)

configure_publication_plots()
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logging.getLogger("fontTools").setLevel(logging.WARNING)

FIGURE_CAPTIONS = {
    "dataset_distribution": "Protein-value distribution shown with a robust display transform; modeling inputs are unchanged.",
    "dataset_correlation": "Spearman correlation overview using a deterministic top-40 variance-ranked subset for readability; datasets with 40 or fewer numeric features show all features.",
    "missingness_landscape": "Descriptive feature missingness with the prespecified 50% training-fold retention threshold; modeling decisions remain fold-local.",
    "missingness_retention": "Outer-fold retention frequency under training-only missingness filtering; exactly 50% missingness is retained.",
    "pca": "Principal-component projection of the unsupervised protein feature space.",
    "kpca": "Kernel principal-component projection of non-linear structure in the protein feature space.",
    "tsne": "t-SNE projection of local sample neighborhoods in the protein feature space.",
    "umap": "UMAP projection of the unsupervised protein feature space.",
    "classification_benchmark": "Cross-model comparison using pooled subject-level out-of-fold AUC together with held-out Accuracy, Weighted F1, Sensitivity, and Specificity.",
    "regression_benchmark": "Cross-model comparison of RMSE, MAE, and R².",
    "roc_comparison": "Macro-average ROC compares classes equally, while micro-average ROC aggregates predictions over all classes; binary runs retain one ROC per model.",
    "prediction_comparison": "Observed-versus-predicted comparison across all successfully evaluated regression models.",
    "macro_auc_ranking": "Model ranking by pooled subject-level out-of-fold AUC; fold-mean AUC remains a separate variability summary.",
    "weighted_f1_ranking": "Model ranking by weighted F1; higher values indicate stronger class-balanced performance.",
    "accuracy_ranking": "Model ranking by accuracy; higher values indicate more correct predictions.",
    "rmse_ranking": "Model ranking by RMSE; lower values indicate smaller prediction error.",
    "mae_ranking": "Model ranking by MAE; lower values indicate smaller absolute prediction error.",
    "r2_ranking": "Model ranking by R²; higher values indicate greater explained variance.",
    "confusion_matrix": "Held-out classification counts underlying the reported performance metrics.",
    "roc_curve": "Held-out one-vs-rest class curves with macro- and micro-average discrimination when multiclass; binary runs show the single ROC.",
    "classification_metrics": "Six complementary held-out metrics summarize discrimination and class-detection performance for this model.",
    "classification_nested_cv": "Outer-fold classification performance from nested cross-validation.",
    "observed_predicted": "Held-out regression predictions compared with observed target values.",
    "residuals": "Residual structure for held-out regression predictions.",
    "regression_metrics": "Primary held-out regression metrics for this model.",
    "regression_nested_cv": "Outer-fold regression performance from nested cross-validation.",
    "learning_validation": "Training and validation behavior across the model's scientifically valid optimization or complexity progression.",
    "training_size": "Generalization behavior as the available training sample size increases.",
    "validation_curve": "Validation response across the recorded model diagnostic axis.",
    "optuna_history": "Optimization trajectory recorded during nested hyperparameter search.",
    "boosting_iteration": "Validation behavior across recorded boosting iterations.",
    "shap_beeswarm": "Protein-level contributions to model predictions estimated from SHAP values.",
    "per_class_shap": "Mean signed SHAP effects identify proteins that increase or decrease prediction toward {class_name}.",
    "top_marker_stability": "Reproducibility of this model's highest-ranking biomarkers across repeated validation.",
    "global_stability": "Combined biomarker reproducibility across all executed predictive models.",
    "jaccard": "Fold-set overlap across all executed predictive models.",
    "rank_correlation": "Fold-level biomarker rank agreement across all executed predictive models.",
    "biomarker_frequency": "Biomarker selection recurrence across all executed models and outer folds.",
    "consensus_ranking": "Biomarkers consistently supported by the predictive models included in consensus analysis.",
    "consensus_frequency": "Frequency and proportional support among models included in consensus analysis.",
    "recommended": "Aligned evidence for the same suggested biomarkers: model importance, reproducibility, cross-model support, and the existing final recommendation score. Biomarkers retain the existing recommendation order.{note}",
    "ppi_standard": "The network displays proteins with at least one observed interaction under the applied PPI criteria. Biomarkers without qualifying interactions are omitted from the network visualization but remain part of the biomarker recommendation set; accession labels appear only when no Gene Symbol is available.",
    "ppi_expanded": "Recommended biomarkers with biologically supported STRING interactors and co-expression evidence. The largest connected component is emphasized, expanded proteins remain visually secondary, and accession labels appear only when no Gene Symbol is available.",
    "ppi_hubs": "Network degree, defined as the number of observed interactions, for proteins ordered by the existing production hub ranking. Numeric labels report degree.",
}

FPDF_INTERPRETATION_SLOTS = {
    1: ((34, 150, 142, 125),),
    2: ((10, 150, 91, 125), (109, 150, 91, 125)),
    3: ((8, 150, 62, 125), (74, 150, 62, 125), (140, 150, 62, 125)),
}
MATPLOTLIB_INTERPRETATION_SLOTS = {
    1: ((0.18, 0.08, 0.64, 0.26),),
    2: ((0.07, 0.08, 0.41, 0.26), (0.52, 0.08, 0.41, 0.26)),
    3: ((0.04, 0.07, 0.29, 0.24), (0.355, 0.07, 0.29, 0.24), (0.67, 0.07, 0.29, 0.24)),
}


def report_figure_caption(kind: str, **context: str) -> str:
    return FIGURE_CAPTIONS[kind].format(**context).strip()


class PDFReport(FPDF if FPDF is not None else object):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "BiomarkerML Publication Report", 0, 1, "C")
        self.ln(4)

    def section(self, title: str):
        self.add_page()
        self.set_font("Arial", "B", 24)
        self.cell(0, 14, title, 0, 1, "L")
        self.ln(3)

    def title_line(self, title: str):
        self.set_font("Arial", "B", 14)
        self.multi_cell(0, 8, title)
        self.ln(2)

    def text(self, text: str):
        self.set_font("Arial", size=10)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def add_image(self, img_path: str, title: str, caption: str) -> bool:
        if not img_path or not os.path.exists(img_path):
            return False
        self.add_page()
        self.title_line(title)
        self.text(caption)
        try:
            self.image(img_path, x=10, y=self.get_y(), w=190)
            return True
        except RuntimeError:
            logging.warning("Failed to load image: %s", img_path)
            return False

    def add_table_page(self, title: str, df: pd.DataFrame, caption: str, *, max_rows: int = 12) -> None:
        if df.empty:
            return
        table = df.copy()
        columns = list(table.columns)
        usable_width = 190
        widths = _table_widths_from_data(table, usable_width)
        font_size = max(5.2, min(7.0, 42.0 / max(1, len(columns))))
        row_index = 0
        page_number = 1
        while row_index < len(table):
            self.add_page()
            self.title_line(title if page_number == 1 else f"{title} (continued)")
            self.text(caption)
            self._add_wrapped_table_row(columns, widths, font_size, bold=True)
            rows_on_page = 0
            while row_index < len(table) and rows_on_page < max_rows:
                values = [_format_table_value(table.iloc[row_index][column]) for column in columns]
                lines = _wrapped_table_lines(values, widths, font_size)
                row_height = max(5.2, 3.7 * max(len(cell_lines) for cell_lines in lines) + 1.0)
                if self.get_y() + row_height > 282 and rows_on_page:
                    break
                self._add_wrapped_table_row(values, widths, font_size, prepared_lines=lines)
                row_index += 1
                rows_on_page += 1
            page_number += 1

    def _add_wrapped_table_row(
        self,
        values: list[object],
        widths: list[float],
        font_size: float,
        *,
        bold: bool = False,
        prepared_lines: list[list[str]] | None = None,
    ) -> None:
        lines = prepared_lines or _wrapped_table_lines([str(value) for value in values], widths, font_size)
        line_height = 3.7
        row_height = max(5.2, line_height * max(len(cell_lines) for cell_lines in lines) + 1.0)
        x_start = 10.0
        y_start = self.get_y()
        self.set_font("Arial", "B" if bold else "", font_size)
        x = x_start
        for cell_lines, width in zip(lines, widths):
            self.rect(x, y_start, width, row_height)
            self.set_xy(x + 0.8, y_start + 0.6)
            self.multi_cell(max(1.0, width - 1.6), line_height, "\n".join(cell_lines), border=0)
            x += width
        self.set_xy(x_start, y_start + row_height)

    def add_key_value_page(self, title: str, rows: list[tuple[str, str]], caption: str) -> None:
        if not rows:
            return
        self.add_page()
        self.title_line(title)
        self.set_font("Arial", size=9.5)
        self.multi_cell(0, 5.5, caption)
        self.ln(3)
        label_width = 58
        value_width = 122
        for label, value in rows:
            start_y = self.get_y()
            self.set_font("Arial", "B", 9.5)
            self.cell(label_width, 6, str(label), border=0)
            self.set_xy(10 + label_width, start_y)
            self.set_font("Arial", size=9.5)
            self.multi_cell(value_width, 6, str(value))
            self.ln(1)

    def add_compact_table(self, df: pd.DataFrame, *, x: float = 10, y: float | None = None, width: float = 190, max_rows: int = 12) -> float:
        if df.empty:
            return self.get_y()
        if y is not None:
            self.set_y(y)
        self.set_x(x)
        table = df.head(max_rows).copy()
        columns = list(table.columns)
        widths = _table_widths(columns, width)
        row_h = 5.2
        self.set_font("Arial", "B", 6.6)
        for column, column_width in zip(columns, widths):
            self.cell(column_width, row_h, _pdf_cell(str(column), 20), border=1)
        self.ln()
        self.set_font("Arial", size=6.6)
        for _, row in table.iterrows():
            self.set_x(x)
            for column, column_width in zip(columns, widths):
                self.cell(column_width, row_h, _pdf_cell(_format_table_value(row[column]), 20), border=1)
            self.ln()
        return self.get_y()

    def add_figure_pair_page(self, section: str, figures: list[tuple[str | None, str, str]]) -> None:
        existing = [(path, title, caption) for path, title, caption in figures if path and os.path.exists(path)]
        if not existing:
            return
        for index in range(0, len(existing), 2):
            pair = existing[index : index + 2]
            self.add_page()
            self.title_line(section)
            positions = [(10, 34), (108, 34)] if len(pair) == 2 else [(34, 34)]
            for (path, title, caption), (x, y) in zip(pair, positions):
                self.set_xy(x, y)
                self.set_font("Arial", "B", 10)
                self.multi_cell(88, 5, title)
                self.set_x(x)
                self.set_font("Arial", size=7.5)
                self.multi_cell(88, 4, caption)
                try:
                    self.image(path, x=x, y=y + 20, w=88)
                except RuntimeError:
                    logging.warning("Failed to load image: %s", path)

    def add_full_width_figure_pages(self, section: str, figures: list[tuple[str | None, str, str]]) -> None:
        existing = [(path, title, caption) for path, title, caption in figures if path and os.path.exists(path)]
        for path, title, caption in existing:
            self.add_page()
            self.title_line(section)
            self.set_font("Arial", "B", 11)
            self.multi_cell(0, 5, title)
            self.set_font("Arial", size=8.5)
            self.multi_cell(0, 4.5, caption)
            try:
                self.image(path, x=12, y=self.get_y() + 2, w=186)
            except RuntimeError:
                logging.warning("Failed to load image: %s", path)

    def add_suggested_biomarker_page(
        self,
        figure: tuple[str | None, str, str] | None,
        explanation: str | None,
    ) -> None:
        if not figure or not figure[0] or not os.path.exists(figure[0]):
            return
        path, title, caption = figure
        self.add_page()
        self.title_line("Suggested Biomarkers")
        self.set_font("Arial", "B", 11)
        self.multi_cell(0, 5, title)
        self.set_font("Arial", size=8.2)
        self.multi_cell(0, 4.2, caption)
        image_y = self.get_y() + 2
        try:
            image_x, fitted_y, image_w, image_h = _fit_image_box(path, 10, image_y, 190, 174)
            self.image(path, x=image_x, y=fitted_y, w=image_w, h=image_h)
        except RuntimeError:
            logging.warning("Failed to load image: %s", path)
        if explanation and os.path.exists(explanation):
            try:
                image_x, fitted_y, image_w, image_h = _fit_image_box(explanation, 10, 224, 190, 58)
                self.image(explanation, x=image_x, y=fitted_y, w=image_w, h=image_h)
            except RuntimeError:
                logging.warning("Failed to load image: %s", explanation)

    def add_figure_grid_pages(self, section: str, figures: list[tuple[str | None, str, str]], *, per_page: int = 4) -> None:
        existing = [(path, title, caption) for path, title, caption in figures if path and os.path.exists(path)]
        if not existing:
            return
        for start in range(0, len(existing), per_page):
            chunk = existing[start : start + per_page]
            self.add_page()
            self.title_line(section)
            if len(chunk) == 1:
                positions = [(30, 40, 150)]
            elif len(chunk) == 3:
                positions = [(10, 34, 88), (108, 34, 88), (59, 158, 88)]
            else:
                positions = [(10, 34, 88), (108, 34, 88), (10, 158, 88), (108, 158, 88)]
            for (path, title, caption), (x, y, width) in zip(chunk, positions):
                self.set_xy(x, y)
                self.set_font("Arial", "B", 9.5)
                self.multi_cell(width, 4.6, title)
                self.set_x(x)
                self.set_font("Arial", size=7.1)
                self.multi_cell(width, 3.8, caption)
                try:
                    self.image(path, x=x, y=y + 18, w=width)
                except RuntimeError:
                    logging.warning("Failed to load image: %s", path)

    def add_model_interpretation_pages(self, section: str, figures: list[tuple[str | None, str, str]]) -> None:
        existing = _dedupe_figure_list(
            [(path, title, caption) for path, title, caption in figures if path and os.path.exists(path)]
        )
        if not existing:
            return
        beeswarm = [item for item in existing if item[1] == "Official SHAP Beeswarm"]
        secondary = [item for item in existing if item[1] != "Official SHAP Beeswarm"]
        if not beeswarm:
            self.add_figure_grid_pages(section, secondary, per_page=4)
            return

        class_items = [item for item in secondary if _is_per_class_shap_item(item)]
        support_items = [item for item in secondary if not _is_per_class_shap_item(item)]
        self.add_page()
        self.title_line(section)
        self._add_figure_panel(beeswarm[0], x=12, y=31, width=186, height=103)
        first_secondary = support_items[:1] if len(class_items) > 3 else secondary[:3]
        if first_secondary:
            positions = FPDF_INTERPRETATION_SLOTS[len(first_secondary)]
            for item, (x, y, width, height) in zip(first_secondary, positions):
                self._add_figure_panel(item, x=x, y=y, width=width, height=height)
        if len(class_items) > 3:
            self.add_figure_pair_page(f"{section}: Average SHAP per Class", class_items)
            remaining_support = support_items[1:]
            if remaining_support:
                self.add_figure_grid_pages(f"{section} (continued)", remaining_support, per_page=4)
        elif len(secondary) > 3:
            self.add_figure_grid_pages(f"{section} (continued)", secondary[3:], per_page=4)

    def _add_figure_panel(
        self,
        item: tuple[str, str, str],
        *,
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> None:
        path, title, caption = item
        self.set_xy(x, y)
        self.set_font("Arial", "B", 9.2)
        self.multi_cell(width, 4.5, title)
        caption_y = self.get_y()
        self.set_x(x)
        self.set_font("Arial", size=7.0)
        self.multi_cell(width, 3.6, caption)
        image_y = max(caption_y + 10, self.get_y() + 2)
        try:
            image_x, fitted_y, image_w, image_h = _fit_image_box(path, x, image_y, width, max(12, y + height - image_y))
            self.image(path, x=image_x, y=fitted_y, w=image_w, h=image_h)
        except RuntimeError:
            logging.warning("Failed to load image: %s", path)

    def add_benchmark_figure_page(
        self,
        section: str,
        primary: tuple[str | None, str, str] | None,
        rankings: list[tuple[str | None, str, str]],
    ) -> None:
        existing_rankings = [(path, title, caption) for path, title, caption in rankings if path and os.path.exists(path)]
        has_primary = bool(primary and primary[0] and os.path.exists(primary[0]))
        if not has_primary and not existing_rankings:
            return
        if has_primary and primary is not None:
            self.add_page()
            self.title_line(section)
            path, title, caption = primary
            self.set_xy(12, 34)
            self.set_font("Arial", "B", 11)
            self.multi_cell(184, 4.8, title)
            self.set_x(12)
            self.set_font("Arial", size=8.2)
            self.multi_cell(184, 4.2, caption)
            try:
                self.image(path, x=12, y=58, w=186)
            except RuntimeError:
                logging.warning("Failed to load image: %s", path)
        if existing_rankings:
            self.add_page()
            self.title_line(f"{section}: Performance Ranking")
            positions = [(10, 38, 88), (108, 38, 88), (59, 160, 88)]
            for (path, title, caption), (x, y, width) in zip(existing_rankings[:3], positions):
                self.set_xy(x, y)
                self.set_font("Arial", "B", 9.5)
                self.multi_cell(width, 4.6, title)
                self.set_x(x)
                self.set_font("Arial", size=7.1)
                self.multi_cell(width, 3.8, caption)
                try:
                    self.image(path, x=x, y=y + 18, w=width)
                except RuntimeError:
                    logging.warning("Failed to load image: %s", path)


def _find_suffix(suffix: str, prefix: str | None = None) -> str | None:
    matches = []
    for path in Path(".").rglob("*.png"):
        file = str(path)
        lower = path.name.lower()
        if not lower.endswith(suffix.lower()):
            continue
        if prefix and not lower.startswith(prefix.lower() + "_"):
            continue
        matches.append(file)
    return sorted(matches)[0] if matches else None


def _find_csv(filename: str, prefix: str | None = None) -> str | None:
    candidates = []
    names = [filename]
    if prefix:
        names.insert(0, f"{prefix}_{filename}")
    for name in names:
        candidates.extend(str(path) for path in Path(".").rglob(name))
    return sorted(set(candidates))[0] if candidates else None


def _find_json(filename: str, prefix: str | None = None) -> str | None:
    candidates = []
    names = [filename]
    if prefix:
        names.insert(0, f"{prefix}_{filename}")
    for name in names:
        candidates.extend(str(path) for path in Path(".").rglob(name))
    return sorted(set(candidates))[0] if candidates else None


def _find_png(filename: str, prefix: str | None = None) -> str | None:
    candidates = []
    names = [filename]
    if prefix:
        names.insert(0, f"{prefix}_{filename}")
    for name in names:
        candidates.extend(str(path) for path in Path(".").rglob(name))
    return sorted(set(candidates))[0] if candidates else None


def _find_all_suffix(suffix: str, prefix: str | None = None) -> list[str]:
    matches = []
    for path in Path(".").rglob("*.png"):
        lower = path.name.lower()
        if not lower.endswith(suffix.lower()):
            continue
        if prefix and not lower.startswith(prefix.lower() + "_"):
            continue
        matches.append(str(path))
    return sorted(matches)


def _read_csv(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _read_json(path: str | None) -> dict:
    if not path:
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _manifest_parameters(manifest: pd.DataFrame) -> dict[str, str]:
    if manifest.empty or "RecordType" not in manifest.columns:
        return {}
    parameters = manifest[manifest["RecordType"] == "Parameter"].copy()
    if not {"Name", "Value"}.issubset(parameters.columns):
        return {}
    return {str(row["Name"]): "" if pd.isna(row["Value"]) else str(row["Value"]) for _, row in parameters.iterrows()}


def _workflow_summary_table(prefix: str | None) -> pd.DataFrame:
    manifest = _read_csv(_find_csv("workflow_manifest.csv", prefix))
    if manifest.empty:
        return pd.DataFrame()
    params = _manifest_parameters(manifest)
    samples, features = _dataset_dimensions(params, prefix)
    models = [model for model in params.get("Models", "").split(";") if model]
    runtime = _workflow_runtime(manifest)
    feature_selection = "Automatic" if params.get("FeatureSelection", "").lower() == "yes" else "No"
    rows = [
        ("Task", _format_summary_value(params.get("Task", "").title())),
        ("Dataset", _format_dataset_name(params, prefix)),
        ("Samples", _format_summary_value(samples)),
        ("Features", _format_summary_value(features)),
        ("Models", _format_summary_value(len(models) if models else "")),
        ("Feature Selection", feature_selection),
        ("Nested Cross Validation", "5 Outer x 5 Inner"),
        ("Execution Profile", params.get("Profile", "")),
        ("Random Seed", _format_summary_value(params.get("Seed", str(DEFAULT_RANDOM_STATE)))),
        ("Runtime", runtime),
    ]
    return pd.DataFrame(rows, columns=["Item", "Value"])


def _format_dataset_name(params: dict[str, str], prefix: str | None) -> str:
    for key in ["Input", "WorkflowInput"]:
        value = params.get(key)
        if value:
            name = Path(value).name
            if name:
                return name
    if prefix:
        return str(prefix)
    return ""


def _dataset_dimensions(params: dict[str, str], prefix: str | None) -> tuple[str, str]:
    candidates: list[Path] = []
    for key in ["WorkflowInput", "Input"]:
        value = params.get(key)
        if value:
            candidates.append(Path(value))
    if prefix:
        candidates.extend(Path(".").rglob(f"{prefix}_input.csv"))
    label = params.get("Label", "Label")
    for path in candidates:
        try:
            if not path.exists():
                continue
            df = pd.read_csv(path)
            excluded = {label, "Label", "SampleID"}
            feature_count = len([column for column in df.columns if column not in excluded])
            return str(len(df)), str(feature_count)
        except Exception:
            continue
    return "", ""


def _format_summary_value(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    try:
        number = float(value)
        if number.is_integer():
            return f"{int(number)}"
    except Exception:
        pass
    return str(value)


def _workflow_runtime(manifest: pd.DataFrame) -> str:
    if manifest.empty or "RecordType" not in manifest.columns or "ElapsedTime" not in manifest.columns:
        return ""
    stages = manifest[manifest["RecordType"] == "Stage"]
    seconds = sum(_elapsed_seconds(value) for value in stages["ElapsedTime"].dropna())
    if seconds <= 0:
        return ""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _elapsed_seconds(value) -> int:
    parts = str(value).split(":")
    if len(parts) != 3:
        return 0
    try:
        hours, minutes, seconds = (int(float(part)) for part in parts)
        return hours * 3600 + minutes * 60 + seconds
    except Exception:
        return 0


def _table_widths(columns: list[str], total_width: float) -> list[float]:
    weights = [max(8, min(24, len(str(column)))) for column in columns]
    scale = total_width / sum(weights)
    return [weight * scale for weight in weights]


def _table_widths_from_data(df: pd.DataFrame, total_width: float) -> list[float]:
    weights = []
    for column in df.columns:
        lengths = [len(str(column))]
        lengths.extend(len(_format_table_value(value)) for value in df[column].head(50))
        representative = max(len(str(column)), float(np.percentile(lengths, 75))) if lengths else 8.0
        weights.append(max(7.0, min(32.0, representative)))
    scale = total_width / sum(weights)
    return [weight * scale for weight in weights]


def _wrapped_table_lines(values: list[object], widths: list[float], font_size: float) -> list[list[str]]:
    wrapped = []
    for value, width in zip(values, widths):
        chars = max(4, int(width / max(1.0, font_size * 0.45)))
        cleaned = re.sub(r";\s*", "; ", str(value).replace("\n", " ").strip())
        lines = textwrap.wrap(
            cleaned,
            width=chars,
            break_long_words=True,
            break_on_hyphens=False,
        )
        wrapped.append(lines or [""])
    return wrapped


def _pdf_cell(text: str, max_length: int) -> str:
    cleaned = text.replace("\n", " ").strip()
    return cleaned if len(cleaned) <= max_length else cleaned[: max_length - 3] + "..."


def _format_table_value(value) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"
    try:
        number = float(value)
        if pd.notna(number):
            if number.is_integer():
                return str(int(number))
            return format_effect_value(number, digits=3) if abs(number) < 100 else f"{number:.2f}"
    except Exception:
        pass
    return "" if pd.isna(value) else str(value)


def _fit_image_box(path: str, x: float, y: float, width: float, height: float) -> tuple[float, float, float, float]:
    image_width = image_height = 1.0
    try:
        from PIL import Image

        with Image.open(path) as image:
            image_width, image_height = map(float, image.size)
    except Exception:
        pass
    scale = min(width / image_width, height / image_height)
    fitted_width = image_width * scale
    fitted_height = image_height * scale
    return (
        x + (width - fitted_width) / 2,
        y + (height - fitted_height) / 2,
        fitted_width,
        fitted_height,
    )


def _exploratory_report_items(prefix: str | None) -> list[tuple[str | None, str, str]]:
    return [
        (_find_suffix("_pca_result.png", prefix), "PCA Visualization", report_figure_caption("pca")),
        (_find_suffix("_kpca_result.png", prefix), "Kernel PCA Visualization", report_figure_caption("kpca")),
        (_find_suffix("_tsne_result.png", prefix), "t-SNE Visualization", report_figure_caption("tsne")),
        (_find_suffix("_umap_result.png", prefix), "UMAP Visualization", report_figure_caption("umap")),
    ]


def _dataset_overview_items(prefix: str | None) -> list[tuple[str | None, str, str]]:
    return [
        (_find_suffix("_feature_distribution.png", prefix), "Feature Distribution", report_figure_caption("dataset_distribution")),
        (_find_suffix("_correlation_heatmap.png", prefix), "Correlation Heatmap", report_figure_caption("dataset_correlation")),
        (_find_suffix("_missingness_landscape.png", prefix), "Feature Missingness Landscape", report_figure_caption("missingness_landscape")),
        (_find_suffix("_missingness_retention_stability.png", prefix), "Fold-local Feature Retention", report_figure_caption("missingness_retention")),
    ]


def _model_key(path: str, prefix: str | None, suffix: str) -> str:
    stem = Path(path).stem
    if prefix and stem.lower().startswith(prefix.lower() + "_"):
        stem = stem[len(prefix) + 1 :]
    suffix_stem = suffix[:-4] if suffix.endswith(".png") else suffix
    for variant in [suffix_stem, suffix_stem.lstrip("_")]:
        if variant and stem.lower().endswith(variant.lower()):
            return stem[: -len(variant)].rstrip("_")
    return stem


def _normalize_model_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _pretty_model_name(value: str) -> str:
    names = {
        "RF": "Random Forest",
        "XGB": "XGBoost",
        "LGBM": "LightGBM",
        "LR": "Logistic Regression",
        "SVM": "SVM",
        "KNN": "KNN",
        "GNB": "GaussianNB",
        "NN": "Neural Network",
        "PLSDA": "PLSDA",
        "VAE": "VAE",
        "MLPVAE": "MLP-in-VAE",
        "RF_reg": "Random Forest Regression",
        "XGB_reg": "XGBoost Regression",
        "LGBM_reg": "LightGBM Regression",
        "SVM_reg": "SVR",
        "KNN_reg": "KNN Regression",
        "NN_reg": "Neural Network Regression",
        "PLS_reg": "PLS Regression",
        "VAE_reg": "VAE Regression",
        "MLPVAE_reg": "MLP-in-VAE Regression",
    }
    return names.get(str(value), str(value).replace("_", " ").title())


def _model_artifact_name(path: str, task: str) -> str | None:
    p = Path(path)
    task_dir = "classification" if task == "classification" else "regression"
    parts = [part.lower() for part in p.parts]
    if task_dir not in parts:
        return None
    parent = p.parent.name
    if parent.lower() in {task_dir, "overall_benchmark", "pdf_report", "workflow", "preprocess"}:
        return None
    return parent


def _ranking_order(prefix: str | None, task: str) -> list[str]:
    filename = "classification_model_ranking.csv" if task == "classification" else "regression_model_ranking.csv"
    ranking = _read_csv(_find_csv(filename, prefix))
    if ranking.empty or "Model" not in ranking.columns:
        return []
    if "Rank" in ranking.columns:
        ranking = ranking.sort_values("Rank")
    return [str(model) for model in ranking["Model"].dropna()]


def _is_regression_model_key(key: str) -> bool:
    return key.lower().endswith("_reg")


REPORT_MODEL_ALIASES = {
    "RF": {"RF", "random_forest"},
    "XGB": {"XGB", "xgboost"},
    "LGBM": {"LGBM", "lightgbm"},
    "LR": {"LR", "logistic_regression"},
    "SVM": {"SVM", "svm"},
    "KNN": {"KNN", "knn"},
    "GNB": {"GNB", "gaussiannb"},
    "NN": {"NN", "neural_network"},
    "PLSDA": {"PLSDA", "plsda"},
    "VAE": {"VAE", "vae"},
    "MLPVAE": {"MLPVAE", "vaemlp"},
    "RF_reg": {"RF_reg", "Random_Forest_reg"},
    "XGB_reg": {"XGB_reg", "XGBoost_reg"},
    "LGBM_reg": {"LGBM_reg", "LightGBM_reg"},
    "SVM_reg": {"SVM_reg"},
    "KNN_reg": {"KNN_reg"},
    "NN_reg": {"NN_reg", "Neural_Network_reg"},
    "PLS_reg": {"PLS_reg"},
    "VAE_reg": {"VAE_reg", "VAE_MLP_reg", "vae_reg"},
    "MLPVAE_reg": {"MLPVAE_reg", "MLP_in_VAE_reg", "vaemlp_reg"},
}


def _model_aliases_for_report(model: str) -> set[str]:
    aliases = {str(model)}
    normalized = _normalize_model_name(model)
    for key, values in REPORT_MODEL_ALIASES.items():
        normalized_values = {_normalize_model_name(value) for value in values | {key}}
        if normalized in normalized_values:
            aliases.update(values)
            aliases.add(key)
    return {_normalize_model_name(value) for value in aliases}


def _path_stem_without_prefix(path: str | Path, prefix: str | None) -> str:
    stem = Path(path).stem
    if prefix and stem.lower().startswith(prefix.lower() + "_"):
        stem = stem[len(prefix) + 1 :]
    return stem


def _model_key_for_suffix(path: str | Path, prefix: str | None, suffix: str) -> str:
    stem = _path_stem_without_prefix(path, prefix)
    suffix_stem = suffix[:-4] if suffix.endswith(".png") else suffix
    for variant in [suffix_stem, suffix_stem.lstrip("_")]:
        if variant and stem.lower().endswith(variant.lower()):
            return stem[: -len(variant)].rstrip("_")
    return stem


def _path_matches_model(path: str | Path, prefix: str | None, model: str, suffix: str) -> bool:
    return _normalize_model_name(_model_key_for_suffix(path, prefix, suffix)) in _model_aliases_for_report(model)


def _benchmark_paths(prefix: str | None, task: str) -> dict[str, str | None]:
    if task == "classification":
        return {
            "comparison_csv": _find_csv("classification_model_comparison.csv", prefix),
            "ranking_csv": _find_csv("classification_model_ranking.csv", prefix),
            "summary_json": _find_json("classification_best_model_summary.json", prefix),
            "performance_summary": _find_png("overall_performance_summary.png", prefix),
            "roc": _find_png("all_models_roc_comparison.png", prefix),
            "macro_auc": _find_png("macro_auc_ranking.png", prefix),
            "weighted_f1": _find_png("weighted_f1_ranking.png", prefix),
            "accuracy": _find_png("accuracy_ranking.png", prefix),
        }
    return {
        "comparison_csv": _find_csv("regression_model_comparison.csv", prefix),
        "ranking_csv": _find_csv("regression_model_ranking.csv", prefix),
        "summary_json": _find_json("regression_best_model_summary.json", prefix),
        "performance_summary": _find_png("overall_performance_summary.png", prefix),
        "scatter": _find_png("prediction_scatter_comparison.png", prefix),
        "rmse": _find_png("rmse_ranking.png", prefix),
        "mae": _find_png("mae_ranking.png", prefix),
        "r2": _find_png("r2_ranking.png", prefix),
    }


def _benchmark_figure_items(prefix: str | None, task: str) -> list[tuple[str | None, str, str]]:
    paths = _benchmark_paths(prefix, task)
    if not any(paths.get(key) for key in ["comparison_csv", "ranking_csv", "summary_json"]):
        return []
    if task == "classification":
        return [
            (paths["performance_summary"], "Overall Performance Summary", report_figure_caption("classification_benchmark")),
            (paths["roc"], "All-model ROC Comparison", report_figure_caption("roc_comparison")),
            (paths["macro_auc"], "Pooled OOF AUC Ranking", report_figure_caption("macro_auc_ranking")),
            (paths["weighted_f1"], "Weighted F1 Ranking", report_figure_caption("weighted_f1_ranking")),
            (paths["accuracy"], "Accuracy Ranking", report_figure_caption("accuracy_ranking")),
        ]
    return [
        (paths["performance_summary"], "Overall Performance Summary", report_figure_caption("regression_benchmark")),
        (paths["scatter"], "Prediction Scatter Comparison", report_figure_caption("prediction_comparison")),
        (paths["rmse"], "RMSE Ranking", report_figure_caption("rmse_ranking")),
        (paths["mae"], "MAE Ranking", report_figure_caption("mae_ranking")),
        (paths["r2"], "R2 Ranking", report_figure_caption("r2_ranking")),
    ]


def _benchmark_table_for_report(df: pd.DataFrame, task: str) -> pd.DataFrame:
    if df.empty:
        return df
    columns = (
        ["Model", "Accuracy", "Weighted F1", "Macro AUC", "Micro AUC", "Sensitivity", "Specificity"]
        if task == "classification"
        else ["Model", "RMSE", "MAE", "R2"]
    )
    out = df[[column for column in columns if column in df.columns]].copy()
    return out.rename(columns={"R2": "R²"})


def _benchmark_ranking_for_report(df: pd.DataFrame, task: str) -> pd.DataFrame:
    if df.empty:
        return df
    columns = (
        ["Rank", "Model", "Macro AUC", "Weighted F1", "Accuracy", "Micro AUC"]
        if task == "classification"
        else ["Rank", "Model", "RMSE", "MAE", "R2"]
    )
    out = df[[column for column in columns if column in df.columns]].copy()
    return out.rename(columns={"R2": "R²"})


def _structured_model_pages(prefix: str | None, task: str) -> dict[str, dict[str, list[tuple[str, str, str]]]]:
    if task == "classification":
        specs = {
            "page1": [
                ("_confusion_matrix.png", "Confusion Matrix", report_figure_caption("confusion_matrix")),
                ("_roc_curve.png", "ROC Curve", report_figure_caption("roc_curve")),
            ],
            "page2": [
                ("_publication_metrics.png", "Evaluation Metrics", report_figure_caption("classification_metrics")),
                ("_nested_cv_f1_auc.png", "Nested Cross Validation", report_figure_caption("classification_nested_cv")),
            ],
            "diagnostics": [
                ("_learning_curve.png", "Learning Curve", report_figure_caption("learning_validation")),
                ("_training_size_curve.png", "Training Size Diagnostic", report_figure_caption("training_size")),
            ],
        }
    else:
        specs = {
            "page1": [
                ("_overall_prediction.png", "Observed vs Predicted", report_figure_caption("observed_predicted")),
                ("_predictions.png", "Observed vs Predicted", report_figure_caption("observed_predicted")),
                ("_overall_residuals.png", "Residual Plot", report_figure_caption("residuals")),
                ("_residuals.png", "Residual Plot", report_figure_caption("residuals")),
            ],
            "page2": [
                ("_metrics.png", "Evaluation Metrics", report_figure_caption("regression_metrics")),
                ("_average_metrics.png", "Evaluation Metrics", report_figure_caption("regression_metrics")),
                ("_metrics_line_plot.png", "Nested Cross Validation", report_figure_caption("regression_nested_cv")),
                ("_metrics_over_folds.png", "Nested Cross Validation", report_figure_caption("regression_nested_cv")),
            ],
            "diagnostics": [
                ("_learning_curve.png", "Learning Curve", report_figure_caption("learning_validation")),
                ("_training_size_curve.png", "Training Size Diagnostic", report_figure_caption("training_size")),
                ("_validation_curve.png", "Validation Curve", report_figure_caption("validation_curve")),
                ("_optuna_optimization_history.png", "Optuna Optimization History", report_figure_caption("optuna_history")),
                ("_boosting_iteration.png", "Boosting Iteration Diagnostic", report_figure_caption("boosting_iteration")),
            ],
        }
    pages: dict[str, dict[str, list[tuple[str, str, str]]]] = {}
    for page, page_specs in specs.items():
        for suffix, title, caption in page_specs:
            for path in _find_all_suffix(suffix, prefix):
                model_name = _model_artifact_name(path, task)
                if model_name is None:
                    continue
                key = _model_key(path, prefix, suffix)
                if not key:
                    continue
                if task == "classification" and _is_regression_model_key(key):
                    continue
                if task == "regression" and (not _is_regression_model_key(key) or "_fold" in key.lower()):
                    continue
                artifact_title = _learning_progression_report_title(path, title)
                pages.setdefault(model_name, {"page1": [], "page2": [], "diagnostics": []})
                if path not in [item[0] for item in pages[model_name][page]]:
                    pages[model_name][page].append((path, artifact_title, caption))
    order = {_normalize_model_name(model): index for index, model in enumerate(_ranking_order(prefix, task))}
    return dict(
        sorted(
            pages.items(),
            key=lambda item: (order.get(_normalize_model_name(item[0]), 10_000), item[0].lower()),
        )
    )


def _model_interpretability_items(prefix: str | None, model: str, task: str) -> list[tuple[str, str, str]]:
    items: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    specs = [
        (
            "_shap_summary_beeswarm.png",
            "Official SHAP Beeswarm",
            report_figure_caption("shap_beeswarm"),
        ),
    ]
    for suffix, title, caption in specs:
        for path in _find_all_suffix(suffix, prefix):
            if not _path_matches_model(path, prefix, model, suffix):
                continue
            key = str(Path(path))
            if key in seen:
                continue
            seen.add(key)
            items.append((path, title, caption))
    if task == "classification":
        items.extend(_model_per_class_shap_items(prefix, model, seen))
    items.extend(_model_stability_items(prefix, model, seen))
    return items


def _model_per_class_shap_items(prefix: str | None, model: str, seen: set[str]) -> list[tuple[str, str, str]]:
    items: list[tuple[str, str, str]] = []
    aliases = _model_aliases_for_report(model)
    for path in sorted(Path(".").rglob("*.png")):
        lower = path.name.lower()
        if not lower.endswith(".png") or "_per_class_shap_" not in lower:
            continue
        if prefix and not lower.startswith(prefix.lower() + "_"):
            continue
        stem = _path_stem_without_prefix(path, prefix)
        if "_per_class_shap_" not in stem:
            continue
        model_key, class_name = stem.split("_per_class_shap_", 1)
        if _normalize_model_name(model_key) not in aliases:
            continue
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        title = f"Average SHAP per Class: {class_name.replace('_', ' ')}"
        display_class = class_name.replace("_", " ")
        caption = report_figure_caption("per_class_shap", class_name=display_class)
        interpretation = _per_class_shap_interpretation(path, display_class)
        if interpretation:
            caption = f"{caption} {interpretation}"
        items.append((str(path), title, caption))
    return items


def _per_class_shap_interpretation(path: str | Path, class_name: str) -> str:
    """Summarize existing signed SHAP effects for a concise report caption."""
    figure = Path(path)
    marker = "_per_class_shap_"
    figure_prefix = figure.name.split(marker, 1)[0] if marker in figure.name else ""
    matching_table = figure.parent / f"{figure_prefix}_per_class_shap.csv" if figure_prefix else None
    candidates = [matching_table] if matching_table is not None and matching_table.exists() else []
    if not candidates:
        candidates = sorted(figure.parent.glob("*_per_class_shap.csv"), key=lambda item: item.name.lower())
    if not candidates:
        return ""
    table = _read_csv(str(candidates[0]))
    if table.empty or not {"Feature", "ClassName", "MeanSHAP"}.issubset(table.columns):
        return ""
    class_key = _normalize_model_name(class_name)
    rows = table[table["ClassName"].astype(str).map(_normalize_model_name) == class_key].copy()
    rows["MeanSHAP"] = pd.to_numeric(rows["MeanSHAP"], errors="coerce")
    rows = rows.loc[meaningful_display_mask(rows["MeanSHAP"])].dropna(subset=["MeanSHAP"])
    if rows.empty:
        return ""
    display_values = rows["Feature"].astype(str)
    if "DisplayFeature" in rows.columns:
        mapped = rows["DisplayFeature"].fillna("").astype(str).str.strip()
        valid = ~mapped.str.lower().isin({"", "nan", "none", "null"})
        display_values = display_values.where(~valid, mapped)
    rows["_DisplayFeature"] = display_values
    positive = rows[rows["MeanSHAP"] > 0].nlargest(2, "MeanSHAP")["_DisplayFeature"].tolist()
    negative = rows[rows["MeanSHAP"] < 0].nsmallest(2, "MeanSHAP")["_DisplayFeature"].tolist()
    statements = []
    if positive:
        statements.append(f"{', '.join(positive)} most strongly increase prediction toward {class_name}")
    if negative:
        statements.append(f"{', '.join(negative)} most strongly decrease it")
    return "; ".join(statements) + "." if statements else ""


def _model_stability_items(prefix: str | None, model: str, seen: set[str]) -> list[tuple[str, str, str]]:
    items: list[tuple[str, str, str]] = []
    suffix = "_top_marker_stability.png"
    matches = [
        path
        for path in _find_all_suffix(suffix, prefix)
        if _path_matches_model(path, prefix, model, suffix)
    ]
    matches.sort(
        key=lambda path: (
            0 if "Final" in Path(path).parts else 1,
            str(path).lower(),
        )
    )
    for path in matches[:1]:
        key = str(Path(path))
        if key in seen:
            continue
        seen.add(key)
        items.append(
            (
                path,
                "Top Marker Stability",
                report_figure_caption("top_marker_stability"),
            )
        )
    return items


def _is_per_class_shap_item(item: tuple[str, str, str]) -> bool:
    return item[1].startswith("Average SHAP per Class")


def _ppi_metadata_tables(prefix: str | None) -> list[tuple[str, pd.DataFrame, str]]:
    specifications = [
        (
            "ppi_network_nodes.csv",
            "Standard PPI Node Metadata",
            ["Protein", "GeneSymbol", "ImportanceValue", "ImportanceType", "SHAP", "ConsensusScore", "StabilityScore", "SupportFrequency", "Degree", "Betweenness", "Closeness", "Community", "HubRank"],
            "Model-attribution provenance and network centrality for nodes in the Standard PPI; SHAP is populated only for genuine SHAP values.",
        ),
        (
            "ppi_network_edges.csv",
            "Standard PPI Edge Metadata",
            ["Source", "Target", "EdgeType", "STRINGScore", "CoexpressionScore", "Confidence"],
            "Interaction type and confidence for edges connecting recommended biomarkers.",
        ),
        (
            "ppi_expanded_network_nodes.csv",
            "Expanded PPI Node Evidence",
            [
                "Protein", "GeneSymbol", "IsSeedBiomarker", "IsExpandedInteractor", "ConnectedSeedProtein",
                "STRINGConfidence", "CorrelationCoefficient", "CorrelationMethod", "ImportanceValue", "ImportanceType", "SHAP", "ConsensusScore",
                "StabilityScore", "SupportFrequency",
            ],
            "Seed status, expansion evidence, and provenance-aware biomarker attribution in the Expanded PPI; SHAP is populated only for genuine SHAP values.",
        ),
        (
            "ppi_expanded_network_nodes.csv",
            "Expanded PPI Node Centrality",
            ["Protein", "GeneSymbol", "Degree", "Betweenness", "Closeness", "Community", "HubRank"],
            "Network centrality and community assignment for every seed and expanded interactor.",
        ),
        (
            "ppi_expanded_network_edges.csv",
            "Expanded PPI Edge Metadata",
            [
                "Source", "Target", "EdgeType", "STRINGScore", "CoexpressionScore", "Confidence",
                "IsExpandedEdge", "ConnectedSeedProtein", "STRINGConfidence", "CorrelationCoefficient", "CorrelationMethod",
            ],
            "STRING confidence and expression-correlation support for Standard and expanded interactions.",
        ),
    ]
    tables = []
    for filename, title, columns, caption in specifications:
        table = _read_csv(_find_csv(filename, prefix))
        available = [column for column in columns if column in table.columns]
        if available:
            tables.append(
                (
                    title,
                    table[available].rename(
                        columns={
                            "IsSeedBiomarker": "Seed",
                            "IsExpandedInteractor": "Expanded",
                            "ConnectedSeedProtein": "SeedProtein",
                            "STRINGConfidence": "STRING",
                            "CorrelationCoefficient": "Correlation",
                            "CorrelationMethod": "Method",
                            "ConsensusScore": "Consensus",
                            "StabilityScore": "Stability",
                            "SupportFrequency": "Support",
                        }
                    ),
                    caption,
                )
            )
    return tables


def _build_report_with_matplotlib(prefix: str | None, output: str) -> None:
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(output) as pdf:
        _matplotlib_cover_page(pdf, prefix)
        _matplotlib_table_page(pdf, "Workflow Summary", _workflow_summary_table(prefix), "Scientifically relevant workflow configuration for this BiomarkerML run.")
        _matplotlib_figure_pages(pdf, "Dataset Overview", _dataset_overview_items(prefix), per_page=1)
        _matplotlib_figure_pages(pdf, "Exploratory Analysis", _exploratory_report_items(prefix), per_page=2)
        _matplotlib_benchmark_pages(pdf, prefix, "classification")
        _matplotlib_benchmark_pages(pdf, prefix, "regression")
        _matplotlib_model_pages(pdf, prefix, "classification")
        _matplotlib_model_pages(pdf, prefix, "regression")
        _matplotlib_figure_pages(pdf, "Biomarker Reproducibility Across Models", _global_stability_items(prefix), per_page=2)
        candidates = _consensus_candidate_table(prefix)
        if _consensus_available(prefix):
            if not candidates.empty:
                _matplotlib_table_page(
                    pdf,
                    "Consensus Biomarkers: Contributing Models",
                    candidates,
                    _consensus_rule_text(candidates),
                    max_rows=16,
                )
            _matplotlib_figure_pages(pdf, "Consensus Biomarkers", _consensus_items(prefix), per_page=2)
        _matplotlib_suggested_biomarker_page(
            pdf,
            _recommended_items(prefix),
            _recommended_explanation_item(),
        )
        _matplotlib_figure_pages(pdf, "Biological Network Interpretation", _ppi_items(prefix), per_page=1)
        stages = _manifest_stages_for_report(prefix)
        if not stages.empty:
            columns = [column for column in ["Name", "Status", "StartTime", "EndTime", "ElapsedTime"] if column in stages.columns]
            _matplotlib_table_page(pdf, "Workflow Manifest", stages[columns], "Completed workflow stage status and timing.", max_rows=24)


def _save_pdf_page(pdf, fig) -> None:
    import matplotlib as mpl

    with mpl.rc_context({"savefig.bbox": None}):
        pdf.savefig(fig)


def _matplotlib_cover_page(pdf, prefix: str | None) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    ax.text(0.5, 0.75, "BiomarkerML Publication Report", ha="center", fontsize=22, fontweight="bold")
    ax.text(0.5, 0.68, "Publication-ready model benchmarking, protein-level interpretation, biomarker reliability, and biological network analysis.", ha="center", fontsize=10, wrap=True)
    if prefix:
        ax.text(0.5, 0.62, f"Run: {prefix}", ha="center", fontsize=11, fontweight="bold")
    _save_pdf_page(pdf, fig)
    plt.close(fig)


def _matplotlib_table_page(pdf, title: str, df: pd.DataFrame, caption: str, *, max_rows: int = 12, note: str | None = None) -> None:
    if df.empty:
        return
    import matplotlib.pyplot as plt

    formatted = df.astype(object)
    formatted = formatted.map(_format_table_value) if hasattr(formatted, "map") else formatted.applymap(_format_table_value)
    wide = len(formatted.columns) > 6 or any(formatted[column].astype(str).str.len().max() > 45 for column in formatted.columns)
    page_size = (11.69, 8.27) if wide else (8.27, 11.69)
    widths = _table_widths_from_data(formatted, 1.0)
    font_size = max(5.2, min(7.2, 44.0 / max(1, len(formatted.columns))))
    usable_width_points = page_size[0] * 72.0 * 0.92
    width_points = [width * usable_width_points for width in widths]
    line_budget = 28 if wide else 42
    pages: list[list[int]] = []
    current_page: list[int] = []
    current_lines = 0
    for row_index, (_, row) in enumerate(formatted.iterrows()):
        line_count = max(len(lines) for lines in _wrapped_table_lines(row.tolist(), width_points, font_size))
        if current_page and (len(current_page) >= max_rows or current_lines + line_count > line_budget):
            pages.append(current_page)
            current_page = []
            current_lines = 0
        current_page.append(row_index)
        current_lines += line_count
    if current_page:
        pages.append(current_page)

    for page_index, row_indices in enumerate(pages):
        table = formatted.iloc[row_indices].copy()
        fig, ax = plt.subplots(figsize=page_size)
        ax.axis("off")
        page_title = title if page_index == 0 else f"{title} (continued)"
        if note:
            table_y = 0.06
            table_h = 0.66
        else:
            table_y = 0.08
            table_h = 0.78

        wrapped_rows = []
        row_line_counts = []
        for _, row in table.iterrows():
            lines = _wrapped_table_lines(row.tolist(), width_points, font_size)
            wrapped_rows.append(["\n".join(cell_lines) for cell_lines in lines])
            row_line_counts.append(max(len(cell_lines) for cell_lines in lines))
        header_lines = _wrapped_table_lines(list(table.columns), width_points, font_size)
        column_labels = ["\n".join(lines) for lines in header_lines]
        rendered = ax.table(
            cellText=wrapped_rows,
            colLabels=column_labels,
            colWidths=widths,
            loc="center",
            cellLoc="center",
            colLoc="center",
            bbox=[0.04, table_y, 0.92, table_h],
        )
        rendered.auto_set_font_size(False)
        rendered.set_fontsize(font_size)
        height_weights = [max(len(lines) for lines in header_lines)] + row_line_counts
        height_total = sum(max(1, value) for value in height_weights)
        for row_index, line_count in enumerate(height_weights):
            row_height = table_h * max(1, line_count) / height_total
            for column_index in range(len(table.columns)):
                cell = rendered[(row_index, column_index)]
                cell.set_height(row_height)
                cell.set_text_props(wrap=True, fontweight="bold" if row_index == 0 else "normal")
        fig.text(0.05, 0.96, page_title, va="top", fontsize=16, fontweight="bold")
        fig.text(0.05, 0.91, caption, va="top", fontsize=9, wrap=True)
        if note:
            fig.text(0.05, 0.86, note, va="top", fontsize=8.5, wrap=True)
        _save_pdf_page(pdf, fig)
        plt.close(fig)


def _matplotlib_figure_pages(pdf, section: str, figures: list[tuple[str | None, str, str]], *, per_page: int = 4) -> None:
    existing = [(path, title, caption) for path, title, caption in figures if path and os.path.exists(path)]
    if not existing:
        return
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    for start in range(0, len(existing), per_page):
        chunk = existing[start : start + per_page]
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.05, 0.965, section, va="top", fontsize=16, fontweight="bold")
        for (path, title, caption), (x, y, w, h) in zip(chunk, _matplotlib_grid_positions(len(chunk))):
            fig.text(x, y + h + 0.055, _wrap_figure_text(title, w, title=True), va="top", fontsize=10.5, fontweight="bold")
            fig.text(x, y + h + 0.025, _wrap_figure_text(caption, w), va="top", fontsize=7.8)
            ax = fig.add_axes([x, y, w, h])
            try:
                ax.imshow(mpimg.imread(path))
            except Exception:
                ax.text(0.5, 0.5, f"Failed to load {path}", ha="center", va="center", fontsize=8)
            ax.axis("off")
        _save_pdf_page(pdf, fig)
        plt.close(fig)


def _matplotlib_suggested_biomarker_page(
    pdf,
    figures: list[tuple[str | None, str, str]],
    explanation: str | None,
) -> None:
    existing = [(path, title, caption) for path, title, caption in figures if path and os.path.exists(path)]
    if not existing:
        return
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    path, title, caption = existing[0]
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.05, 0.965, "Suggested Biomarkers", va="top", fontsize=16, fontweight="bold")
    fig.text(0.05, 0.925, title, va="top", fontsize=10.5, fontweight="bold")
    fig.text(0.05, 0.895, _wrap_figure_text(caption, 0.9), va="top", fontsize=7.8)
    main_ax = fig.add_axes([0.05, 0.28, 0.90, 0.57])
    main_ax.imshow(mpimg.imread(path))
    main_ax.axis("off")
    if explanation and os.path.exists(explanation):
        explanation_ax = fig.add_axes([0.05, 0.055, 0.90, 0.20])
        explanation_ax.imshow(mpimg.imread(explanation))
        explanation_ax.axis("off")
    _save_pdf_page(pdf, fig)
    plt.close(fig)


def _matplotlib_interpretation_pages(pdf, section: str, figures: list[tuple[str | None, str, str]]) -> None:
    existing = _dedupe_figure_list(
        [(path, title, caption) for path, title, caption in figures if path and os.path.exists(path)]
    )
    if not existing:
        return
    beeswarm = [item for item in existing if item[1] == "Official SHAP Beeswarm"]
    secondary = [item for item in existing if item[1] != "Official SHAP Beeswarm"]
    if not beeswarm:
        _matplotlib_figure_pages(pdf, section, secondary, per_page=4)
        return

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.05, 0.965, section, va="top", fontsize=16, fontweight="bold")
    _matplotlib_add_panel(fig, beeswarm[0], (0.08, 0.53, 0.84, 0.34), title_y=0.925)
    class_items = [item for item in secondary if _is_per_class_shap_item(item)]
    support_items = [item for item in secondary if not _is_per_class_shap_item(item)]
    first_secondary = support_items[:1] if len(class_items) > 3 else secondary[:3]
    if first_secondary:
        positions = MATPLOTLIB_INTERPRETATION_SLOTS[len(first_secondary)]
        for item, position in zip(first_secondary, positions):
            _matplotlib_add_panel(fig, item, position, title_y=position[1] + position[3] + 0.06)
    _save_pdf_page(pdf, fig)
    plt.close(fig)
    if len(class_items) > 3:
        _matplotlib_figure_pages(pdf, f"{section}: Average SHAP per Class", class_items, per_page=2)
        remaining_support = support_items[1:]
        if remaining_support:
            _matplotlib_figure_pages(pdf, f"{section} (continued)", remaining_support, per_page=4)
    elif len(secondary) > 3:
        _matplotlib_figure_pages(pdf, f"{section} (continued)", secondary[3:], per_page=4)


def _matplotlib_add_panel(fig, item: tuple[str, str, str], position: tuple[float, float, float, float], *, title_y: float) -> None:
    import matplotlib.image as mpimg

    path, title, caption = item
    x, y, width, height = position
    fig.text(x, title_y, _wrap_figure_text(title, width, title=True), va="top", fontsize=9.5, fontweight="bold")
    fig.text(x, title_y - 0.025, _wrap_figure_text(caption, width), va="top", fontsize=7.0)
    ax = fig.add_axes([x, y, width, height])
    try:
        ax.imshow(mpimg.imread(path))
    except Exception:
        ax.text(0.5, 0.5, "Figure could not be rendered", ha="center", va="center", fontsize=8)
    ax.axis("off")


def _wrap_figure_text(value: str, panel_width: float, *, title: bool = False) -> str:
    chars = max(24, int(panel_width * (88 if title else 115)))
    return textwrap.fill(str(value), width=chars, break_long_words=False, break_on_hyphens=False)


def _matplotlib_grid_positions(count: int) -> list[tuple[float, float, float, float]]:
    if count == 1:
        return [(0.08, 0.13, 0.84, 0.66)]
    if count == 2:
        return [(0.08, 0.55, 0.84, 0.28), (0.08, 0.13, 0.84, 0.28)]
    if count == 3:
        return [(0.07, 0.55, 0.40, 0.28), (0.53, 0.55, 0.40, 0.28), (0.30, 0.13, 0.40, 0.28)]
    return [(0.07, 0.55, 0.40, 0.28), (0.53, 0.55, 0.40, 0.28), (0.07, 0.13, 0.40, 0.28), (0.53, 0.13, 0.40, 0.28)]


def _matplotlib_benchmark_pages(pdf, prefix: str | None, task: str) -> None:
    paths = _benchmark_paths(prefix, task)
    comparison = _benchmark_table_for_report(_read_csv(paths["comparison_csv"]), task)
    summary = _read_json(paths["summary_json"])
    figures = _benchmark_figure_items(prefix, task)
    if comparison.empty and not summary and not any(path and os.path.exists(path) for path, _, _ in figures):
        return
    title = "Classification Benchmark Overview" if task == "classification" else "Regression Benchmark Overview"
    _matplotlib_table_page(
        pdf,
        title,
        comparison,
        "Benchmark outputs were generated immediately after prediction and are presented without recomputing benchmark statistics.",
        max_rows=12,
        note=_benchmark_summary_text(summary) if summary else None,
    )
    if figures:
        if figures[:1]:
            _matplotlib_figure_pages(pdf, f"{title}: Overall Performance Summary", figures[:1], per_page=1)
        if len(figures) > 1:
            _matplotlib_figure_pages(pdf, f"{title}: ROC Comparison", figures[1:2], per_page=1)
        if len(figures) > 2:
            _matplotlib_benchmark_ranking_page(pdf, f"{title}: Performance Ranking", figures[2:])


def _matplotlib_benchmark_ranking_page(pdf, section: str, figures: list[tuple[str | None, str, str]]) -> None:
    existing = [(path, title, caption) for path, title, caption in figures if path and os.path.exists(path)]
    if not existing:
        return
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.05, 0.965, section, va="top", fontsize=16, fontweight="bold")
    positions = [
        (0.07, 0.55, 0.40, 0.29),
        (0.53, 0.55, 0.40, 0.29),
        (0.30, 0.14, 0.40, 0.29),
    ]
    for (path, title, caption), (x, y, w, h) in zip(existing[:3], positions):
        fig.text(x, y + h + 0.042, title, va="top", fontsize=9.2, fontweight="bold", wrap=True)
        fig.text(x, y + h + 0.016, caption, va="top", fontsize=6.8, wrap=True)
        ax = fig.add_axes([x, y, w, h])
        try:
            ax.imshow(mpimg.imread(path))
        except Exception:
            ax.text(0.5, 0.5, f"Failed to load {path}", ha="center", va="center", fontsize=8)
        ax.axis("off")
    _save_pdf_page(pdf, fig)
    plt.close(fig)


def _matplotlib_model_pages(pdf, prefix: str | None, task: str) -> None:
    pages = _structured_model_pages(prefix, task)
    if not pages:
        return
    section = "Classification Individual Models" if task == "classification" else "Regression Individual Models"
    for model, model_pages in pages.items():
        label = _pretty_model_name(model)
        page1, page2 = _model_report_pages(model_pages)
        _matplotlib_figure_pages(pdf, f"{section}: {label} Performance", page1, per_page=4)
        _matplotlib_figure_pages(pdf, f"{section}: {label} Diagnostics", page2, per_page=4)
        _matplotlib_interpretation_pages(
            pdf,
            f"{section}: {label} Interpretation",
            _model_interpretability_items(prefix, model, task),
        )


def _matplotlib_final_biomarker_summary(pdf, prefix: str | None) -> None:
    df = _read_csv(_find_csv("recommended_biomarkers.csv", prefix))
    if df.empty:
        return
    _matplotlib_table_page(
        pdf,
        "Final Biomarker Summary",
        _recommended_summary_table(df),
        "Top biomarkers ranked by the existing integrated BiomarkerML recommendation output.",
        max_rows=10,
        note=_recommended_summary_note(prefix, df),
    )


def _add_cover(pdf: PDFReport, prefix: str | None) -> None:
    pdf.add_page()
    pdf.set_font("Arial", "B", 24)
    pdf.cell(0, 18, "BiomarkerML Publication Report", 0, 1, "C")
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 7, "Publication-ready model benchmarking, protein-level interpretation, biomarker reliability, and biological network analysis.", align="C")
    if prefix:
        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, f"Run: {prefix}", 0, 1, "C")


def _add_workflow_summary(pdf: PDFReport, prefix: str | None) -> None:
    summary = _workflow_summary_table(prefix)
    if summary.empty:
        return
    rows = [(str(row["Item"]), str(row["Value"])) for _, row in summary.iterrows()]
    pdf.add_key_value_page("Workflow Summary", rows, "Scientifically relevant workflow configuration for this BiomarkerML run.")


def _add_benchmark_overview(pdf: PDFReport, prefix: str | None, task: str) -> None:
    paths = _benchmark_paths(prefix, task)
    comparison = _benchmark_table_for_report(_read_csv(paths["comparison_csv"]), task)
    ranking = _benchmark_ranking_for_report(_read_csv(paths["ranking_csv"]), task)
    summary = _read_json(paths["summary_json"])
    figures = _benchmark_figure_items(prefix, task)
    if comparison.empty and ranking.empty and not summary:
        return

    title = "Classification Benchmark Overview" if task == "classification" else "Regression Benchmark Overview"
    pdf.add_page()
    pdf.title_line(title)
    pdf.set_font("Arial", size=8.5)
    pdf.multi_cell(0, 4.7, "Benchmark outputs were generated immediately after prediction. The report presents those artifacts without recomputing benchmark statistics.")
    pdf.ln(1)
    if not comparison.empty:
        pdf.set_font("Arial", "B", 9.5)
        pdf.cell(0, 5, "Model Comparison", 0, 1)
        y_after_table = pdf.add_compact_table(comparison, max_rows=12)
    else:
        y_after_table = pdf.get_y()
    if summary:
        pdf.set_y(y_after_table + 3)
        pdf.set_font("Arial", "B", 9.5)
        pdf.cell(0, 5, "Automatically Recommended Model", 0, 1)
        pdf.set_font("Arial", size=8.2)
        pdf.multi_cell(0, 4.5, _benchmark_summary_text(summary))
    if figures[:1] and figures[0][0] and os.path.exists(figures[0][0]):
        y_image = max(pdf.get_y() + 11, 130)
        pdf.set_xy(12, y_image - 10)
        pdf.set_font("Arial", "B", 9.5)
        pdf.cell(0, 4.8, figures[0][1], 0, 1)
        pdf.set_x(12)
        pdf.set_font("Arial", size=7.4)
        pdf.multi_cell(184, 3.8, figures[0][2])
        try:
            pdf.image(figures[0][0], x=13, y=y_image, w=184)
        except RuntimeError:
            logging.warning("Failed to load image: %s", figures[0][0])

    comparison_section = "Classification Benchmark Figures" if task == "classification" else "Regression Benchmark Figures"
    primary = figures[1] if len(figures) > 1 else None
    pdf.add_benchmark_figure_page(comparison_section, primary, figures[2:])


def _benchmark_summary_text(summary: dict) -> str:
    metrics = summary.get("Metrics", {}) or {}
    metric_text = ", ".join(f"{_metric_label(key)}: {_format_table_value(value)}" for key, value in metrics.items() if value is not None)
    parts = [
        f"Best model: {summary.get('BestModel', '')}",
        f"Reason: {summary.get('Reason', '')}",
    ]
    if metric_text:
        parts.append(metric_text)
    return "\n".join(parts)


def _metric_label(value: str) -> str:
    return "R²" if str(value) == "R2" else str(value)


def _add_individual_models(pdf: PDFReport, prefix: str | None, task: str) -> None:
    pages = _structured_model_pages(prefix, task)
    if not pages:
        return
    section = "Classification Individual Models" if task == "classification" else "Regression Individual Models"
    for model, model_pages in pages.items():
        label = _pretty_model_name(model)
        page1, page2 = _model_report_pages(model_pages)
        pdf.add_figure_grid_pages(f"{section}: {label} Performance", page1, per_page=4)
        pdf.add_figure_grid_pages(f"{section}: {label} Diagnostics", page2, per_page=4)
        pdf.add_model_interpretation_pages(
            f"{section}: {label} Interpretation",
            _model_interpretability_items(prefix, model, task),
        )


def _model_report_pages(model_pages: dict[str, list[tuple[str, str, str]]]) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    performance_metrics, remaining_page2 = _take_first_title(model_pages["page2"], "Evaluation Metrics")
    nested_cv, other_page2 = _take_first_title(remaining_page2, "Nested Cross Validation")
    page1 = _dedupe_figure_list(model_pages["page1"] + performance_metrics)[:4]
    diagnostics = _prioritize_diagnostic_figures(nested_cv + other_page2 + model_pages["diagnostics"])
    return page1, diagnostics[:4]


def _learning_progression_report_title(path: str, default: str) -> str:
    if not str(path).lower().endswith("_learning_curve.png"):
        return default
    table = _read_csv(str(Path(path).with_suffix(".csv")))
    if table.empty:
        return default
    diagnostic = (
        str(table["DiagnosticType"].dropna().iloc[0]).lower()
        if "DiagnosticType" in table.columns and table["DiagnosticType"].notna().any()
        else ""
    )
    progression = (
        str(table["ProgressionLabel"].dropna().iloc[0]).lower()
        if "ProgressionLabel" in table.columns and table["ProgressionLabel"].notna().any()
        else ""
    )
    if "epoch" in diagnostic or "epoch" in progression:
        return "Learning Curve"
    if "regular" in diagnostic or "regular" in progression:
        return "Regularization Validation"
    return "Model Complexity Validation"


def _take_first_title(items: list[tuple[str, str, str]], title: str) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    selected: list[tuple[str, str, str]] = []
    remaining: list[tuple[str, str, str]] = []
    for item in items:
        if item[1] == title and not selected:
            selected.append(item)
        else:
            remaining.append(item)
    return selected, remaining


def _prioritize_diagnostic_figures(items: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    priority = {
        "Nested Cross Validation": 0,
        "Learning Curve": 1,
        "Regularization Validation": 1,
        "Model Complexity Validation": 1,
        "Training Size Diagnostic": 2,
        "Validation Curve": 4,
        "Optuna Optimization History": 5,
        "Boosting Iteration Diagnostic": 6,
    }
    deduped = _dedupe_figure_list(items)
    return sorted(deduped, key=lambda item: (priority.get(item[1], 100), item[1], item[0]))


def _dedupe_figure_list(items: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    seen_paths: set[str] = set()
    seen_titles: set[str] = set()
    clean: list[tuple[str, str, str]] = []
    for item in items:
        path, title, _ = item
        key = str(Path(path).resolve())
        if key in seen_paths:
            continue
        if title in seen_titles:
            continue
        seen_paths.add(key)
        seen_titles.add(title)
        clean.append(item)
    return clean


def _consensus_candidate_table(prefix: str | None) -> pd.DataFrame:
    table = _read_csv(_find_csv("consensus_candidate_model_metrics.csv", prefix))
    if table.empty:
        table = _read_csv(_find_csv("consensus_model_candidates.csv", prefix))
    if table.empty:
        return table
    regression = any(column in table.columns for column in ["MeanRMSE", "MeanMAE", "MeanR2"])
    task = "regression" if regression else "classification"
    ranking_name = "regression_model_ranking.csv" if regression else "classification_model_ranking.csv"
    ranking = _read_csv(_find_csv(ranking_name, prefix))
    rows = []
    for _, candidate in table.iterrows():
        model = str(candidate.get("Model", ""))
        benchmark = _matching_model_row(ranking, model)
        included = _as_bool(candidate.get("IsConsensusCandidate", False))
        row = {
            "Model": str(benchmark.get("Model", model)) if benchmark is not None else model,
            "Benchmark Rank": benchmark.get("Rank", "") if benchmark is not None else "",
            "Included in Consensus": included,
        }
        if task == "classification":
            row["Macro AUC"] = benchmark.get("Macro AUC", candidate.get("MeanAUC", np.nan)) if benchmark is not None else candidate.get("MeanAUC", np.nan)
            row["Weighted F1"] = benchmark.get("Weighted F1", candidate.get("MeanWeightedF1", np.nan)) if benchmark is not None else candidate.get("MeanWeightedF1", np.nan)
        else:
            row["RMSE"] = benchmark.get("RMSE", candidate.get("MeanRMSE", np.nan)) if benchmark is not None else candidate.get("MeanRMSE", np.nan)
            row["MAE"] = benchmark.get("MAE", candidate.get("MeanMAE", np.nan)) if benchmark is not None else candidate.get("MeanMAE", np.nan)
            row["R²"] = benchmark.get("R2", benchmark.get("R²", candidate.get("MeanR2", np.nan))) if benchmark is not None else candidate.get("MeanR2", np.nan)
        rows.append(row)
    result = pd.DataFrame(rows)
    if "Benchmark Rank" in result.columns:
        result = result.sort_values("Benchmark Rank", key=lambda values: pd.to_numeric(values, errors="coerce").fillna(np.inf))
    return result.reset_index(drop=True)


def _consensus_available(prefix: str | None) -> bool:
    return not _read_csv(_find_csv("consensus_biomarkers.csv", prefix)).empty


def _matching_model_row(df: pd.DataFrame, model: str) -> pd.Series | None:
    if df.empty or "Model" not in df.columns:
        return None
    aliases = _model_aliases_for_report(model)
    matches = df[df["Model"].astype(str).map(_normalize_model_name).isin(aliases)]
    return matches.iloc[0] if not matches.empty else None


def _as_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def _consensus_rule_text(candidates: pd.DataFrame) -> str:
    if candidates.empty:
        return ""
    included = candidates.loc[candidates["Included in Consensus"], "Model"].astype(str).tolist()
    models = ", ".join(included) if included else "none"
    return (
        f"Models included in consensus: {models}. "
        "Consensus biomarkers were derived by the existing consensus workflow from shared top-ranked proteins; "
        "the report does not recompute thresholds, model inclusion, or rankings."
    )


def _global_stability_items(prefix: str | None) -> list[tuple[str | None, str, str]]:
    return [
        (_find_suffix("_stability_heatmap.png", prefix), "Global Biomarker Stability", report_figure_caption("global_stability")),
        (
            _find_png("Stability.png") or _find_suffix("_biomarker_frequency_plot.png", prefix),
            "Cross-model Biomarker Frequency",
            report_figure_caption("biomarker_frequency"),
        ),
    ]


def _consensus_items(prefix: str | None) -> list[tuple[str | None, str, str]]:
    return [
        (
            _find_png("Consensus.png") or _find_suffix("_consensus_biomarker_ranking.png", prefix),
            "Consensus Biomarker Ranking",
            report_figure_caption("consensus_ranking"),
        ),
        (
            _find_png("Consensus_Support.png") or _find_suffix("_consensus_frequency_plot.png", prefix),
            "Consensus Support Frequency",
            report_figure_caption("consensus_frequency"),
        ),
    ]


def _recommended_items(prefix: str | None) -> list[tuple[str | None, str, str]]:
    note = _single_model_consensus_note(prefix)
    caption = report_figure_caption("recommended", note=f" {note}" if note else "")
    integrated = _find_png("Suggested_Biomarkers.png")
    return [
        (
            integrated or _find_suffix("_recommended_biomarkers.png", prefix),
            "Suggested Biomarkers",
            caption,
        )
    ]


def _recommended_explanation_item() -> str | None:
    return _find_png("Suggested_Biomarkers_Explanation.png")


def _resolve_ppi_figure(
    prefix: str | None,
    canonical_name: str,
    runtime_name: str,
    suffix: str,
) -> str | None:
    """Resolve current-run PPI figures before retaining legacy recursive fallback."""
    run_directory = Path(prefix) if prefix else Path(".")
    for candidate in (
        run_directory / "Final" / "PPI" / canonical_name,
        run_directory / "ppi" / runtime_name,
    ):
        if candidate.is_file():
            return str(candidate)
    return _find_suffix(suffix, prefix)


def _ppi_items(prefix: str | None) -> list[tuple[str | None, str, str]]:
    standard_figure = _resolve_ppi_figure(
        prefix,
        "Recommended_Biomarkers_PPI.png",
        f"{prefix}_ppi_network.png" if prefix else "ppi_network.png",
        "_ppi_network.png",
    )
    expanded_figure = _resolve_ppi_figure(
        prefix,
        "Expanded_PPI.png",
        f"{prefix}_ppi_expanded_network.png" if prefix else "ppi_expanded_network.png",
        "_ppi_expanded_network.png",
    )
    return [
        (standard_figure, "Recommended Biomarkers PPI", report_figure_caption("ppi_standard")),
        (
            _find_suffix("_ppi_network_hub_proteins.png", prefix),
            "Recommended Biomarkers: Hub Proteins",
            report_figure_caption("ppi_hubs"),
        ),
        (expanded_figure, "Expanded PPI Network", report_figure_caption("ppi_expanded")),
        (
            _find_suffix("_ppi_expanded_network_hub_proteins.png", prefix),
            "Expanded PPI: Hub Proteins",
            report_figure_caption("ppi_hubs"),
        ),
    ]


def _single_model_consensus_note(prefix: str | None) -> str:
    manifest = _read_csv(_find_csv("workflow_manifest.csv", prefix))
    if manifest.empty or not {"RecordType", "Name", "Status"}.issubset(manifest.columns):
        return ""
    rows = manifest[(manifest["RecordType"] == "Stage") & (manifest["Name"] == "Consensus")]
    if rows.empty or str(rows.iloc[0]["Status"]).upper() != "SKIPPED":
        return ""
    detail = str(rows.iloc[0].get("Detail", ""))
    if "one" not in detail.lower() and "single" not in detail.lower():
        return ""
    return "Consensus biomarker analysis was skipped because only one predictive model was executed."


def _add_global_biomarker_sections(pdf: PDFReport, prefix: str | None) -> None:
    pdf.add_figure_pair_page("Biomarker Reproducibility Across Models", _global_stability_items(prefix))
    candidates = _consensus_candidate_table(prefix)
    if _consensus_available(prefix):
        if not candidates.empty:
            pdf.add_table_page("Consensus Biomarkers: Contributing Models", candidates, _consensus_rule_text(candidates), max_rows=16)
        pdf.add_figure_pair_page("Consensus Biomarkers", _consensus_items(prefix))
    recommended = _recommended_items(prefix)
    pdf.add_suggested_biomarker_page(
        recommended[0] if recommended else None,
        _recommended_explanation_item(),
    )
    pdf.add_full_width_figure_pages("Biological Network Interpretation", _ppi_items(prefix))


def _add_manifest_section(pdf: PDFReport, prefix: str | None) -> None:
    stages = _manifest_stages_for_report(prefix)
    if stages.empty:
        return
    columns = [column for column in ["Name", "Status", "StartTime", "EndTime", "ElapsedTime"] if column in stages.columns]
    pdf.add_table_page("Workflow Manifest", stages[columns], "Completed workflow stage status and timing.", max_rows=24)


def _manifest_stages_for_report(prefix: str | None) -> pd.DataFrame:
    manifest = _read_csv(_find_csv("workflow_manifest.csv", prefix))
    if manifest.empty:
        return pd.DataFrame()
    stages = manifest[manifest.get("RecordType", "") == "Stage"].copy()
    if stages.empty or not {"Name", "Status"}.issubset(stages.columns):
        return stages
    pdf_row = stages["Name"].astype(str) == "PDF"
    if pdf_row.any():
        complete_pdf = ~stages["Status"].astype(str).eq("RUNNING")
        for column in ["EndTime", "ElapsedTime"]:
            if column in stages.columns:
                complete_pdf &= stages[column].fillna("").astype(str).str.strip().ne("")
        stages = stages.loc[~pdf_row | complete_pdf].copy()
    return stages


def _add_final_biomarker_summary(pdf: PDFReport, prefix: str | None) -> bool:
    path = _find_csv("recommended_biomarkers.csv", prefix)
    df = _read_csv(path)
    if df.empty:
        return False
    pdf.add_page()
    pdf.title_line("Final Biomarker Summary")
    top = _recommended_summary_table(df)
    pdf.text("Top biomarkers ranked by the existing integrated BiomarkerML recommendation output.")
    pdf.set_font("Arial", "B", 7)
    widths = _table_widths(list(top.columns), 190)
    for column, width in zip(top.columns, widths):
        pdf.cell(width, 6, _pdf_cell(str(column), 18), border=1)
    pdf.ln()
    pdf.set_font("Arial", size=7)
    for _, row in top.iterrows():
        for column, width in zip(top.columns, widths):
            pdf.cell(width, 6, _pdf_cell(_format_table_value(row[column]), 18), border=1)
        pdf.ln()
    pdf.ln(4)
    pdf.text(_recommended_summary_note(prefix, df))
    return True


def _recommended_summary_display_names(
    df: pd.DataFrame,
    source_column: str,
    *,
    offline: bool = False,
) -> dict[str, str]:
    """Resolve summary labels through the canonical publication annotation path."""
    from engines.ppi.protein_network import publication_gene_symbol_map

    identifiers = df[source_column].fillna("").astype(str).tolist()
    annotation_tables: list[pd.DataFrame] = []
    if "GeneSymbol" in df.columns:
        annotation_tables.append(
            pd.DataFrame(
                {
                    "Protein": df[source_column],
                    "GeneSymbol": df["GeneSymbol"],
                }
            )
        )
    for path in sorted(Path(".").rglob("*_ppi*_network_nodes.csv")):
        table = _read_csv(str(path))
        if {"Protein", "GeneSymbol"}.issubset(table.columns):
            annotation_tables.append(table[["Protein", "GeneSymbol"]])
    for path in sorted(Path(".").rglob("*_per_class_shap.csv")):
        table = _read_csv(str(path))
        if {"Feature", "DisplayFeature"}.issubset(table.columns):
            annotation_tables.append(
                table[["Feature", "DisplayFeature"]].rename(
                    columns={"Feature": "Protein", "DisplayFeature": "GeneSymbol"}
                )
            )

    annotations = pd.DataFrame(columns=["Protein", "GeneSymbol"])
    if annotation_tables:
        annotations = pd.concat(annotation_tables, ignore_index=True)
        annotations["Protein"] = annotations["Protein"].fillna("").astype(str).str.strip()
        annotations["GeneSymbol"] = annotations["GeneSymbol"].fillna("").astype(str).str.strip()
        invalid = annotations["GeneSymbol"].str.lower().isin({"", "nan", "none", "null"})
        annotations["_HasGeneSymbol"] = (~invalid) & (
            annotations["GeneSymbol"] != annotations["Protein"]
        )
        annotations = (
            annotations.sort_values("_HasGeneSymbol", ascending=False, kind="stable")
            .drop_duplicates("Protein")
            .drop(columns="_HasGeneSymbol")
        )
    return publication_gene_symbol_map(
        identifiers,
        annotations=annotations,
        offline=offline,
    )


def _recommended_summary_table(
    df: pd.DataFrame,
    *,
    offline: bool = False,
) -> pd.DataFrame:
    source_column = next(
        (column for column in ("Protein", "Feature", "UniProtID") if column in df.columns),
        None,
    )
    view = df.head(10).copy()
    if source_column is not None:
        display_names = _recommended_summary_display_names(
            view,
            source_column,
            offline=offline,
        )
        source_values = view[source_column].fillna("").astype(str)
        view.insert(
            0,
            "Biomarker",
            [display_names.get(identifier, identifier) for identifier in source_values],
        )
    elif "GeneSymbol" in view.columns:
        view.insert(0, "Biomarker", view["GeneSymbol"].fillna("").astype(str))

    preferred = [
        "Biomarker",
        "MeanImportance",
        "MeanAbsSHAP",
        "MeanSHAP",
        "StabilityScore",
        "ConsensusScore",
        "SupportFrequency",
        "SupportingModels",
        "RecommendedBiomarkerScore",
    ]
    columns = [column for column in preferred if column in view.columns]
    if not columns:
        columns = list(view.columns[:6])
    return view[columns[:8]].copy().rename(
        columns={
            "MeanImportance": "Mean importance",
            "MeanAbsSHAP": "Mean |SHAP|",
            "MeanSHAP": "Mean |SHAP|",
            "StabilityScore": "Stability",
            "ConsensusScore": "Consensus",
            "SupportFrequency": "Support",
            "SupportingModels": "Models",
            "RecommendedBiomarkerScore": "Recommendation",
        }
    )


def _recommended_summary_note(prefix: str | None, df: pd.DataFrame) -> str:
    parts = [
        "Recommendations summarize protein-level model attribution or importance, biomarker stability, and consensus support. Values are labelled as SHAP only when their recorded provenance is SHAP."
    ]
    if note := _single_model_consensus_note(prefix):
        parts.append(note)
    annotation_columns = [column for column in ["GeneSymbol", "UniProtID", "EntrezID", "GOAnnotations"] if column in df.columns]
    has_annotation = any(df[column].fillna("").astype(str).str.strip().ne("").any() for column in annotation_columns)
    parts.append("Available biological identifiers are reported without additional interpretation." if has_annotation else "Biological annotations were not available in the recommendation output.")
    status = _workflow_stage_status(prefix, "PPI")
    if status:
        parts.append(f"PPI stage status: {status}.")
    return " ".join(parts)


def _workflow_stage_status(prefix: str | None, stage: str) -> str:
    manifest = _read_csv(_find_csv("workflow_manifest.csv", prefix))
    if manifest.empty or not {"RecordType", "Name", "Status"}.issubset(manifest.columns):
        return ""
    rows = manifest[(manifest["RecordType"] == "Stage") & (manifest["Name"] == stage)]
    return str(rows.iloc[0]["Status"]) if not rows.empty else ""


def build_report(
    prefix: str | None = None,
    output: str = "model_reports.pdf",
    *,
    offline: bool = False,
) -> None:
    if prefix:
        from engines.report.publication_export import generate_final_export

        generate_final_export(Path.cwd(), prefix, offline=offline)
    if FPDF is None:
        _build_report_with_matplotlib(prefix, output)
        print(f"PDF report saved to {output}")
        return
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=12)
    _add_cover(pdf, prefix)
    _add_workflow_summary(pdf, prefix)
    pdf.add_full_width_figure_pages("Dataset Overview", _dataset_overview_items(prefix))
    pdf.add_figure_pair_page("Exploratory Analysis", _exploratory_report_items(prefix))
    _add_benchmark_overview(pdf, prefix, "classification")
    _add_benchmark_overview(pdf, prefix, "regression")
    _add_individual_models(pdf, prefix, "classification")
    _add_individual_models(pdf, prefix, "regression")
    _add_global_biomarker_sections(pdf, prefix)
    _add_final_biomarker_summary(pdf, prefix)
    _add_manifest_section(pdf, prefix)
    pdf.output(output)
    print(f"PDF report saved to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a BiomarkerML PDF report from workflow outputs.")
    parser.add_argument("--input", default=None, help="Task input directory containing workflow outputs.")
    parser.add_argument("-p", "--prefix", default=None)
    parser.add_argument("--outdir", default=".", help="Base output directory")
    parser.add_argument("--feature_selection", choices=["yes", "no"], default="no", help="Accepted for CLI consistency")
    parser.add_argument("--jobs", type=int, default=1, help="Accepted for CLI consistency")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("-o", "--output", default="publication_report.pdf")
    args = parser.parse_args()
    prefix = build_output_prefix(args.input, args.prefix)
    set_global_seed(args.seed)
    outdir = task_output_dir(args.outdir, "pdf_report")
    search_root = Path(args.input).resolve() if args.input and Path(args.input).is_dir() else Path(args.outdir).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = outdir / output.name
    previous = Path.cwd()
    os.chdir(search_root)
    try:
        build_report(prefix, str(output))
    finally:
        os.chdir(previous)


if __name__ == "__main__":
    main()
