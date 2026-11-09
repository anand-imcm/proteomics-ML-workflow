from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

try:
    from .runtime_warnings import run_with_runtime_output_filter
    from .execution_profile import PROFILE_CHOICES, cleanup_execution_cache, prepare_execution_profile
    from .overall_benchmark import run_overall_benchmark_task
    from .publication_utils import (
        DEFAULT_RANDOM_STATE,
        configure_publication_plots,
        ensure_publication_report_alias,
        resolve_workflow_run_directory,
        set_global_seed,
        stage_task_inputs,
        task_output_dir,
        task_run_context,
        write_artifact_alias,
    )
    from .workflow_tasks import (
        _alias_ppi_outputs,
        _alias_reliability_outputs,
        run_classification,
        run_classification_shap,
        run_regression,
        run_regression_shap,
    )
except ImportError:
    from runtime_warnings import run_with_runtime_output_filter
    from execution_profile import PROFILE_CHOICES, cleanup_execution_cache, prepare_execution_profile
    from overall_benchmark import run_overall_benchmark_task
    from publication_utils import (
        DEFAULT_RANDOM_STATE,
        configure_publication_plots,
        ensure_publication_report_alias,
        resolve_workflow_run_directory,
        set_global_seed,
        stage_task_inputs,
        task_output_dir,
        task_run_context,
        write_artifact_alias,
    )
    from workflow_tasks import (
        _alias_ppi_outputs,
        _alias_reliability_outputs,
        run_classification,
        run_classification_shap,
        run_regression,
        run_regression_shap,
    )


CLASSIFICATION_MODELS = ["KNN", "RF", "NN", "SVM", "XGB", "PLSDA", "VAE", "LGBM", "LR", "MLPVAE", "GNB"]
REGRESSION_MODELS = ["NN_reg", "RF_reg", "SVM_reg", "XGB_reg", "PLS_reg", "KNN_reg", "LGBM_reg", "VAE_reg", "MLPVAE_reg"]
CLASSIFICATION_SHAP_MODELS = ["KNN", "RF", "NN", "SVM", "XGB", "PLSDA", "VAE", "LGBM", "LR", "MLPVAE", "GNB"]
REGRESSION_SHAP_MODELS = ["NN_reg", "RF_reg", "SVM_reg", "XGB_reg", "PLS_reg", "KNN_reg", "LGBM_reg", "VAE_reg", "MLPVAE_reg"]
PREPROCESSING_TUNING_SUFFIXES = {
    "umap": "_umap_optuna_study.json",
    "tsne": "_tsne_optuna_study.json",
    "kpca": "_kpca_optuna_study.json",
    "elasticnet": "_elasticnet_optuna_study.json",
}
STAGES = [
    "ExecutionProfile",
    "ExploratoryAnalysis",
    "Prediction",
    "OverallBenchmark",
    "SHAP",
    "BiomarkerStability",
    "Consensus",
    "RecommendedBiomarkers",
    "PPI",
    "PDF",
]


@dataclass
class WorkflowState:
    args: argparse.Namespace
    prefix: str
    models: list[str]
    input_csv: Path | None = None
    profile_report: object | None = None
    shap_models: list[str] = field(default_factory=list)
    stage_status: dict[str, tuple[str, str]] = field(default_factory=dict)
    stage_timing: dict[str, dict[str, str]] = field(default_factory=dict)
    preprocessing_tuning_artifacts: dict[str, Path] = field(default_factory=dict)

    @property
    def manifest_path(self) -> Path:
        return task_output_dir(self.args.outdir, "workflow") / f"{self.prefix}_workflow_manifest.csv"

    def register_preprocessing_tuning_artifact(self, method: str, artifact: str | os.PathLike[str]) -> None:
        suffix = PREPROCESSING_TUNING_SUFFIXES.get(method)
        if suffix is None:
            raise ValueError(f"Unsupported preprocessing tuning study: {method}")
        expected = (task_output_dir(self.args.outdir, "preprocess") / f"{self.prefix}{suffix}").resolve()
        actual = Path(artifact).resolve(strict=True)
        if actual != expected:
            raise ValueError(
                f"Preprocessing tuning artifact does not belong to this workflow run: "
                f"expected '{expected}', received '{actual}'."
            )
        self.preprocessing_tuning_artifacts[method] = actual


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the canonical BiomarkerML workflow.")
    parser.add_argument("--input", required=True, help="Input CSV file.")
    parser.add_argument("--task", required=True, choices=["classification", "regression"], help="Scientific analysis task.")
    parser.add_argument("--label", required=True, help="Target column name.")
    parser.add_argument("--prefix", default=None, help="Output prefix. Defaults to Result_<YYYYMMDD_HHMMSS>.")
    parser.add_argument("--outdir", default="results", help="Base output directory.")
    parser.add_argument("--visualization", default="yes", choices=["yes", "no"], help="Run unsupervised exploratory analysis.")
    parser.add_argument("--exploration_only", dest="exploration_only", default="no", choices=["yes", "no"], help="Run exploratory analysis and stop.")
    parser.add_argument("--visualization_only", dest="exploration_only", choices=["yes", "no"], help=argparse.SUPPRESS)
    parser.add_argument("--feature_selection", default="yes", choices=["yes", "no"], help="Use automatic protein feature selection.")
    parser.add_argument("--imputation", choices=["median", "knn"], default="median", help="Fold-local biomarker imputation method.")
    parser.add_argument("--covariates", nargs="*", default=None, metavar="COLUMN", help="Optional adjustment predictors; off by default.")
    parser.add_argument("--categorical-covariates", nargs="*", default=None, metavar="COLUMN", help="Covariates to encode categorically, including numeric-coded categories.")
    parser.add_argument("--models", nargs="*", default=None, help="Models to run. Defaults to all supported models for the selected task.")
    parser.add_argument("--run_shap", default="no", choices=["yes", "no"], help="Run protein-level SHAP and biomarker reliability.")
    parser.add_argument("--run_ppi", default="no", choices=["yes", "no"], help="Run PPI analysis from recommended biomarkers.")
    parser.add_argument("--profile", choices=PROFILE_CHOICES, default="balanced", help="Runtime execution profile.")
    parser.add_argument("--jobs", type=int, default=None, help="Parallel workers. Defaults to the selected execution profile.")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE, help="Random seed.")
    parser.add_argument("--num_features", type=int, default=20, help="Top features for SHAP visualizations.")
    parser.add_argument("--species", type=int, default=9606, help="STRING species identifier for PPI.")
    parser.add_argument("--offline", action="store_true", help="Disable online annotation and STRING requests for PPI.")
    parser.add_argument("--top_n", type=int, default=100, help="Top recommended biomarkers used as PPI seeds.")
    parser.add_argument("--score_threshold", type=int, default=400, help="STRING expanded-network confidence threshold.")
    parser.add_argument("--combined_score_threshold", type=int, default=800, help="STRING standard-network confidence threshold.")
    parser.add_argument("--correlation_method", default="spearman", choices=["spearman", "pearson"], help="Co-expression correlation method.")
    parser.add_argument("--correlation_threshold", type=float, default=0.8, help="Co-expression absolute correlation threshold.")
    parser.add_argument("--expanded_limit", type=int, default=5, help="Maximum expanded STRING interactors per seed.")
    args = parser.parse_args()
    if any(arg == "--visualization_only" or arg.startswith("--visualization_only=") for arg in sys.argv[1:]):
        print("WARNING:\n\n--visualization_only is deprecated and will be removed in a future release.\n\nPlease use:\n\n--exploration_only")
    return args


def main() -> None:
    run_with_runtime_output_filter(_main)


def _main() -> None:
    args = parse_args()
    _validate_workflow_args(args)
    resolved_prefix, run_dir = resolve_workflow_run_directory(args.outdir, args.prefix)
    args.prefix = resolved_prefix
    args.outdir = str(run_dir)
    print(f"Workflow output directory: {run_dir}")
    models = _resolve_models(args.task, args.models)
    state = WorkflowState(args=args, prefix=resolved_prefix, models=models)
    try:
        run_execution_profile(state)
        set_global_seed(args.seed)
        state.input_csv = _prepare_labeled_input(args.input, args.label, args.outdir, resolved_prefix)
        write_workflow_manifest(state)
        run_exploratory_analysis(state)
        if args.exploration_only == "yes":
            skip_prediction_downstream(state, "Exploratory-analysis-only workflow.")
            run_pdf(state, "visualization_report.pdf")
            return

        run_prediction(state)
        run_overall_benchmark(state)
        if args.run_shap == "no":
            skip_stage(state, "SHAP", "User requested --run_shap no.")
            skip_stage(state, "BiomarkerStability", "SHAP was not requested.")
            skip_stage(state, "Consensus", "SHAP was not requested.")
            skip_stage(state, "RecommendedBiomarkers", "SHAP was not requested.")
            skip_stage(state, "PPI", "SHAP was not requested.")
            run_pdf(state, "basic_report.pdf")
            return

        run_shap(state)
        run_biomarker_stability(state)
        run_consensus(state)
        run_recommended_biomarkers(state)
        if args.run_ppi == "no":
            skip_stage(state, "PPI", "User requested --run_ppi no.")
            run_pdf(state, "extended_report.pdf")
            return

        run_ppi(state)
        run_pdf(state, "final_report.pdf")
    finally:
        if state.profile_report is not None:
            cleanup_execution_cache(state.profile_report.settings)


def _validate_workflow_args(args: argparse.Namespace) -> None:
    if args.exploration_only == "yes" and args.visualization == "no":
        raise SystemExit("--exploration_only requires --visualization yes.")
    if args.run_ppi == "yes" and args.run_shap == "no":
        raise SystemExit("PPI requires SHAP outputs because PPI consumes Recommended Biomarkers.")


def _resolve_models(task: str, requested: list[str] | None) -> list[str]:
    defaults = CLASSIFICATION_MODELS if task == "classification" else REGRESSION_MODELS
    if not requested or requested == ["all"]:
        return defaults.copy()
    unknown = sorted(set(requested) - set(defaults))
    if unknown:
        allowed = ", ".join(defaults)
        raise SystemExit(f"Unsupported {task} model(s): {', '.join(unknown)}. Allowed models: all, {allowed}")
    return requested


def _prepare_profile(args: argparse.Namespace, models: list[str]):
    report_dir = task_output_dir(args.outdir, "resource_report")
    return prepare_execution_profile(
        input_path=args.input,
        outdir=args.outdir,
        report_dir=report_dir,
        task_name="workflow",
        profile=args.profile,
        requested_jobs=args.jobs,
        models=models,
        task_type=args.task,
        shap_enabled=args.run_shap == "yes",
        ppi_enabled=args.run_ppi == "yes",
        benchmark_enabled=args.exploration_only != "yes",
    )


def run_execution_profile(state: WorkflowState) -> None:
    def action() -> None:
        state.profile_report = _prepare_profile(state.args, state.models)

    run_stage(state, "ExecutionProfile", action, "Runtime resources configured.")


def run_exploratory_analysis(state: WorkflowState) -> None:
    if state.args.visualization == "no":
        skip_stage(state, "ExploratoryAnalysis", "User requested --visualization no.")
        return

    def action() -> None:
        studies = _run_visualization(state.args, require_input_csv(state), state.prefix, workflow_jobs(state))
        for method, artifact in studies.items():
            state.register_preprocessing_tuning_artifact(method, artifact)

    run_stage(state, "ExploratoryAnalysis", action, "Unsupervised exploratory analysis completed.")


def run_prediction(state: WorkflowState) -> None:
    def action() -> None:
        _run_prediction(state.args, require_input_csv(state), state.prefix, state.models, workflow_jobs(state))

    run_stage(state, "Prediction", action, f"{state.args.task.title()} prediction completed.")


def run_overall_benchmark(state: WorkflowState) -> None:
    def action() -> None:
        ns = SimpleNamespace(
            input=str(task_output_dir(state.args.outdir, state.args.task)),
            prefix=state.prefix,
            outdir=state.args.outdir,
            task=state.args.task,
            feature_selection=state.args.feature_selection,
            jobs=workflow_jobs(state),
            seed=state.args.seed,
        )
        run_overall_benchmark_task(ns)

    run_stage(state, "OverallBenchmark", action, "Overall benchmark completed.")


def run_shap(state: WorkflowState) -> None:
    def action() -> None:
        state.shap_models = _run_shap(state.args, require_input_csv(state), state.prefix, state.models, workflow_jobs(state))
        if not state.shap_models:
            raise RuntimeError("No SHAP-compatible predictive models were selected.")

    run_stage(state, "SHAP", action, "Protein-level SHAP completed.")


def run_biomarker_stability(state: WorkflowState) -> None:
    from engines.reliability.biomarker_reliability import run_stability

    def action() -> None:
        prepare_reliability_inputs(state)
        with task_run_context(state.args.outdir, "biomarker_reliability") as workdir:
            run_stability(state.prefix)
            _alias_reliability_outputs(state.prefix, workdir)

    run_stage(state, "BiomarkerStability", action, "Biomarker stability completed.")


def run_consensus(state: WorkflowState) -> None:
    from engines.reliability.biomarker_reliability import run_consensus as compute_consensus

    def action() -> None:
        prepare_reliability_inputs(state)
        with task_run_context(state.args.outdir, "biomarker_reliability") as workdir:
            _clear_consensus_outputs(state.prefix, workdir)
            compute_consensus(state.prefix, task=state.args.task)
            _alias_reliability_outputs(state.prefix, workdir)

    if len(state.models) <= 1:
        with task_run_context(state.args.outdir, "biomarker_reliability") as workdir:
            _clear_consensus_outputs(state.prefix, workdir)
        print("Consensus Biomarkers skipped because only one predictive model was executed.")
        skip_stage(state, "Consensus", "Only one predictive model was executed.")
        return
    run_stage(state, "Consensus", action, "Consensus biomarkers completed.")


def run_recommended_biomarkers(state: WorkflowState) -> None:
    from engines.reliability.biomarker_reliability import run_recommended, write_publication_summary

    def action() -> None:
        prepare_reliability_inputs(state)
        with task_run_context(state.args.outdir, "biomarker_reliability") as workdir:
            run_recommended(state.prefix)
            write_publication_summary(state.prefix, task=state.args.task, input_csv=str(require_input_csv(state)))
            _alias_reliability_outputs(state.prefix, workdir)

    run_stage(state, "RecommendedBiomarkers", action, "Recommended biomarkers completed.")


def run_ppi(state: WorkflowState) -> None:
    def action() -> None:
        _run_ppi_from_recommended(state.args, require_input_csv(state), state.prefix, workflow_jobs(state))

    run_stage(state, "PPI", action, "PPI analysis completed.")


def run_pdf(state: WorkflowState, output_name: str) -> None:
    def action() -> None:
        _build_adaptive_report(
            state.args.outdir,
            state.prefix,
            output_name,
            offline=state.args.offline,
        )

    run_stage(state, "PDF", action, "publication_report.pdf")


def run_stage(state: WorkflowState, stage: str, action, success_detail: str) -> None:
    start_dt = datetime.now()
    start_perf = time.perf_counter()
    state.stage_timing[stage] = {
        "StartTime": start_dt.isoformat(timespec="seconds"),
        "EndTime": "",
        "ElapsedTime": "",
    }
    state.stage_status[stage] = ("RUNNING", "")
    write_workflow_manifest(state)
    _print_stage_progress(stage, "RUNNING")
    try:
        action()
    except Exception as exc:
        end_dt = datetime.now()
        state.stage_timing[stage] = {
            "StartTime": start_dt.isoformat(timespec="seconds"),
            "EndTime": end_dt.isoformat(timespec="seconds"),
            "ElapsedTime": _format_elapsed(time.perf_counter() - start_perf),
        }
        _print_stage_progress(stage, "FAIL", detail=str(exc), elapsed=state.stage_timing[stage]["ElapsedTime"])
        mark_stage(state, stage, "FAIL", str(exc))
        raise
    end_dt = datetime.now()
    state.stage_timing[stage] = {
        "StartTime": start_dt.isoformat(timespec="seconds"),
        "EndTime": end_dt.isoformat(timespec="seconds"),
        "ElapsedTime": _format_elapsed(time.perf_counter() - start_perf),
    }
    _print_stage_progress(stage, "PASS", detail=success_detail, elapsed=state.stage_timing[stage]["ElapsedTime"])
    mark_stage(state, stage, "PASS", success_detail)


def mark_stage(state: WorkflowState, stage: str, status: str, detail: str = "") -> None:
    state.stage_status[stage] = (status, detail)
    write_workflow_manifest(state)


def skip_stage(state: WorkflowState, stage: str, detail: str = "") -> None:
    now = datetime.now().isoformat(timespec="seconds")
    state.stage_timing.setdefault(stage, {"StartTime": now, "EndTime": now, "ElapsedTime": "00:00:00"})
    _print_stage_progress(stage, "SKIPPED", detail=detail, elapsed="00:00:00")
    mark_stage(state, stage, "SKIPPED", detail)


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _print_stage_progress(stage: str, status: str, *, detail: str = "", elapsed: str = "") -> None:
    print("\n" + "=" * 56)
    print("BiomarkerML Workflow Progress")
    print("=" * 56)
    print(f"Stage: {stage}")
    print(f"Status: {status}")
    if detail:
        print(detail)
    if elapsed:
        print(f"Elapsed: {elapsed}")
    print("=" * 56)


def skip_prediction_downstream(state: WorkflowState, detail: str) -> None:
    for stage in ["Prediction", "OverallBenchmark", "SHAP", "BiomarkerStability", "Consensus", "RecommendedBiomarkers", "PPI"]:
        skip_stage(state, stage, detail)


def workflow_jobs(state: WorkflowState) -> int:
    if state.profile_report is None:
        return state.args.jobs or 1
    return state.profile_report.settings.joblib_workers


def require_input_csv(state: WorkflowState) -> Path:
    if state.input_csv is None:
        raise RuntimeError("Workflow input CSV has not been prepared.")
    return state.input_csv


def prepare_reliability_inputs(state: WorkflowState) -> Path:
    outdir = task_output_dir(state.args.outdir, "biomarker_reliability")
    stage_task_inputs(
        task_output_dir(state.args.outdir, state.args.task),
        outdir,
        state.prefix,
        patterns=(
            f"{state.prefix}_*_fold*_biomarkers.csv",
            f"{state.prefix}_*_outer_fold_metrics.csv",
            f"{state.prefix}_*_predictions.csv",
        ),
    )
    stage_task_inputs(
        task_output_dir(state.args.outdir, f"{state.args.task}_shap"),
        outdir,
        state.prefix,
        patterns=(f"{state.prefix}_*_shap_values.csv",),
    )
    _stage_source_input(require_input_csv(state), outdir)
    return outdir


def _prepare_labeled_input(input_path: str, label: str, outdir: str, prefix: str) -> Path:
    source = Path(input_path).resolve()
    if not source.exists():
        raise SystemExit(f"Input CSV not found: {source}")
    df = pd.read_csv(source)
    if label not in df.columns:
        raise SystemExit(f"Label column '{label}' was not found in {source}.")
    if label != "Label":
        if "Label" in df.columns:
            raise SystemExit("Input contains both the requested label column and an existing 'Label' column.")
        df = df.rename(columns={label: "Label"})
    input_dir = task_output_dir(outdir, "workflow") / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    normalized = input_dir / f"{prefix}_input.csv"
    df.to_csv(normalized, index=False)
    return normalized


def write_workflow_manifest(state: WorkflowState) -> None:
    args = state.args
    rows = [
        {"RecordType": "Parameter", "Name": "Input", "Value": str(Path(args.input).resolve()), "Status": "", "Detail": ""},
        {"RecordType": "Parameter", "Name": "WorkflowInput", "Value": str(state.input_csv or ""), "Status": "", "Detail": ""},
        {"RecordType": "Parameter", "Name": "Task", "Value": args.task, "Status": "", "Detail": ""},
        {"RecordType": "Parameter", "Name": "Label", "Value": args.label, "Status": "", "Detail": ""},
        {"RecordType": "Parameter", "Name": "Prefix", "Value": state.prefix, "Status": "", "Detail": ""},
        {"RecordType": "Parameter", "Name": "Visualization", "Value": args.visualization, "Status": "", "Detail": ""},
        {"RecordType": "Parameter", "Name": "ExplorationOnly", "Value": args.exploration_only, "Status": "", "Detail": ""},
        {"RecordType": "Parameter", "Name": "FeatureSelection", "Value": args.feature_selection, "Status": "", "Detail": ""},
        {"RecordType": "Parameter", "Name": "Imputation", "Value": args.imputation, "Status": "", "Detail": ""},
        {"RecordType": "Parameter", "Name": "MissingnessThreshold", "Value": 0.50, "Status": "", "Detail": "drop >0.50; retain ==0.50"},
        {"RecordType": "Parameter", "Name": "Covariates", "Value": ";".join(args.covariates or []), "Status": "", "Detail": "predictive adjustment; excluded from biomarkers"},
        {"RecordType": "Parameter", "Name": "CategoricalCovariates", "Value": ";".join(args.categorical_covariates or []), "Status": "", "Detail": "fold-local one-hot encoding"},
        {"RecordType": "Parameter", "Name": "Models", "Value": ";".join(state.models), "Status": "", "Detail": ""},
        {"RecordType": "Parameter", "Name": "RunSHAP", "Value": args.run_shap, "Status": "", "Detail": ""},
        {"RecordType": "Parameter", "Name": "RunPPI", "Value": args.run_ppi, "Status": "", "Detail": ""},
        {"RecordType": "Parameter", "Name": "Profile", "Value": args.profile, "Status": "", "Detail": ""},
        {"RecordType": "Parameter", "Name": "Seed", "Value": args.seed, "Status": "", "Detail": ""},
    ]
    manifest_json = state.manifest_path.with_suffix(".json")
    tuning_references = [
        os.path.relpath(path, start=manifest_json.parent)
        for _, path in sorted(state.preprocessing_tuning_artifacts.items())
    ]
    if tuning_references:
        rows.append({
            "RecordType": "Parameter",
            "Name": "PreprocessingAdaptiveTuningArtifacts",
            "Value": ";".join(tuning_references),
            "Status": "",
            "Detail": "Explicit current-run adaptive-tuning records",
        })
    for stage in STAGES:
        status, detail = state.stage_status.get(stage, ("PENDING", ""))
        timing = state.stage_timing.get(stage, {})
        rows.append({
            "RecordType": "Stage",
            "Name": stage,
            "Value": "",
            "Status": status,
            "Detail": detail,
            "StartTime": timing.get("StartTime", ""),
            "EndTime": timing.get("EndTime", ""),
            "ElapsedTime": timing.get("ElapsedTime", ""),
        })
    pd.DataFrame(rows).to_csv(state.manifest_path, index=False)
    if state.input_csv is not None and state.input_csv.exists():
        try:
            from .data_contracts import resolve_feature_roles, write_reproducibility_manifest
        except ImportError:
            from data_contracts import resolve_feature_roles, write_reproducibility_manifest
        manifest_data = pd.read_csv(state.input_csv)
        roles = resolve_feature_roles(
            manifest_data,
            target="Label",
            identifier="SampleID",
            covariates=args.covariates,
            categorical_covariates=args.categorical_covariates,
        )
        write_reproducibility_manifest(
            manifest_json,
            input_path=state.input_csv,
            task=args.task,
            model="workflow",
            seed=args.seed,
            roles=roles,
            imputation=args.imputation,
            final_hyperparameters={},
            artifacts=[state.manifest_path.name],
            tuning_artifacts=tuning_references,
        )


def _run_visualization(args: argparse.Namespace, input_csv: Path, prefix: str, jobs: int) -> dict[str, Path]:
    from engines.preprocessing.Step1_Preprocessing import perform_dimensionality_reduction

    warnings: list[dict[str, str]] = []
    completed = 0
    tuning_artifacts: dict[str, Path] = {}
    _write_exploratory_summaries(input_csv, args.outdir, prefix)
    with task_run_context(args.outdir, "preprocess"):
        for method in ["pca", "kpca", "tsne", "umap"]:
            try:
                study_artifact = perform_dimensionality_reduction(
                    str(input_csv),
                    prefix,
                    method=method,
                    dims=3,
                    standardize=True,
                    random_state=args.seed,
                )
                if study_artifact is not None:
                    tuning_artifacts[method] = Path(study_artifact)
                completed += 1
            except Exception as exc:
                message = f"{method.upper()} visualization skipped: {exc}"
                print(f"WARNING: {message}")
                warnings.append({"Method": method, "Status": "Skipped", "Reason": str(exc)})
        if warnings:
            pd.DataFrame(warnings).to_csv(f"{prefix}_exploratory_warnings.csv", index=False)
    if completed == 0:
        raise SystemExit("Exploratory analysis failed for every requested method.")
    return tuning_artifacts


def _write_exploratory_summaries(input_csv: Path, outdir: str, prefix: str) -> None:
    configure_publication_plots()
    import matplotlib.pyplot as plt

    df = pd.read_csv(input_csv)
    feature_df = df.drop(columns=["Label"], errors="ignore").select_dtypes(include=[np.number])
    workdir = task_output_dir(outdir, "preprocess")
    df.describe(include="all").transpose().to_csv(workdir / f"{prefix}_dataset_summary.csv")
    missing = pd.DataFrame(
        {
            "Feature": df.columns,
            "MissingCount": df.isna().sum().to_numpy(),
            "MissingFraction": (df.isna().mean()).to_numpy(),
        }
    )
    missing.to_csv(workdir / f"{prefix}_missing_value_summary.csv", index=False)
    if feature_df.empty:
        return

    values = feature_df.to_numpy(dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size:
        display_values, x_label, subtitle = _feature_distribution_display(values)
        fig = plt.figure(figsize=(8.8, 5.4))
        grid = fig.add_gridspec(2, 2, width_ratios=[3.3, 1.25], height_ratios=[4.0, 1.0], hspace=0.08, wspace=0.28)
        ax = fig.add_subplot(grid[0, 0])
        ax_box = fig.add_subplot(grid[1, 0], sharex=ax)
        ax_stats = fig.add_subplot(grid[:, 1])
        counts, bins, _ = ax.hist(
            display_values,
            bins=min(90, max(20, int(np.sqrt(display_values.size)))),
            color="#4C78A8",
            edgecolor="white",
            linewidth=0.4,
            alpha=0.88,
        )
        _plot_distribution_kde(ax, display_values, bins, counts)
        _horizontal_boxplot(
            ax_box,
            display_values,
            widths=0.55,
            patch_artist=True,
            boxprops={"facecolor": "#D8E5F2", "edgecolor": "#4C78A8"},
            medianprops={"color": "#E45756", "linewidth": 1.4},
        )
        ax_box.set_yticks([])
        plt.setp(ax.get_xticklabels(), visible=False)
        ax_box.set_xlabel(x_label)
        ax.set_ylabel("Frequency")
        ax_stats.axis("off")
        ax_stats.text(0.0, 0.98, _feature_distribution_stats(feature_df, values), va="top", ha="left", fontsize=9.5, linespacing=1.45)
        _set_overview_title(fig, ax, "Dataset Overview", subtitle)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax_box.spines["top"].set_visible(False)
        ax_box.spines["right"].set_visible(False)
        ax_box.spines["left"].set_visible(False)
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.86, wspace=0.3, hspace=0.08)
        fig.savefig(workdir / f"{prefix}_feature_distribution.png")
        plt.close(fig)

    heatmap_df, subset_used = _select_correlation_features(feature_df)
    corr = heatmap_df.corr(method="spearman").fillna(0.0)
    display = corr
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(display, cmap="vlag" if _has_colormap("vlag") else "coolwarm", vmin=-1, vmax=1, aspect="auto")
    step = max(1, int(np.ceil(display.shape[0] / 24)))
    ticks = np.arange(0, display.shape[0], step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([_short_feature_label(display.columns[i]) for i in ticks], rotation=90, fontsize=6)
    ax.set_yticklabels([_short_feature_label(display.index[i]) for i in ticks], fontsize=6)
    subtitle = (
        f"Deterministic top {display.shape[0]} variance-ranked features shown"
        if subset_used
        else f"{display.shape[0]} numeric features shown"
    )
    _set_overview_title(fig, ax, "Correlation Heatmap", subtitle)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Spearman r")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(workdir / f"{prefix}_correlation_heatmap.png")
    plt.close(fig)


def _set_overview_title(fig, ax, title: str, subtitle: str) -> None:
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.97)
    ax.set_title(subtitle, fontsize=9.5, fontweight="normal", color="#4B5563", pad=8)


def _plot_distribution_kde(ax, display_values: np.ndarray, bins: np.ndarray, counts: np.ndarray) -> None:
    if display_values.size < 3 or np.nanstd(display_values) == 0:
        return
    try:
        from scipy.stats import gaussian_kde
    except Exception:
        return
    x_grid = np.linspace(float(np.nanmin(display_values)), float(np.nanmax(display_values)), 256)
    bin_width = float(np.nanmedian(np.diff(bins))) if len(bins) > 1 else 1.0
    density = gaussian_kde(display_values)(x_grid) * display_values.size * bin_width
    ax.plot(x_grid, density, color="#E45756", linewidth=1.8, label="KDE")
    ax.legend(frameon=False, loc="upper right")


def _horizontal_boxplot(ax, values: np.ndarray, **kwargs) -> None:
    try:
        ax.boxplot(values, orientation="horizontal", **kwargs)
    except TypeError:
        legacy_kwargs = dict(kwargs)
        legacy_kwargs["vert"] = False
        ax.boxplot(values, **legacy_kwargs)


def _feature_distribution_stats(feature_df: pd.DataFrame, values: np.ndarray) -> str:
    q1, median, q3 = np.nanpercentile(values, [25, 50, 75])
    p1, p99 = np.nanpercentile(values, [1, 99])
    missing_rate = float(feature_df.isna().to_numpy().mean()) if feature_df.size else 0.0
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values))
    minimum = float(np.nanmin(values))
    maximum = float(np.nanmax(values))
    return "\n".join(
        [
            "Dataset Overview",
            f"Samples: {feature_df.shape[0]:,}",
            f"Features: {feature_df.shape[1]:,}",
            f"Missing rate: {missing_rate:.2%}",
            f"Minimum: {minimum:.3g}",
            f"Maximum: {maximum:.3g}",
            f"Median: {median:.3g}",
            f"Mean: {mean:.3g}",
            f"Std. dev.: {std:.3g}",
            f"IQR: {(q3 - q1):.3g}",
            f"1st percentile: {p1:.3g}",
            f"99th percentile: {p99:.3g}",
        ]
    )


def _feature_distribution_display(values: np.ndarray) -> tuple[np.ndarray, str, str]:
    lower, upper = np.nanpercentile(values, [1.0, 99.0])
    clipped = np.clip(values, lower, upper) if np.isfinite(lower) and np.isfinite(upper) and lower < upper else values.copy()
    finite = clipped[np.isfinite(clipped)]
    if finite.size == 0:
        return finite, "Feature value", "Finite feature values"

    if np.nanmin(finite) >= 0:
        return np.log1p(finite), "log1p(feature value)", "Visualization only: 1st-99th percentile clipping + log1p display"
    transformed = np.sign(finite) * np.log1p(np.abs(finite))
    return transformed, "signed log1p(|feature value|)", "Visualization only: 1st-99th percentile clipping + signed log1p display"


def _select_correlation_features(feature_df: pd.DataFrame, max_features: int = 40) -> tuple[pd.DataFrame, bool]:
    clean = feature_df.replace([np.inf, -np.inf], np.nan)
    if clean.shape[1] <= max_features:
        return clean, False
    variance = clean.var(axis=0, skipna=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ordered = sorted(variance.index, key=lambda col: (-float(variance.loc[col]), str(col)))
    return clean.loc[:, ordered[:max_features]], True


def _short_feature_label(value: object, max_length: int = 18) -> str:
    text = str(value)
    return text if len(text) <= max_length else f"{text[: max_length - 1]}..."


def _has_colormap(name: str) -> bool:
    import matplotlib.pyplot as plt

    try:
        plt.get_cmap(name)
        return True
    except ValueError:
        return False


def _run_prediction(args: argparse.Namespace, input_csv: Path, prefix: str, models: list[str], jobs: int) -> None:
    ns = SimpleNamespace(
        input=str(input_csv),
        prefix=prefix,
        outdir=args.outdir,
        feature_selection=args.feature_selection,
        jobs=jobs,
        seed=args.seed,
        profile=args.profile,
        models=models,
        imputation=args.imputation,
        covariates=args.covariates,
        categorical_covariates=args.categorical_covariates,
    )
    if args.task == "classification":
        run_classification(ns)
    else:
        run_regression(ns)


def _run_shap(args: argparse.Namespace, input_csv: Path, prefix: str, models: list[str], jobs: int) -> list[str]:
    supported = CLASSIFICATION_SHAP_MODELS if args.task == "classification" else REGRESSION_SHAP_MODELS
    shap_models = [model for model in models if model in supported]
    if not shap_models:
        return []
    model_dir = task_output_dir(args.outdir, args.task)
    ns = SimpleNamespace(
        input=str(model_dir),
        prefix=prefix,
        outdir=args.outdir,
        feature_selection=args.feature_selection,
        jobs=jobs,
        seed=args.seed,
        profile=args.profile,
        models=shap_models,
        num_features=args.num_features,
        species=args.species,
        offline=args.offline,
    )
    if args.task == "classification":
        run_classification_shap(ns)
    else:
        run_regression_shap(ns)
    _stage_source_input(input_csv, task_output_dir(args.outdir, "classification_shap" if args.task == "classification" else "regression_shap"))
    return shap_models


def _clear_consensus_outputs(prefix: str, workdir: Path) -> None:
    for suffix in [
        "consensus_biomarkers.csv",
        "consensus_biomarker_summary.csv",
        "consensus_model_candidates.csv",
        "consensus_candidate_model_metrics.csv",
        "consensus_biomarker_ranking.png",
        "consensus_biomarker_ranking.pdf",
        "consensus_biomarker_ranking.tiff",
        "consensus_frequency_plot.png",
        "consensus_frequency_plot.pdf",
        "consensus_frequency_plot.tiff",
    ]:
        path = workdir / f"{prefix}_{suffix}"
        if path.exists():
            path.unlink()
        alias = workdir / suffix
        if alias.exists():
            alias.unlink()


def _run_ppi_from_recommended(args: argparse.Namespace, input_csv: Path, prefix: str, jobs: int) -> None:
    from engines.ppi.protein_network import run_network

    set_global_seed(args.seed)
    outdir = task_output_dir(args.outdir, "ppi")
    reliability_dir = task_output_dir(args.outdir, "biomarker_reliability")
    recommended_source = reliability_dir / f"{prefix}_recommended_biomarkers.csv"
    if not recommended_source.exists():
        raise SystemExit("Canonical recommended-biomarker input is required before PPI can run.")
    recommended_target = outdir / recommended_source.name
    if recommended_target.exists() and recommended_target.resolve() != recommended_source.resolve():
        recommended_target.unlink()
    if recommended_target.resolve() != recommended_source.resolve():
        shutil.copy2(recommended_source, recommended_target)
    _stage_source_input(input_csv, outdir)
    with task_run_context(args.outdir, "ppi") as workdir:
        _remove_direct_consensus_ppi_inputs(prefix, workdir)
        pattern = _write_recommended_ppi_importance(prefix, workdir, args.task)
        ns = SimpleNamespace(
            score_thresholdHere=args.score_threshold,
            combined_score_thresholdHere=args.combined_score_threshold,
            SHAPthresh=args.top_n,
            patternChosen=pattern,
            converProId="TRUE",
            proteinExpFile=str(input_csv),
            CorMethod=args.correlation_method,
            CorThreshold=args.correlation_threshold,
            prefix=prefix,
            species=args.species,
            offline=args.offline,
            expanded_limit=args.expanded_limit,
            jobs=jobs,
        )
        run_network(ns)
        _alias_ppi_outputs(prefix, workdir)


def _remove_direct_consensus_ppi_inputs(prefix: str, workdir: Path) -> None:
    for filename in [f"{prefix}_consensus_biomarkers.csv", "consensus_biomarkers.csv"]:
        path = workdir / filename
        if path.exists():
            path.unlink()


def _write_recommended_ppi_importance(prefix: str, workdir: Path, task: str) -> str:
    source = workdir / f"{prefix}_recommended_biomarkers.csv"
    if not source.exists():
        raise SystemExit("Recommended biomarkers are required before PPI can run.")
    df = pd.read_csv(source)
    if "FeatureRole" in df.columns:
        df = df.loc[df["FeatureRole"].astype(str).str.upper().eq("BIOMARKER")].copy()
    if df.empty:
        raise SystemExit("Recommended-biomarker PPI input contains no biomarker-role features.")
    feature_col = "Protein" if "Protein" in df.columns else "Feature" if "Feature" in df.columns else None
    if feature_col is None:
        raise SystemExit("Recommended biomarkers must contain a Protein or Feature column.")
    score_col = next((col for col in ["RecommendedBiomarkerScore", "ConfidenceScore", "MeanImportance", "MeanSHAP"] if col in df.columns), None)
    if score_col is None:
        raise SystemExit("Recommended biomarkers must contain a score column for PPI seeding.")
    out = pd.DataFrame(
        {
            "Feature": df[feature_col].astype(str),
            "ImportanceValue": pd.to_numeric(df[score_col], errors="coerce").fillna(0.0),
            "Model": "RecommendedBiomarkers",
            "Task": task,
            "FeatureSource": "recommended_biomarkers",
            "Fold": "Full",
            "ClassName": "Global",
            "IsProteinLevel": True,
            "ImportanceType": "RecommendedBiomarkerScore",
            "Importance_Source": "recommended_biomarker_score",
            "FeatureRole": "BIOMARKER",
        }
    )
    filename = f"{prefix}_recommended_ppi_importance_values.csv"
    out.to_csv(workdir / filename, index=False)
    return "recommended_ppi_importance_values.csv"


def _build_adaptive_report(
    outdir: str,
    prefix: str,
    output_name: str,
    *,
    offline: bool = False,
) -> None:
    from engines.report.Step5_PDF_summary_analysis import build_report

    report_dir = task_output_dir(outdir, "pdf_report")
    output = report_dir / "Publication_Report.pdf"
    previous = Path.cwd()
    os.chdir(Path(outdir).resolve())
    try:
        build_report(prefix, str(output), offline=offline)
        ensure_publication_report_alias(output)
    finally:
        os.chdir(previous)


def _stage_source_input(input_csv: Path, outdir: Path) -> None:
    target = outdir / input_csv.name
    if target.exists():
        return
    try:
        target.symlink_to(input_csv)
    except Exception:
        shutil.copy2(input_csv, target)


if __name__ == "__main__":
    main()
