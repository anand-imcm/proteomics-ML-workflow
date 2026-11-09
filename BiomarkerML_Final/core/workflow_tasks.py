from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from joblib._parallel_backends import LokyBackend

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for path in [
    ROOT,
    SCRIPT_DIR,
    ROOT / "engines",
    ROOT / "engines" / "classification",
    ROOT / "engines" / "regression",
    ROOT / "engines" / "deep_learning",
    ROOT / "engines" / "reliability",
]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from .publication_utils import (
        add_common_cli_arguments,
        build_output_prefix,
        ensure_publication_report_alias,
        set_global_seed,
        stage_task_inputs,
        task_output_dir,
        task_run_context,
        write_artifact_alias,
    )
    from .execution_profile import cleanup_execution_cache, prepare_execution_profile
except ImportError:
    from publication_utils import (
        add_common_cli_arguments,
        build_output_prefix,
        ensure_publication_report_alias,
        set_global_seed,
        stage_task_inputs,
        task_output_dir,
        task_run_context,
        write_artifact_alias,
    )
    from execution_profile import cleanup_execution_cache, prepare_execution_profile


class _WorkflowLokyBackend(LokyBackend):
    def configure(self, n_jobs=1, parallel=None, prefer=None, require=None, **kwargs):
        kwargs["idle_worker_timeout"] = None
        # Loky workers start fresh Python interpreters, so the model-worker warning
        # filter must be installed by the executor rather than only by its parent.
        kwargs.setdefault("initializer", _install_sklearn_parallel_delayed_warning_filter)
        return super().configure(
            n_jobs=n_jobs,
            parallel=parallel,
            prefer=prefer,
            require=require,
            **kwargs,
        )

SKLEARN_PARALLEL_DELAYED_WARNING = (
    r".*(?:"
    r"`sklearn\.utils\.parallel\.Parallel` needs to be used in conjunction with "
    r"`sklearn\.utils\.parallel\.delayed` instead of `joblib\.delayed`"
    r"|"
    r"`sklearn\.utils\.parallel\.delayed` should be used with "
    r"`sklearn\.utils\.parallel\.Parallel` to make it possible to propagate the "
    r"scikit-learn configuration of the current thread to the joblib workers"
    r").*"
)

SKLEARN_PARALLEL_DELAYED_PYTHONWARNINGS = ",".join(
    (
        "ignore:`sklearn.utils.parallel.Parallel` needs to be used in conjunction with "
        "`sklearn.utils.parallel.delayed` instead of `joblib.delayed`:UserWarning",
        "ignore:`sklearn.utils.parallel.delayed` should be used with "
        "`sklearn.utils.parallel.Parallel` to make it possible to propagate the "
        "scikit-learn configuration of the current thread to the joblib workers:UserWarning",
    )
)


def _install_sklearn_parallel_delayed_warning_filter() -> None:
    """Install the exact sklearn nested-parallel warning filter in each loky worker."""
    warnings.filterwarnings(
        "ignore",
        message=SKLEARN_PARALLEL_DELAYED_WARNING,
        category=UserWarning,
    )


@contextmanager
def _suppress_nested_sklearn_parallel_delayed_warning():
    previous = os.environ.get("PYTHONWARNINGS")
    os.environ["PYTHONWARNINGS"] = (
        SKLEARN_PARALLEL_DELAYED_PYTHONWARNINGS
        if not previous
        else f"{previous},{SKLEARN_PARALLEL_DELAYED_PYTHONWARNINGS}"
    )
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=SKLEARN_PARALLEL_DELAYED_WARNING,
                category=UserWarning,
            )
            yield
    finally:
        if previous is None:
            os.environ.pop("PYTHONWARNINGS", None)
        else:
            os.environ["PYTHONWARNINGS"] = previous


def common_parser(description: str, *, require_input: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    add_common_cli_arguments(parser, require_input=require_input)
    return parser


@contextmanager
def task_execution_profile(
    args,
    task: str,
    *,
    models: list[str] | None = None,
    task_type: str | None = None,
    shap_enabled: bool = False,
    ppi_enabled: bool = False,
    benchmark_enabled: bool = False,
):
    report_dir = task_output_dir(args.outdir, "resource_report")
    report = prepare_execution_profile(
        input_path=getattr(args, "input", None),
        outdir=getattr(args, "outdir", "."),
        report_dir=report_dir,
        task_name=task,
        profile=getattr(args, "profile", "balanced"),
        requested_jobs=getattr(args, "jobs", None),
        models=models or [],
        task_type=task_type or task,
        shap_enabled=shap_enabled,
        ppi_enabled=ppi_enabled,
        benchmark_enabled=benchmark_enabled,
    )
    args.jobs = report.settings.joblib_workers
    completed = False
    try:
        yield report
        completed = True
    finally:
        if completed:
            cleanup_execution_cache(report.settings)


def run_preprocess(args) -> None:
    prefix = build_output_prefix(args.input, args.prefix)
    methods = args.methods or ["pca", "kpca", "umap", "tsne", "pls"]
    with task_execution_profile(args, "preprocess", models=methods, task_type="preprocess"):
        set_global_seed(args.seed)
        env = dict(os.environ)
        pythonpath = ":".join([str(ROOT), str(SCRIPT_DIR)])
        env["PYTHONPATH"] = pythonpath + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "engines" / "preprocessing" / "Step1_Preprocessing.py"),
                "--input",
                str(Path(args.input).resolve()),
                "--prefix",
                prefix,
                "--outdir",
                str(Path(args.outdir).resolve()),
                "--jobs",
                str(args.jobs),
                "--methods",
                *methods,
                "--dimensions",
                str(args.dimensions),
                "--seed",
                str(args.seed),
            ],
            env=env,
        )


def run_classification(args) -> None:
    from sklearn.utils.parallel import Parallel, delayed
    from engines.classification import run_model

    input_path = Path(args.input).resolve()
    prefix = build_output_prefix(input_path, args.prefix)
    models = args.models or ["KNN", "RF", "NN", "SVM", "XGB", "PLSDA", "VAE", "LGBM", "LR", "MLPVAE", "GNB"]
    with task_execution_profile(args, "classification", models=models, task_type="classification"):
        set_global_seed(args.seed)
        with task_run_context(args.outdir, "classification") as outdir:
            print(f"Classification models queued: {', '.join(models)}")
            # sklearn.delayed snapshots configuration and warning filters while
            # these tasks are constructed. Enter the exact filter boundary before
            # worker creation so its wrappers carry that state into loky workers.
            with _suppress_nested_sklearn_parallel_delayed_warning():
                Parallel(n_jobs=args.jobs, verbose=0, backend=_WorkflowLokyBackend())(
                    delayed(_run_logged_model)(
                        run_model,
                        model,
                        str(input_path),
                        prefix,
                        args.feature_selection,
                        len(models),
                        index,
                        getattr(args, "imputation", "median"),
                        getattr(args, "covariates", None),
                        getattr(args, "categorical_covariates", None),
                    )
                    for index, model in enumerate(models, start=1)
                )
            _organize_model_outputs(prefix, outdir, models, _classification_model_keys(models), "classification")


def run_regression(args) -> None:
    from sklearn.utils.parallel import Parallel, delayed
    from engines.regression import run_model

    input_path = Path(args.input).resolve()
    prefix = build_output_prefix(input_path, args.prefix)
    models = args.models or ["NN_reg", "RF_reg", "SVM_reg", "XGB_reg", "PLS_reg", "KNN_reg", "LGBM_reg", "VAE_reg", "MLPVAE_reg"]
    with task_execution_profile(args, "regression", models=models, task_type="regression"):
        set_global_seed(args.seed)
        with task_run_context(args.outdir, "regression") as outdir:
            print(f"Regression models queued: {', '.join(models)}")
            with _suppress_nested_sklearn_parallel_delayed_warning():
                Parallel(n_jobs=args.jobs, verbose=0, backend=_WorkflowLokyBackend())(
                    delayed(_run_logged_model)(
                        run_model,
                        model,
                        str(input_path),
                        prefix,
                        args.feature_selection,
                        len(models),
                        index,
                        getattr(args, "imputation", "median"),
                        getattr(args, "covariates", None),
                        getattr(args, "categorical_covariates", None),
                    )
                    for index, model in enumerate(models, start=1)
                )
            _organize_model_outputs(prefix, outdir, models, _regression_model_keys(models), "regression")


def run_classification_shap(args) -> None:
    from engines.classification.Step4_Classification_SHAP import _resolve_model_name, process_model

    prefix = build_output_prefix(args.input, args.prefix)
    models = args.models or ["KNN", "RF", "NN", "SVM", "XGB", "PLSDA", "VAE", "LGBM", "LR", "MLPVAE", "GNB"]
    with task_execution_profile(args, "classification_shap", models=models, task_type="classification", shap_enabled=True):
        set_global_seed(args.seed)
        outdir = task_output_dir(args.outdir, "classification_shap")
        stage_task_inputs(args.input, outdir, prefix)
        with task_run_context(args.outdir, "classification_shap"):
            total = len(models)
            for index, model in enumerate(models, start=1):
                print(f"SHAP model: {model} ({index}/{total})")
                process_model(
                    prefix,
                    model,
                    args.num_features,
                    species=getattr(args, "species", 9606),
                    offline=getattr(args, "offline", False),
                )
            _organize_model_outputs(prefix, outdir, models, [_resolve_model_name(m) for m in models], "classification_shap")


def run_regression_shap(args) -> None:
    from engines.regression.Step3_Regression_SHAP import _resolve_model_name, calculate_shap_values

    prefix = build_output_prefix(args.input, args.prefix)
    models = args.models or ["NN_reg", "RF_reg", "SVM_reg", "XGB_reg", "PLS_reg", "KNN_reg", "LGBM_reg", "VAE_reg", "MLPVAE_reg"]
    with task_execution_profile(args, "regression_shap", models=models, task_type="regression", shap_enabled=True):
        set_global_seed(args.seed)
        outdir = task_output_dir(args.outdir, "regression_shap")
        stage_task_inputs(args.input, outdir, prefix)
        with task_run_context(args.outdir, "regression_shap"):
            total = len(models)
            for index, model in enumerate(models, start=1):
                print(f"SHAP model: {model} ({index}/{total})")
                calculate_shap_values(
                    model,
                    args.num_features,
                    prefix,
                    species=getattr(args, "species", 9606),
                    offline=getattr(args, "offline", False),
                )
            _organize_model_outputs(prefix, outdir, models, [_resolve_model_name(m) for m in models], "regression_shap")


def run_biomarker_reliability(args) -> None:
    from engines.reliability.biomarker_reliability import run_publication_biomarker_analysis

    prefix = build_output_prefix(args.input, args.prefix)
    with task_execution_profile(args, "biomarker_reliability", task_type=args.task):
        set_global_seed(args.seed)
        outdir = task_output_dir(args.outdir, "biomarker_reliability")
        stage_task_inputs(args.input, outdir, prefix)
        with task_run_context(args.outdir, "biomarker_reliability") as workdir:
            run_publication_biomarker_analysis(prefix, task=args.task, input_csv=args.dataset)
            _alias_reliability_outputs(prefix, workdir)



def _resolve_recommended_ppi_input(source: Path, prefix: str) -> Path | None:
    # Explicit precedence avoids recursive basename collisions.
    candidates = [
        source / "biomarker_reliability" / f"{prefix}_recommended_biomarkers.csv",
        source / f"{prefix}_recommended_biomarkers.csv",
        source / "biomarker_reliability" / "recommended_biomarkers.csv",
        source / "recommended_biomarkers.csv",
    ]
    return next((path for path in candidates if path.is_file()), None)


def _stage_recommended_ppi_table(source: Path, outdir: Path, prefix: str) -> Path:
    table = pd.read_csv(source)
    if "FeatureRole" in table.columns:
        table = table.loc[table["FeatureRole"].astype(str).str.upper().eq("BIOMARKER")].copy()
    if table.empty:
        raise SystemExit("PPI requires at least one recommended biomarker.")
    canonical = outdir / f"{prefix}_recommended_biomarkers.csv"
    table.to_csv(canonical, index=False)
    table.to_csv(outdir / "recommended_biomarkers.csv", index=False)
    return canonical

def run_ppi(args) -> None:
    from engines.ppi.protein_network import run_network

    prefix = build_output_prefix(args.input, args.prefix)
    with task_execution_profile(args, "ppi", task_type="ppi", ppi_enabled=True):
        set_global_seed(args.seed)
        outdir = task_output_dir(args.outdir, "ppi")
        source = Path(args.input).resolve()
        if source.is_dir():
            recommended = _resolve_recommended_ppi_input(source, prefix)
            if recommended is None:
                raise SystemExit("Explicit recommended-biomarker input was not found for PPI.")
            _stage_recommended_ppi_table(recommended, outdir, prefix)
        elif source.is_file():
            _stage_recommended_ppi_table(source, outdir, prefix)
        expression = Path(args.protein_expression).resolve() if args.protein_expression else source
        with task_run_context(args.outdir, "ppi") as workdir:
            ns = SimpleNamespace(
                score_thresholdHere=args.score_threshold,
                combined_score_thresholdHere=args.combined_score_threshold,
                SHAPthresh=args.top_n,
                patternChosen="recommended_biomarkers.csv",
                converProId="TRUE",
                proteinExpFile=str(expression),
                CorMethod=args.correlation_method,
                CorThreshold=args.correlation_threshold,
                prefix=prefix,
                species=args.species,
                offline=args.offline,
                expanded_limit=args.expanded_limit,
            )
            run_network(ns)
            _alias_ppi_outputs(prefix, workdir)


def run_pdf_report(args) -> None:
    from engines.report.Step5_PDF_summary_analysis import build_report

    prefix = build_output_prefix(args.input, args.prefix)
    with task_execution_profile(args, "pdf_report", task_type="report"):
        set_global_seed(args.seed)
        outdir = task_output_dir(args.outdir, "pdf_report")
        search_root = Path(args.input).resolve() if args.input and Path(args.input).is_dir() else Path(args.outdir).resolve()
        previous = Path.cwd()
        os.chdir(search_root)
        try:
            canonical_report = outdir / "Publication_Report.pdf"
            build_report(
                prefix,
                str(canonical_report),
                offline=getattr(args, "offline", False),
            )
            ensure_publication_report_alias(canonical_report)
        finally:
            os.chdir(previous)


def _format_seconds(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _run_logged_model(
    runner,
    model: str,
    input_path: str,
    prefix: str,
    feature_selection: str,
    total: int,
    index: int,
    imputation: str = "median",
    covariates=None,
    categorical_covariates=None,
) -> None:
    start = time.perf_counter()
    print("\n" + "=" * 56)
    print("BiomarkerML Model Progress")
    print("=" * 56)
    print(f"Current model: {model} ({index}/{total})")
    print(f"Completed models: {index - 1}")
    print(f"Remaining models: {total - index + 1}")
    print("=" * 56)
    with _suppress_nested_sklearn_parallel_delayed_warning():
        runner(model, input_path, prefix, feature_selection, imputation, covariates, categorical_covariates)
    elapsed = _format_seconds(time.perf_counter() - start)
    print(f"Completed model: {model} ({index}/{total}) in {elapsed}")
    print(f"Remaining models: {total - index}")


def _organize_model_outputs(prefix: str, outdir: Path, requested_models: list[str], model_keys: list[str], task: str) -> None:
    for requested, key in zip(requested_models, model_keys):
        model_dir = outdir / str(requested)
        model_dir.mkdir(parents=True, exist_ok=True)
        patterns = {f"{prefix}_{key}_*", f"{task}_{key}_*", f"{task}_{key.lower()}_*"}
        if task.endswith("_shap"):
            base_task = task.removesuffix("_shap")
            patterns.update({f"{base_task}_{key}_*", f"{base_task}_{key.lower()}_*"})
        for pattern in sorted(patterns):
            for path in sorted(outdir.glob(pattern)):
                if path.is_file():
                    target = model_dir / path.name
                    if target.exists():
                        target.unlink()
                    shutil.move(str(path), str(target))


def _alias_reliability_outputs(prefix: str, outdir: Path) -> None:
    for suffix in [
        "stability_matrix.csv",
        "biomarker_frequency.csv",
        "biomarker_stability_summary.csv",
        "consensus_biomarkers.csv",
        "consensus_biomarker_summary.csv",
        "recommended_biomarkers.csv",
        "publication_summary.xlsx",
        "publication_summary.csv",
        "stability_heatmap.png",
        "jaccard_matrix.png",
        "rank_correlation_matrix.png",
        "biomarker_frequency_plot.png",
        "consensus_biomarker_ranking.png",
        "consensus_frequency_plot.png",
        "recommended_biomarkers.png",
    ]:
        write_artifact_alias(outdir / f"{prefix}_{suffix}", outdir / suffix)


def _alias_ppi_outputs(prefix: str, outdir: Path) -> None:
    for suffix in [
        "ppi_network.png",
        "ppi_expanded_network.png",
        "ppi_network_nodes.csv",
        "ppi_network_edges.csv",
        "ppi_expanded_network_nodes.csv",
        "ppi_expanded_network_edges.csv",
        "hub_proteins.csv",
        "ppi_report.csv",
        "ppi_report.md",
    ]:
        write_artifact_alias(outdir / f"{prefix}_{suffix}", outdir / suffix)


def _classification_model_keys(models: list[str]) -> list[str]:
    aliases = {
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
    return [aliases.get(model, model) for model in models]


def _regression_model_keys(models: list[str]) -> list[str]:
    aliases = {
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
    return [aliases.get(model, model) for model in models]
