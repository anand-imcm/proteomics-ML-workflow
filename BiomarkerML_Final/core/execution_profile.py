from __future__ import annotations

import contextlib
import csv
import io
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

PROFILE_CHOICES = ("safe", "balanced", "aggressive", "expert")
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "CUDA_VISIBLE_DEVICES",
)
TARGET_COLUMNS = {"label", "target", "outcome", "class", "group", "y"}
ID_COLUMNS = {"sampleid", "sample_id", "id", "patientid", "patient_id", "subject", "subject_id"}
DEEP_MODEL_TOKENS = ("vae", "mlp", "nn", "neural")
_THREADPOOL_LIMITER = None


@dataclass(frozen=True)
class SystemResources:
    operating_system: str
    python_version: str
    cpu_model: str
    physical_cores: int
    logical_cores: int
    available_ram_gb: float | None
    gpu_available: bool
    cuda_available: bool
    gpu_name: str
    gpu_memory_gb: float | None
    threading_backend: str
    blas_backend: str
    disk_free_gb: float | None
    temp_dir: str
    threading_environment: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetProfile:
    input_path: str
    samples: int | None
    features: int | None
    task: str
    model_count: int
    models: tuple[str, ...]
    deep_learning_present: bool
    shap_enabled: bool
    ppi_enabled: bool
    benchmark_enabled: bool
    complexity_score: float | None
    complexity_label: str
    estimated_runtime: str
    estimated_memory_gb: float | None


@dataclass(frozen=True)
class ExecutionSettings:
    profile: str
    recommended_profile: str
    joblib_workers: int
    optuna_workers: int
    torch_threads: int
    torch_interop_threads: int
    dataloader_workers: int
    batch_size: int
    shap_background_size: int
    use_gpu: bool
    pinned_memory: bool
    cache_dir: str
    temp_dir: str
    environment: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionReport:
    system: SystemResources
    dataset: DatasetProfile
    settings: ExecutionSettings


def prepare_execution_profile(
    *,
    input_path: str | os.PathLike[str] | None,
    outdir: str | os.PathLike[str] | None,
    report_dir: str | os.PathLike[str] | None,
    task_name: str,
    profile: str = "balanced",
    requested_jobs: int | None = None,
    models: Iterable[str] | None = None,
    task_type: str | None = None,
    shap_enabled: bool = False,
    ppi_enabled: bool = False,
    benchmark_enabled: bool = False,
) -> ExecutionReport:
    selected = _normalize_profile(profile)
    base_outdir = Path(outdir or ".").resolve()
    base_outdir.mkdir(parents=True, exist_ok=True)

    system = detect_system_resources(base_outdir)
    dataset = analyze_dataset(
        input_path=input_path,
        models=tuple(models or ()),
        task_type=task_type or task_name,
        shap_enabled=shap_enabled,
        ppi_enabled=ppi_enabled,
        benchmark_enabled=benchmark_enabled,
    )
    recommended = recommend_profile(system, dataset)
    settings = build_execution_settings(
        selected,
        recommended,
        system,
        dataset,
        base_outdir=base_outdir,
        task_name=task_name,
        requested_jobs=requested_jobs,
    )
    apply_execution_settings(settings)

    report = ExecutionReport(system=system, dataset=dataset, settings=settings)
    print(format_resource_report(report))
    write_resource_report(report, Path(report_dir or base_outdir))
    return report


def detect_system_resources(outdir: str | os.PathLike[str] | None = None) -> SystemResources:
    base = Path(outdir or ".").resolve()
    logical = os.cpu_count() or 1
    physical = _physical_cpu_count(logical)
    available_ram = _available_ram_gb()
    gpu_available, cuda_available, gpu_name, gpu_memory = _detect_gpu()
    threading_env = {key: os.environ[key] for key in THREAD_ENV_VARS if key in os.environ}
    return SystemResources(
        operating_system=platform.platform(),
        python_version=platform.python_version(),
        cpu_model=_cpu_model(),
        physical_cores=physical,
        logical_cores=logical,
        available_ram_gb=available_ram,
        gpu_available=gpu_available,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory,
        threading_backend=_threading_backend(threading_env),
        blas_backend=_blas_backend(),
        disk_free_gb=_disk_free_gb(base),
        temp_dir=tempfile.gettempdir(),
        threading_environment=threading_env,
    )


def analyze_dataset(
    *,
    input_path: str | os.PathLike[str] | None,
    models: tuple[str, ...] = (),
    task_type: str = "unknown",
    shap_enabled: bool = False,
    ppi_enabled: bool = False,
    benchmark_enabled: bool = False,
) -> DatasetProfile:
    path = Path(input_path).resolve() if input_path else None
    csv_path = _find_dataset_csv(path)
    samples, features, inferred_task = _inspect_csv(csv_path)
    task = _normalize_task_type(task_type, inferred_task)
    model_names = tuple(str(model) for model in models)
    deep_learning = any(_is_deep_model(model) for model in model_names)
    score, label = _complexity(samples, features, max(1, len(model_names)), deep_learning, shap_enabled, ppi_enabled, benchmark_enabled)
    return DatasetProfile(
        input_path=str(path) if path else "",
        samples=samples,
        features=features,
        task=task,
        model_count=len(model_names),
        models=model_names,
        deep_learning_present=deep_learning,
        shap_enabled=shap_enabled,
        ppi_enabled=ppi_enabled,
        benchmark_enabled=benchmark_enabled,
        complexity_score=score,
        complexity_label=label,
        estimated_runtime=_runtime_label(label),
        estimated_memory_gb=_memory_estimate(samples, features, deep_learning, shap_enabled),
    )


def recommend_profile(system: SystemResources, dataset: DatasetProfile) -> str:
    ram = system.available_ram_gb or 0.0
    complexity = dataset.complexity_label
    if system.logical_cores <= 2 or (ram and ram < 8.0):
        return "safe"
    if complexity in {"Large", "Very Large"} and ram and ram < 16.0:
        return "safe"
    if system.physical_cores >= 12 and (ram >= 32.0 or ram == 0.0) and complexity in {"Medium", "Large", "Very Large"}:
        return "aggressive"
    return "balanced"


def build_execution_settings(
    profile: str,
    recommended_profile: str,
    system: SystemResources,
    dataset: DatasetProfile,
    *,
    base_outdir: Path,
    task_name: str,
    requested_jobs: int | None = None,
) -> ExecutionSettings:
    cache_dir = base_outdir / "cache" / task_name
    temp_dir = base_outdir / "temp" / task_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    workers = _worker_count(profile, system, requested_jobs)
    torch_threads = _torch_thread_count(profile, system)
    use_gpu = _use_gpu(profile, system, dataset)
    batch_size = _batch_size(profile, system, dataset)
    shap_background = _shap_background_size(profile, dataset)
    env = _profile_environment(profile, workers, torch_threads, cache_dir, temp_dir, use_gpu, batch_size, shap_background)

    return ExecutionSettings(
        profile=profile,
        recommended_profile=recommended_profile,
        joblib_workers=workers,
        optuna_workers=1 if profile == "safe" else max(1, min(workers, 2 if profile == "balanced" else workers)),
        torch_threads=torch_threads,
        torch_interop_threads=1 if profile == "safe" else max(1, min(2, torch_threads)),
        dataloader_workers=0 if profile in {"safe", "expert"} else max(1, min(4 if profile == "aggressive" else 2, workers)),
        batch_size=batch_size,
        shap_background_size=shap_background,
        use_gpu=use_gpu,
        pinned_memory=bool(use_gpu and profile in {"balanced", "aggressive"}),
        cache_dir=str(cache_dir),
        temp_dir=str(temp_dir),
        environment=env,
    )


def apply_execution_settings(settings: ExecutionSettings) -> None:
    os.environ["BIOMARKERML_PROFILE"] = settings.profile
    os.environ["BIOMARKERML_RECOMMENDED_PROFILE"] = settings.recommended_profile
    os.environ["BIOMARKERML_JOBLIB_WORKERS"] = str(settings.joblib_workers)
    os.environ["BIOMARKERML_OPTUNA_WORKERS"] = str(settings.optuna_workers)
    os.environ["BIOMARKERML_DATALOADER_WORKERS"] = str(settings.dataloader_workers)
    os.environ["BIOMARKERML_BATCH_SIZE"] = str(settings.batch_size)
    os.environ["BIOMARKERML_SHAP_BACKGROUND_SIZE"] = str(settings.shap_background_size)
    os.environ["BIOMARKERML_USE_GPU"] = "1" if settings.use_gpu else "0"
    os.environ["BIOMARKERML_CACHE_DIR"] = settings.cache_dir
    os.environ["BIOMARKERML_TEMP_DIR"] = settings.temp_dir

    if settings.profile != "expert":
        for key, value in settings.environment.items():
            os.environ[key] = value

    _configure_threadpool(settings.torch_threads, settings.profile)
    _configure_torch(settings)


def cleanup_execution_cache(settings: ExecutionSettings) -> None:
    temp_dir = Path(settings.temp_dir)
    if temp_dir.exists() and temp_dir.name and temp_dir.parent.name == "temp":
        shutil.rmtree(temp_dir, ignore_errors=True)


def format_resource_report(report: ExecutionReport) -> str:
    rows = _report_rows(report)
    width = 34
    lines = [
        "==================================",
        "BiomarkerML Resource Report",
        "==================================",
    ]
    for key, value in rows:
        lines.append(f"{key:<{width}} {value}")
    lines.append("==================================")
    return "\n".join(lines)


def write_resource_report(report: ExecutionReport, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = _report_rows(report)
    (report_dir / "resource_report.txt").write_text(format_resource_report(report) + "\n", encoding="utf-8")
    with (report_dir / "resource_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Field", "Value"])
        writer.writerows(rows)


def _report_rows(report: ExecutionReport) -> list[tuple[str, str]]:
    system = report.system
    dataset = report.dataset
    settings = report.settings
    return [
        ("Execution Profile", settings.profile),
        ("Recommended Profile", settings.recommended_profile),
        ("Operating System", system.operating_system),
        ("Python Version", system.python_version),
        ("CPU", system.cpu_model or "Unavailable"),
        ("Physical Cores", str(system.physical_cores)),
        ("Logical Cores", str(system.logical_cores)),
        ("RAM", _format_gb(system.available_ram_gb)),
        ("GPU", system.gpu_name if system.gpu_available else "Unavailable"),
        ("CUDA", "Available" if system.cuda_available else "Unavailable"),
        ("GPU Memory", _format_gb(system.gpu_memory_gb)),
        ("Threading Backend", system.threading_backend),
        ("BLAS Backend", system.blas_backend),
        ("Disk Free Space", _format_gb(system.disk_free_gb)),
        ("Temporary Directory", settings.temp_dir),
        ("Dataset Size", _format_count(dataset.samples, "samples")),
        ("Number of Features", _format_count(dataset.features, "features")),
        ("Task", dataset.task),
        ("Models", ", ".join(dataset.models) if dataset.models else "Not specified"),
        ("Deep Learning Present", "yes" if dataset.deep_learning_present else "no"),
        ("SHAP Enabled", "yes" if dataset.shap_enabled else "no"),
        ("PPI Enabled", "yes" if dataset.ppi_enabled else "no"),
        ("Benchmark Enabled", "yes" if dataset.benchmark_enabled else "no"),
        ("Estimated Runtime", dataset.estimated_runtime),
        ("Estimated Memory Usage", _format_gb(dataset.estimated_memory_gb)),
        ("Joblib Workers", str(settings.joblib_workers)),
        ("Optuna Workers", str(settings.optuna_workers)),
        ("PyTorch Threads", str(settings.torch_threads)),
        ("DataLoader Workers", str(settings.dataloader_workers)),
        ("Batch Size", str(settings.batch_size)),
        ("SHAP Background Size", str(settings.shap_background_size)),
        ("GPU Use", "enabled" if settings.use_gpu else "disabled"),
        ("Cache Directory", settings.cache_dir),
    ]


def _normalize_profile(profile: str | None) -> str:
    value = (profile or "balanced").strip().lower()
    if value not in PROFILE_CHOICES:
        raise ValueError(f"profile must be one of: {', '.join(PROFILE_CHOICES)}")
    return value


def _physical_cpu_count(logical: int) -> int:
    try:
        import psutil

        count = psutil.cpu_count(logical=False)
        if count:
            return int(count)
    except Exception:
        pass
    return max(1, logical // 2)


def _available_ram_gb() -> float | None:
    try:
        import psutil

        return psutil.virtual_memory().available / 1024**3
    except Exception:
        pass
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return float(pages * page_size) / 1024**3
    except Exception:
        pass
    if sys.platform == "darwin":
        return _macos_available_ram_gb()
    return None


def _cpu_model() -> str:
    if sys.platform == "darwin":
        try:
            model = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=1,
            ).strip()
            if model:
                return model
        except Exception:
            pass
    processor = platform.processor()
    if processor:
        return processor
    machine = platform.uname().machine
    if machine:
        return machine
    if Path("/proc/cpuinfo").exists():
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[-1].strip()
        except Exception:
            return ""
    return ""


def _macos_available_ram_gb() -> float | None:
    try:
        output = subprocess.check_output(["vm_stat"], stderr=subprocess.DEVNULL, text=True, timeout=1)
    except Exception:
        return None
    page_size = 4096
    pages = 0
    for line in output.splitlines():
        if "page size of" in line:
            try:
                page_size = int(line.split("page size of", 1)[1].split("bytes", 1)[0].strip())
            except Exception:
                page_size = 4096
            continue
        key, _, value = line.partition(":")
        if key.strip() in {"Pages free", "Pages inactive", "Pages speculative"}:
            try:
                pages += int(value.strip().strip(".").replace(",", ""))
            except Exception:
                pass
    return pages * page_size / 1024**3 if pages else None


def _detect_gpu() -> tuple[bool, bool, str, float | None]:
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        if not cuda:
            return False, False, "Unavailable", None
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        return True, True, props.name, props.total_memory / 1024**3
    except Exception:
        return False, False, "Unavailable", None


def _threading_backend(env: dict[str, str]) -> str:
    if env:
        return ", ".join(f"{key}={value}" for key, value in sorted(env.items()))
    return "Process default"


def _blas_backend() -> str:
    try:
        import numpy as np

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            np.__config__.show()
        text = buffer.getvalue().lower()
    except Exception:
        return "Unavailable"
    for name in ("mkl", "openblas", "accelerate", "blis", "atlas"):
        if name in text:
            return name
    return "Detected"


def _disk_free_gb(path: Path) -> float | None:
    try:
        return shutil.disk_usage(path).free / 1024**3
    except Exception:
        return None


def _find_dataset_csv(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.is_file() and path.suffix.lower() == ".csv":
        return path
    if not path.is_dir():
        return None
    candidates = sorted(path.rglob("*.csv"))
    for candidate in candidates:
        try:
            columns = [str(col).lower() for col in _read_csv_header(candidate)]
        except Exception:
            continue
        if TARGET_COLUMNS.intersection(columns):
            return candidate
    return candidates[0] if candidates else None


def _inspect_csv(path: Path | None) -> tuple[int | None, int | None, str]:
    if path is None or not path.exists():
        return None, None, "unknown"
    try:
        columns = _read_csv_header(path)
    except Exception:
        return None, None, "unknown"
    samples = _count_csv_rows(path)
    normalized = [str(col).strip().lower() for col in columns]
    feature_count = sum(1 for col in normalized if col not in TARGET_COLUMNS and col not in ID_COLUMNS)
    inferred = _infer_task_from_csv(path, columns)
    return samples, feature_count, inferred


def _read_csv_header(path: Path) -> list[str]:
    try:
        import pandas as pd

        return list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            return next(csv.reader(handle), [])


def _count_csv_rows(path: Path) -> int | None:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return max(0, sum(1 for _ in handle) - 1)
    except Exception:
        return None


def _infer_task_from_csv(path: Path, columns: list[str]) -> str:
    target = next((column for column in columns if str(column).strip().lower() in TARGET_COLUMNS), None)
    if target is None:
        return "unknown"
    try:
        import pandas as pd

        y = pd.read_csv(path, usecols=[target])[target].dropna()
        numeric = pd.to_numeric(y, errors="coerce")
        if numeric.notna().all() and y.nunique() > min(10, max(3, int(len(y) ** 0.5))):
            return "regression"
        return "classification"
    except Exception:
        return "unknown"


def _normalize_task_type(task_type: str, inferred: str) -> str:
    task = (task_type or "").lower()
    if "classification" in task or task in {"classify", "classification_shap"}:
        return "classification"
    if "regression" in task or task in {"regress", "regression_shap"}:
        return "regression"
    if inferred != "unknown":
        return inferred
    return task or "unknown"


def _is_deep_model(model: str) -> bool:
    text = str(model).lower()
    return any(token in text for token in DEEP_MODEL_TOKENS)


def _complexity(
    samples: int | None,
    features: int | None,
    model_count: int,
    deep_learning: bool,
    shap_enabled: bool,
    ppi_enabled: bool,
    benchmark_enabled: bool,
) -> tuple[float | None, str]:
    if not samples or not features:
        return None, "Unknown"
    score = float(samples) * float(features) * float(model_count)
    if deep_learning:
        score *= 1.8
    if shap_enabled:
        score *= 1.6
    if ppi_enabled:
        score *= 1.2
    if benchmark_enabled:
        score *= 2.0
    if score < 1e5:
        return score, "Very Small"
    if score < 1e6:
        return score, "Small"
    if score < 5e6:
        return score, "Medium"
    if score < 2e7:
        return score, "Large"
    return score, "Very Large"


def _runtime_label(complexity_label: str) -> str:
    return {
        "Very Small": "Very Small (<5 minutes)",
        "Small": "Small (5-20 minutes)",
        "Medium": "Medium (20-90 minutes)",
        "Large": "Large (1.5-6 hours)",
        "Very Large": "Very Large (>6 hours)",
    }.get(complexity_label, "Unknown")


def _memory_estimate(samples: int | None, features: int | None, deep_learning: bool, shap_enabled: bool) -> float | None:
    if not samples or not features:
        return None
    base = samples * features * 8 * 8 / 1024**3
    if deep_learning:
        base += 1.5
    if shap_enabled:
        base += 1.0
    return max(0.5, round(base, 2))


def _worker_count(profile: str, system: SystemResources, requested_jobs: int | None) -> int:
    if requested_jobs is not None and requested_jobs > 0:
        return int(requested_jobs)
    if profile in {"safe", "expert"}:
        return 1
    if profile == "balanced":
        return max(1, min(4, max(1, system.physical_cores // 2)))
    return max(1, min(system.logical_cores, max(1, system.physical_cores - 1)))


def _torch_thread_count(profile: str, system: SystemResources) -> int:
    if profile in {"safe", "expert"}:
        return 1
    if profile == "balanced":
        return max(1, min(4, max(1, system.physical_cores // 2)))
    return max(1, system.physical_cores)


def _use_gpu(profile: str, system: SystemResources, dataset: DatasetProfile) -> bool:
    explicit = os.environ.get("BIOMARKERML_USE_GPU", "").strip().lower()
    if explicit in {"1", "true", "yes", "on"}:
        return system.cuda_available
    if explicit in {"0", "false", "no", "off"}:
        return False
    if profile == "safe":
        return False
    if profile == "expert":
        return system.cuda_available and bool(explicit)
    return system.cuda_available and dataset.deep_learning_present


def _batch_size(profile: str, system: SystemResources, dataset: DatasetProfile) -> int:
    if profile in {"safe", "expert"}:
        return 32
    ram = system.available_ram_gb or 16.0
    if profile == "balanced":
        return 64 if ram >= 16.0 and dataset.deep_learning_present else 32
    return 128 if ram >= 32.0 and dataset.deep_learning_present else 64


def _shap_background_size(profile: str, dataset: DatasetProfile) -> int:
    if not dataset.shap_enabled:
        return 0
    if profile == "safe":
        return 25
    if profile == "aggressive":
        return 100
    return 50


def _profile_environment(
    profile: str,
    workers: int,
    torch_threads: int,
    cache_dir: Path,
    temp_dir: Path,
    use_gpu: bool,
    batch_size: int,
    shap_background: int,
) -> dict[str, str]:
    if profile == "expert":
        return {}
    threads = str(max(1, torch_threads))
    env = {
        "OMP_NUM_THREADS": threads,
        "MKL_NUM_THREADS": threads,
        "OPENBLAS_NUM_THREADS": threads,
        "NUMEXPR_NUM_THREADS": threads,
        "VECLIB_MAXIMUM_THREADS": threads,
        "LOKY_MAX_CPU_COUNT": str(max(1, workers)),
        "MPLCONFIGDIR": str(cache_dir / "matplotlib"),
        "XDG_CACHE_HOME": str(cache_dir / "xdg"),
        "TMPDIR": str(temp_dir),
        "BIOMARKERML_BATCH_SIZE": str(batch_size),
        "BIOMARKERML_SHAP_BACKGROUND_SIZE": str(shap_background),
    }
    if not use_gpu:
        env.setdefault("CUDA_VISIBLE_DEVICES", "")
    for path_key in ("MPLCONFIGDIR", "XDG_CACHE_HOME"):
        Path(env[path_key]).mkdir(parents=True, exist_ok=True)
    return env


def _configure_threadpool(threads: int, profile: str) -> None:
    if profile == "expert":
        return
    try:
        from threadpoolctl import threadpool_limits

        global _THREADPOOL_LIMITER
        _THREADPOOL_LIMITER = threadpool_limits(limits=max(1, int(threads)))
    except Exception:
        pass


def _configure_torch(settings: ExecutionSettings) -> None:
    if settings.profile == "expert":
        return
    try:
        import torch

        torch.set_num_threads(max(1, int(settings.torch_threads)))
        torch.set_num_interop_threads(max(1, int(settings.torch_interop_threads)))
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    except Exception:
        pass


def _format_gb(value: float | None) -> str:
    if value is None:
        return "Unknown"
    return f"{value:.2f} GB"


def _format_count(value: int | None, unit: str) -> str:
    if value is None:
        return f"Unknown {unit}"
    return f"{value} {unit}"
