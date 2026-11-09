from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from .publication_utils import DEFAULT_RANDOM_STATE, configure_publication_plots
    from .version import BIOMARKERML_VERSION
except ImportError:
    from publication_utils import DEFAULT_RANDOM_STATE, configure_publication_plots
    from version import BIOMARKERML_VERSION

MISSINGNESS_THRESHOLD = 0.50
MISSING_CATEGORY = "__MISSING__"


def preserve_dataframe_output(transformer):
    """Request sklearn's public pandas-output contract when it is available."""
    set_output = getattr(transformer, "set_output", None)
    return set_output(transform="pandas") if callable(set_output) else transformer


@dataclass(frozen=True)
class FeatureRoles:
    identifier: str | None
    target: str
    biomarkers: tuple[str, ...]
    covariates: tuple[str, ...]
    numeric_covariates: tuple[str, ...]
    categorical_covariates: tuple[str, ...]

    def role_for(self, feature: str) -> str:
        if feature == self.identifier:
            return "IDENTIFIER"
        if feature == self.target:
            return "TARGET"
        if feature in self.covariates:
            return "COVARIATE"
        if feature in self.biomarkers:
            return "BIOMARKER"
        raise KeyError(feature)


def resolve_feature_roles(
    data: pd.DataFrame,
    *,
    target: str = "Label",
    identifier: str | None = "SampleID",
    covariates: Sequence[str] | None = None,
    categorical_covariates: Sequence[str] | None = None,
) -> FeatureRoles:
    requested = [str(value) for value in (covariates or [])]
    categorical_requested = [str(value) for value in (categorical_covariates or [])]
    if len(requested) != len(set(requested)):
        raise ValueError("Covariate names must not be duplicated.")
    missing = sorted(set(requested + categorical_requested) - set(data.columns))
    if missing:
        raise ValueError(f"Requested covariate columns were not found: {', '.join(missing)}")
    if target in requested:
        raise ValueError("The target column cannot also be a covariate.")
    if identifier and identifier in requested:
        raise ValueError("The identifier column cannot also be a covariate.")
    if not set(categorical_requested).issubset(requested):
        raise ValueError("Every --categorical-covariates value must also appear in --covariates.")

    numeric = []
    categorical = []
    for name in requested:
        if name in categorical_requested or not pd.api.types.is_numeric_dtype(data[name]):
            categorical.append(name)
        else:
            numeric.append(name)
    excluded = {target, *requested}
    if identifier and identifier in data.columns:
        excluded.add(identifier)
    biomarkers = [str(column) for column in data.columns if column not in excluded]
    if not biomarkers:
        raise ValueError("No biomarker columns remain after assigning feature roles.")
    return FeatureRoles(
        identifier=identifier if identifier in data.columns else None,
        target=target,
        biomarkers=tuple(biomarkers),
        covariates=tuple(requested),
        numeric_covariates=tuple(numeric),
        categorical_covariates=tuple(categorical),
    )


def modeling_frame(data: pd.DataFrame, roles: FeatureRoles) -> pd.DataFrame:
    frame = data.loc[:, list(roles.biomarkers + roles.covariates)].copy()
    for column in roles.biomarkers + roles.numeric_covariates:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in roles.categorical_covariates:
        frame[column] = frame[column].astype("object")
    return frame


class FoldLocalMissingnessFilter(BaseEstimator, TransformerMixin):
    """Retain biomarkers missing in at most half of training-fold subjects."""

    def __init__(self, threshold: float = MISSINGNESS_THRESHOLD):
        self.threshold = threshold

    def fit(self, X, y=None):
        frame = self._as_frame(X, fitting=True)
        self.feature_names_in_ = np.asarray([str(column) for column in frame.columns], dtype=object)
        self.missing_fraction_ = frame.isna().mean(axis=0).astype(float)
        self.missing_fraction_.index = self.feature_names_in_
        self.available_count_ = pd.Series(len(frame), index=self.feature_names_in_, dtype=int)
        self.retained_features_ = [name for name in self.feature_names_in_ if self.missing_fraction_[name] <= self.threshold]
        self.removed_features_ = [name for name in self.feature_names_in_ if name not in self.retained_features_]
        if not self.retained_features_:
            raise ValueError("No biomarkers remain after fold-local missingness filtering.")
        return self

    def transform(self, X):
        if not hasattr(self, "retained_features_"):
            raise ValueError("FoldLocalMissingnessFilter is not fitted.")
        frame = self._as_frame(X, fitting=False)
        missing = [name for name in self.feature_names_in_ if name not in frame.columns]
        extra = [str(name) for name in frame.columns if str(name) not in set(self.feature_names_in_)]
        if missing or extra:
            raise ValueError(
                "Biomarker schema differs from the fitted training schema "
                f"(missing={missing[:5]}, extra={extra[:5]})."
            )
        return frame.loc[:, self.retained_features_].copy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.retained_features_, dtype=object)

    def _as_frame(self, X, *, fitting: bool) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            frame = X.copy()
            frame.columns = [str(column) for column in frame.columns]
            return frame
        array = np.asarray(X)
        if array.ndim != 2:
            raise ValueError("Biomarker matrix must be two-dimensional.")
        if fitting:
            columns = [f"Feature_{index + 1}" for index in range(array.shape[1])]
        else:
            columns = list(self.feature_names_in_)
        return pd.DataFrame(array, columns=columns)


class FoldLocalImputer(BaseEstimator, TransformerMixin):
    """Median or scaled distance-weighted KNN imputation fitted on training data."""

    def __init__(self, method: str = "median", n_neighbors: int = 5):
        self.method = method
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        frame = self._frame(X, fitting=True)
        if self.method not in {"median", "knn"}:
            raise ValueError("imputation must be 'median' or 'knn'.")
        self.feature_names_in_ = np.asarray(frame.columns.astype(str), dtype=object)
        self.medians_ = frame.median(axis=0, skipna=True)
        if self.medians_.isna().any():
            names = self.medians_.index[self.medians_.isna()].tolist()
            raise ValueError(f"Cannot impute all-missing retained biomarkers: {names[:5]}")
        self.fallback_used_ = False
        self.fallback_reason_ = ""
        if self.method == "knn":
            self.center_ = frame.mean(axis=0, skipna=True)
            self.scale_ = frame.std(axis=0, skipna=True, ddof=0).replace(0.0, 1.0).fillna(1.0)
            standardized = (frame - self.center_) / self.scale_
            try:
                neighbors = min(max(1, int(self.n_neighbors)), max(1, len(frame) - 1))
                self.knn_ = KNNImputer(n_neighbors=neighbors, weights="distance")
                transformed = self.knn_.fit_transform(standardized)
                if transformed.shape[1] != frame.shape[1] or not np.isfinite(transformed).all():
                    raise ValueError("KNN imputer did not return a finite matrix with the fitted schema.")
            except Exception as exc:
                self.fallback_used_ = True
                self.fallback_reason_ = str(exc)
                warnings.warn(
                    "Fold-local distance-weighted KNN imputation could not operate; "
                    f"using fold-local medians for this fit ({exc}).",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return self

    def transform(self, X):
        frame = self._frame(X, fitting=False)
        if self.method == "median" or self.fallback_used_:
            result = frame.fillna(self.medians_)
        else:
            try:
                standardized = (frame - self.center_) / self.scale_
                result = pd.DataFrame(
                    self.knn_.transform(standardized),
                    columns=self.feature_names_in_,
                    index=frame.index,
                )
                result = result * self.scale_ + self.center_
            except Exception as exc:
                self.fallback_used_ = True
                self.fallback_reason_ = str(exc)
                warnings.warn(
                    "Fold-local distance-weighted KNN transform failed; using fitted "
                    f"training medians for this fold ({exc}).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                result = frame.fillna(self.medians_)
        if result.isna().any().any():
            raise ValueError("Fold-local imputation left missing biomarker values.")
        return result

    def _frame(self, X, *, fitting: bool) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            frame = X.copy()
            frame.columns = [str(column) for column in frame.columns]
        else:
            array = np.asarray(X)
            columns = [f"Feature_{i + 1}" for i in range(array.shape[1])] if fitting else list(self.feature_names_in_)
            frame = pd.DataFrame(array, columns=columns)
        if not fitting and list(frame.columns) != list(self.feature_names_in_):
            raise ValueError("Imputation input schema differs from the fitted training schema.")
        return frame.apply(pd.to_numeric, errors="coerce")


class RoleAwarePreprocessor(BaseEstimator, TransformerMixin):
    """Fold-local biomarker preprocessing plus covariates that bypass selection."""

    def __init__(
        self,
        biomarker_columns: Sequence[str],
        numeric_covariates: Sequence[str] = (),
        categorical_covariates: Sequence[str] = (),
        imputation: str = "median",
        selector=None,
        missingness_threshold: float = MISSINGNESS_THRESHOLD,
    ):
        self.biomarker_columns = biomarker_columns
        self.numeric_covariates = numeric_covariates
        self.categorical_covariates = categorical_covariates
        self.imputation = imputation
        self.selector = selector
        self.missingness_threshold = missingness_threshold

    def fit(self, X, y=None):
        frame = self._validate_frame(X, fitting=True)
        biomarkers = frame.loc[:, list(self.biomarker_columns)].apply(pd.to_numeric, errors="coerce")
        self.missingness_filter_ = FoldLocalMissingnessFilter(
            threshold=self.missingness_threshold,
        ).fit(biomarkers)
        retained = self.missingness_filter_.transform(biomarkers)
        self.biomarker_imputer_ = FoldLocalImputer(self.imputation).fit(retained)
        imputed = self.biomarker_imputer_.transform(retained)
        self.biomarker_scaler_ = StandardScaler().fit(imputed)
        scaled = pd.DataFrame(
            self.biomarker_scaler_.transform(imputed), columns=imputed.columns, index=imputed.index
        )
        self.selector_ = clone(self.selector) if self.selector is not None else None
        if self.selector_ is not None:
            self.selector_.fit(scaled, y)
            support = np.asarray(self.selector_.get_support(), dtype=bool)
            if support.size != scaled.shape[1] or not support.any():
                raise ValueError("Biomarker feature selection produced an invalid or empty feature set.")
            self.selected_biomarker_names_ = list(np.asarray(scaled.columns, dtype=object)[support])
        else:
            self.selected_biomarker_names_ = list(scaled.columns)

        self.numeric_imputer_ = None
        self.numeric_scaler_ = None
        if self.numeric_covariates:
            numeric = frame.loc[:, list(self.numeric_covariates)].apply(pd.to_numeric, errors="coerce")
            if numeric.isna().all(axis=0).any():
                names = numeric.columns[numeric.isna().all(axis=0)].tolist()
                raise ValueError(f"Numeric covariates are all missing in the training fold: {names}")
            self.numeric_imputer_ = SimpleImputer(strategy="median").fit(numeric)
            numeric_imputed = self.numeric_imputer_.transform(numeric)
            self.numeric_scaler_ = StandardScaler().fit(numeric_imputed)

        self.categorical_imputer_ = None
        self.categorical_encoder_ = None
        self.encoded_categorical_names_ = []
        if self.categorical_covariates:
            categorical = frame.loc[:, list(self.categorical_covariates)].astype("object")
            categorical = categorical.where(pd.notna(categorical), np.nan)
            self.categorical_imputer_ = SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY).fit(categorical)
            categorical_filled = self.categorical_imputer_.transform(categorical).astype(str)
            kwargs = {"handle_unknown": "ignore"}
            try:
                self.categorical_encoder_ = OneHotEncoder(sparse_output=False, **kwargs)
            except TypeError:
                self.categorical_encoder_ = OneHotEncoder(sparse=False, **kwargs)
            encoded = self.categorical_encoder_.fit(categorical_filled)
            self.encoded_categorical_names_ = [str(name) for name in encoded.get_feature_names_out(list(self.categorical_covariates))]
        self.covariate_feature_names_ = list(self.numeric_covariates) + self.encoded_categorical_names_
        self.feature_names_out_ = self.selected_biomarker_names_ + self.covariate_feature_names_
        self.feature_roles_out_ = {
            **{name: "BIOMARKER" for name in self.selected_biomarker_names_},
            **{name: "COVARIATE" for name in self.covariate_feature_names_},
        }
        return self

    def transform(self, X):
        frame = self._validate_frame(X, fitting=False)
        biomarkers = frame.loc[:, list(self.biomarker_columns)].apply(pd.to_numeric, errors="coerce")
        retained = self.missingness_filter_.transform(biomarkers)
        imputed = self.biomarker_imputer_.transform(retained)
        scaled = pd.DataFrame(
            self.biomarker_scaler_.transform(imputed), columns=imputed.columns, index=frame.index
        )
        if self.selector_ is not None:
            selected_values = self.selector_.transform(scaled)
            selected = pd.DataFrame(selected_values, columns=self.selected_biomarker_names_, index=frame.index)
        else:
            selected = scaled.loc[:, self.selected_biomarker_names_]
        pieces = [selected]
        if self.numeric_covariates:
            numeric = frame.loc[:, list(self.numeric_covariates)].apply(pd.to_numeric, errors="coerce")
            values = self.numeric_scaler_.transform(self.numeric_imputer_.transform(numeric))
            pieces.append(pd.DataFrame(values, columns=list(self.numeric_covariates), index=frame.index))
        if self.categorical_covariates:
            categorical = frame.loc[:, list(self.categorical_covariates)].astype("object")
            values = self.categorical_encoder_.transform(self.categorical_imputer_.transform(categorical).astype(str))
            pieces.append(pd.DataFrame(values, columns=self.encoded_categorical_names_, index=frame.index))
        output = pd.concat(pieces, axis=1)
        return output.loc[:, self.feature_names_out_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)

    def biomarker_diagnostics(self, fold: int) -> pd.DataFrame:
        filt = self.missingness_filter_
        return pd.DataFrame(
            {
                "Feature": filt.feature_names_in_,
                "Role": "BIOMARKER",
                "Fold": int(fold),
                "TrainingMissingRate": [float(filt.missing_fraction_[name]) for name in filt.feature_names_in_],
                "AvailableTrainingSamples": [int(filt.available_count_[name]) for name in filt.feature_names_in_],
                "Retained": [name in set(filt.retained_features_) for name in filt.feature_names_in_],
                "ImputationMethodRequested": self.imputation,
                "ImputationMethodUsed": "median" if self.biomarker_imputer_.fallback_used_ else self.imputation,
                "ImputationFallbackReason": self.biomarker_imputer_.fallback_reason_,
            }
        )

    def _validate_frame(self, X, *, fitting: bool) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("RoleAwarePreprocessor requires a pandas DataFrame with named features.")
        frame = X.copy()
        frame.columns = [str(column) for column in frame.columns]
        expected = list(self.biomarker_columns) + list(self.numeric_covariates) + list(self.categorical_covariates)
        missing = [name for name in expected if name not in frame.columns]
        if missing:
            raise ValueError(f"Model input is missing fitted feature columns: {missing[:5]}")
        if fitting:
            self.feature_names_in_ = np.asarray(expected, dtype=object)
        return frame.loc[:, expected]


def selector_for_method(method: str, *, task: str):
    try:
        from .publication_utils import BorutaProteinSelector, make_elasticnet_selector
    except ImportError:
        from publication_utils import BorutaProteinSelector, make_elasticnet_selector

    if method == "boruta":
        return BorutaProteinSelector(task=task, random_state=DEFAULT_RANDOM_STATE, max_iter=60, n_estimators=400)
    if method == "elasticnet":
        return make_elasticnet_selector(task=task)
    return None


def write_missingness_qc(
    prefix: str,
    X: pd.DataFrame,
    roles: FeatureRoles,
    fold_tables: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    biomarkers = X.loc[:, list(roles.biomarkers)]
    overall = pd.DataFrame(
        {
            "Feature": biomarkers.columns.astype(str),
            "Role": "BIOMARKER",
            "OverallMissingRate": biomarkers.isna().mean().to_numpy(dtype=float),
            "OverallAvailableSamples": int(len(biomarkers)),
        }
    )
    folds = pd.concat(list(fold_tables), ignore_index=True) if fold_tables else pd.DataFrame()
    if not folds.empty:
        qc = folds.merge(overall, on=["Feature", "Role"], how="left")
        frequency = folds.groupby("Feature", as_index=False)["Retained"].mean().rename(columns={"Retained": "RetentionFrequency"})
        qc = qc.merge(frequency, on="Feature", how="left")
    else:
        qc = overall.assign(RetentionFrequency=np.nan)
    qc.to_csv(f"{prefix}_missingness_qc.csv", index=False)
    _plot_missingness_landscape(overall, f"{prefix}_missingness_landscape.png")
    if not folds.empty:
        _plot_retention_stability(folds, f"{prefix}_missingness_retention_stability.png")
    return qc


def _plot_missingness_landscape(overall: pd.DataFrame, output: str) -> None:
    configure_publication_plots()
    import matplotlib.pyplot as plt

    ordered = overall.sort_values("OverallMissingRate", ascending=False).reset_index(drop=True)
    x = np.arange(len(ordered))
    retained = ordered["OverallMissingRate"].le(MISSINGNESS_THRESHOLD).to_numpy()
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    ax.plot(x, ordered["OverallMissingRate"], color="#374151", linewidth=0.8, alpha=0.72)
    ax.scatter(x[retained], ordered.loc[retained, "OverallMissingRate"], s=9, color="#2A9D8F", alpha=0.82, label="Retained (<=50%)", rasterized=True)
    ax.scatter(x[~retained], ordered.loc[~retained, "OverallMissingRate"], s=9, color="#D1495B", alpha=0.82, label="Removed (>50%)", rasterized=True)
    ax.axhline(MISSINGNESS_THRESHOLD, color="#B23A48", linestyle="--", linewidth=1.3, label="50% threshold")
    removed = int((ordered["OverallMissingRate"] > MISSINGNESS_THRESHOLD).sum())
    rates = ordered["OverallMissingRate"].to_numpy(dtype=float)
    summary = (
        f"Features: {len(ordered):,}\n"
        f"Retained: {len(ordered) - removed:,}  |  Removed: {removed:,}\n"
        f"Median: {np.nanmedian(rates):.1%}  |  IQR: "
        f"{np.nanpercentile(rates, 25):.1%}-{np.nanpercentile(rates, 75):.1%}"
    )
    ax.set(title="Feature Missingness Landscape", xlabel="Biomarkers ranked by descriptive missingness", ylabel="Overall missing rate")
    ax.text(0.99, 0.96, summary, transform=ax.transAxes, ha="right", va="top", fontsize=8.5, bbox={"facecolor": "white", "edgecolor": "#D1D5DB", "alpha": 0.92, "pad": 5})
    ax.set_ylim(0, 1.02)
    ax.set_xlim(-max(1, len(ordered) * 0.01), max(1, len(ordered) - 1) + max(1, len(ordered) * 0.01))
    ax.legend(frameon=False, loc="upper left", ncol=3, fontsize=8.5)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def _plot_retention_stability(folds: pd.DataFrame, output: str) -> None:
    configure_publication_plots()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    matrix = folds.pivot_table(index="Feature", columns="Fold", values="Retained", aggfunc="first")
    frequency = matrix.mean(axis=1)
    order = (
        pd.DataFrame({"frequency": frequency, "feature": matrix.index.astype(str)}, index=matrix.index)
        .sort_values(["frequency", "feature"], ascending=[False, True])
        .index
    )
    matrix = matrix.loc[order]
    frequency = frequency.loc[order]
    height = min(12.0, max(5.4, 4.8 + 0.004 * len(matrix)))
    fig = plt.figure(figsize=(10.2, height))
    grid = fig.add_gridspec(1, 2, width_ratios=[5.2, 1.15], wspace=0.08)
    ax = fig.add_subplot(grid[0, 0])
    frequency_ax = fig.add_subplot(grid[0, 1], sharey=ax)
    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", interpolation="nearest", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set(title="Fold-local Feature Retention Stability", xlabel="Outer fold", ylabel="Biomarkers ordered by retention frequency")
    ax.set_xticks(np.arange(matrix.shape[1]), [str(value) for value in matrix.columns])
    ax.set_yticks([])
    frequency_ax.barh(np.arange(len(frequency)), frequency.to_numpy(dtype=float), color="#35618D", height=1.0)
    frequency_ax.set_xlim(0, 1)
    frequency_ax.set_xlabel("Retention\nfrequency")
    frequency_ax.set_yticks([])
    frequency_ax.grid(axis="x", color="#E5E7EB", linewidth=0.6)
    frequency_ax.invert_yaxis()
    complete = int(np.isclose(frequency, 1.0).sum())
    variable = int(((frequency > 0) & (frequency < 1)).sum())
    removed = int(np.isclose(frequency, 0.0).sum())
    fig.text(0.5, 0.01, f"Always retained: {complete:,}  |  Fold-variable: {variable:,}  |  Always removed: {removed:,}", ha="center", fontsize=8.5)
    ax.legend(
        handles=[
            Patch(facecolor=plt.get_cmap("RdYlGn")(0.0), label="Removed"),
            Patch(facecolor=plt.get_cmap("RdYlGn")(1.0), label="Retained"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9,
        ncol=2,
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.92, bottom=0.10)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def imputation_fallback_reasons(fold_tables: Sequence[pd.DataFrame]) -> list[str]:
    reasons: set[str] = set()
    for table in fold_tables:
        if table is None or table.empty or "ImputationFallbackReason" not in table.columns:
            continue
        reasons.update(
            str(value)
            for value in table["ImputationFallbackReason"].dropna().tolist()
            if str(value).strip()
        )
    return sorted(reasons)

def write_reproducibility_manifest(
    output: str | os.PathLike[str],
    *,
    input_path: str | os.PathLike[str],
    task: str,
    model: str,
    seed: int,
    roles: FeatureRoles,
    imputation: str,
    final_hyperparameters: Mapping | None = None,
    artifacts: Sequence[str] = (),
    imputation_fallbacks: Sequence[str] = (),
    tuning_studies: Sequence[Mapping] = (),
    tuning_artifacts: Sequence[str] = (),
) -> None:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        revision = None
    versions = {}
    for package in ("numpy", "pandas", "sklearn", "optuna", "joblib"):
        try:
            module = __import__(package)
            versions[package] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[package] = None
    source = Path(input_path)
    payload = {
        "schema_version": 1,
        "biomarkerml_version": BIOMARKERML_VERSION,
        "code_revision": revision,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "task": task,
        "model": model,
        "random_seed": int(seed),
        "python_version": platform.python_version(),
        "package_versions": versions,
        "input": {"name": source.name, "sha256": file_sha256(source)},
        "preprocessing": {
            "missingness_threshold": MISSINGNESS_THRESHOLD,
            "missingness_rule": "drop when training missing fraction > 0.50; retain at exactly 0.50",
            "imputation": imputation,
            "imputation_fallback_triggered": bool(imputation_fallbacks),
            "imputation_fallback_reasons": sorted({str(value) for value in imputation_fallbacks if str(value)}),
            "feature_roles": asdict(roles),
        },
        "final_hyperparameters": dict(final_hyperparameters or {}),
        "adaptive_tuning": [dict(study) for study in tuning_studies],
        "adaptive_tuning_artifacts": [str(value) for value in tuning_artifacts],
        "artifacts": [str(Path(value).name) for value in artifacts],
    }
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
