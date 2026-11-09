from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from core.data_contracts import (
    FeatureRoles,
    RoleAwarePreprocessor,
    modeling_frame,
    preserve_dataframe_output,
    resolve_feature_roles,
    selector_for_method,
    write_missingness_qc,
    imputation_fallback_reasons,
    write_reproducibility_manifest,
)
from core.classification_metrics import classification_sensitivity_specificity
from core.adaptive_optuna import (
    reject_legacy_n_trials,
    run_adaptive_study,
    tolerance_for_objective,
)
from core.publication_utils import (
    DEFAULT_RANDOM_STATE,
    BorutaProteinSelector,
    assert_protein_level_features,
    configure_publication_plots,
    estimator_importance,
    export_outer_fold_biomarkers,
    MLP_HIDDEN_LAYER_ARCHITECTURES,
    decode_mlp_hidden_layer_sizes,
    make_elasticnet_logistic_regression,
    make_elasticnet_selector,
    plot_classification_nested_cv,
    plot_confusion_matrix,
    plot_metric_bars,
    plot_roc_curves,
    resolve_model_feature_selection,
    robust_minmax,
    selected_feature_names_from_pipeline,
    set_global_seed,
    task_run_context,
)

configure_publication_plots()
warnings.filterwarnings("ignore")

RANDOM_STATE = DEFAULT_RANDOM_STATE


@dataclass(frozen=True)
class ClassificationSpec:
    public_name: str
    model_key: str
    final_step: str = "classifier"


class PLSDAClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, n_components: int = 2):
        self.n_components = n_components

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        max_components = max(1, min(X.shape[1], X.shape[0] - 1, len(self.classes_) if len(self.classes_) > 2 else 2))
        self.n_components_ = int(max(1, min(self.n_components, max_components)))
        self.pls_ = PLSRegression(n_components=self.n_components_)
        y_bin = label_binarize(y, classes=self.classes_)
        if y_bin.ndim == 1 or y_bin.shape[1] == 1:
            y_bin = np.column_stack([1 - y_bin.ravel(), y_bin.ravel()])
        self.pls_.fit(X, y_bin)
        return self

    def _raw_scores(self, X):
        scores = np.asarray(self.pls_.predict(np.asarray(X)))
        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)
        if scores.shape[1] == 1:
            scores = np.column_stack([1 - scores.ravel(), scores.ravel()])
        return scores

    def predict_proba(self, X):
        scores = self._raw_scores(X)
        scores = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        denom = exp.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        proba = exp / denom
        if proba.shape[1] != len(self.classes_):
            proba = proba[:, : len(self.classes_)]
            proba = proba / np.maximum(proba.sum(axis=1, keepdims=True), 1e-12)
        return proba

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


MODEL_SPECS = {
    "random_forest": ClassificationSpec("Random Forest", "random_forest"),
    "xgboost": ClassificationSpec("XGBoost", "xgboost"),
    "lightgbm": ClassificationSpec("LightGBM", "lightgbm"),
    "svm": ClassificationSpec("Support Vector Machine", "svm"),
    "logistic_regression": ClassificationSpec("Logistic Regression", "logistic_regression"),
    "knn": ClassificationSpec("K-Nearest Neighbors", "knn"),
    "neural_network": ClassificationSpec("Neural Network", "neural_network"),
    "gaussiannb": ClassificationSpec("Gaussian Naive Bayes", "gaussiannb"),
    "plsda": ClassificationSpec("PLS-DA", "plsda"),
}


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def load_classification_data(
    csv_path: str,
    covariates: list[str] | None = None,
    categorical_covariates: list[str] | None = None,
    *,
    return_roles: bool = False,
):
    df = pd.read_csv(csv_path)
    if "Label" not in df.columns:
        raise ValueError("Input CSV must contain a Label column.")
    roles = resolve_feature_roles(
        df,
        target="Label",
        identifier="SampleID",
        covariates=covariates,
        categorical_covariates=categorical_covariates,
    )
    sample_ids = df["SampleID"].astype(str) if "SampleID" in df.columns else pd.Series([f"Sample_{i+1}" for i in range(len(df))])
    X = modeling_frame(df, roles)
    assert_protein_level_features(roles.biomarkers, "Classification training")
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["Label"].astype(str))
    result = (X, y, encoder, list(roles.biomarkers), sample_ids)
    return (*result, roles) if return_roles else result


def _outer_cv(y: np.ndarray):
    counts = np.bincount(y)
    min_count = int(counts.min()) if len(counts) else 0
    if min_count < 2:
        raise ValueError("At least two samples per class are required for stratified cross-validation.")
    return StratifiedKFold(n_splits=min(5, min_count), shuffle=True, random_state=RANDOM_STATE)


def _inner_cv(y: np.ndarray):
    counts = np.bincount(y)
    min_count = int(counts.min()) if len(counts) else 0
    splits = min(3, min_count)
    if splits < 2:
        return None
    return StratifiedKFold(n_splits=splits, shuffle=True, random_state=RANDOM_STATE)


def _fit_import_xgboost(params: dict[str, Any], n_classes: int):
    try:
        from xgboost import XGBClassifier

        objective = "multi:softprob" if n_classes > 2 else "binary:logistic"
        return XGBClassifier(
            objective=objective,
            eval_metric="mlogloss" if n_classes > 2 else "logloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
            tree_method="hist",
            **params,
        )
    except Exception:
        return HistGradientBoostingClassifier(random_state=RANDOM_STATE, **{k: v for k, v in params.items() if k in {"learning_rate", "max_iter", "max_leaf_nodes", "l2_regularization"}})


def _fit_import_lightgbm(params: dict[str, Any], n_classes: int):
    try:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="multiclass" if n_classes > 2 else "binary",
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbosity=-1,
            **params,
        )
    except Exception:
        return HistGradientBoostingClassifier(random_state=RANDOM_STATE, **{k: v for k, v in params.items() if k in {"learning_rate", "max_iter", "max_leaf_nodes", "l2_regularization"}})


def default_params(model_key: str, n_features: int, n_samples: int, n_classes: int) -> dict[str, Any]:
    if model_key == "random_forest":
        return {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 2, "max_features": "sqrt"}
    if model_key == "xgboost":
        return {"n_estimators": 250, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.7, "reg_lambda": 5.0, "reg_alpha": 0.1}
    if model_key == "lightgbm":
        return {"n_estimators": 250, "num_leaves": 15, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.7, "reg_lambda": 5.0, "min_child_samples": max(3, min(20, n_samples // 10))}
    if model_key == "svm":
        return {"C": 1.0, "gamma": "scale", "kernel": "rbf"}
    if model_key == "logistic_regression":
        return {"C": 1.0, "l1_ratio": 0.5}
    if model_key == "knn":
        return {"n_neighbors": max(1, min(7, n_samples - 1)), "weights": "distance", "metric": "minkowski"}
    if model_key == "neural_network":
        return {"hidden_layer_sizes": (64,), "alpha": 0.01, "learning_rate_init": 0.001}
    if model_key == "gaussiannb":
        return {"var_smoothing": 1e-8}
    if model_key == "plsda":
        return {"n_components": max(1, min(2, n_features, n_samples - 1))}
    return {}


def build_estimator(model_key: str, params: dict[str, Any], n_classes: int):
    if model_key == "random_forest":
        return RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1, **params)
    if model_key == "xgboost":
        return _fit_import_xgboost(params, n_classes)
    if model_key == "lightgbm":
        return _fit_import_lightgbm(params, n_classes)
    if model_key == "svm":
        base_estimator = SVC(class_weight="balanced", random_state=RANDOM_STATE, **params)
        return CalibratedClassifierCV(
            estimator=base_estimator,
            method="sigmoid",
            cv=5,
            ensemble=False,
        )
    if model_key == "logistic_regression":
        return make_elasticnet_logistic_regression(
            solver="saga",
            class_weight="balanced",
            max_iter=20000,
            random_state=RANDOM_STATE,
            **params,
        )
    if model_key == "knn":
        params = dict(params)
        params["n_neighbors"] = max(1, int(params.get("n_neighbors", 5)))
        return KNeighborsClassifier(**params)
    if model_key == "neural_network":
        params = dict(params)
        params["hidden_layer_sizes"] = decode_mlp_hidden_layer_sizes(params["hidden_layer_sizes"])
        return MLPClassifier(
            activation="relu",
            solver="adam",
            early_stopping=True,
            validation_fraction=0.2,
            max_iter=800,
            random_state=RANDOM_STATE,
            **params,
        )
    if model_key == "gaussiannb":
        return GaussianNB(**params)
    if model_key == "plsda":
        return PLSDAClassifier(**params)
    raise ValueError(f"Unsupported classification model: {model_key}")


def build_pipeline(
    model_key: str,
    feature_selection: str,
    params: dict[str, Any],
    n_classes: int,
    roles: FeatureRoles | None = None,
    imputation: str = "median",
) -> Pipeline:
    if roles is None:
        # Backward-compatible construction for callers that supply already prepared arrays.
        scaler = StandardScaler()
        if model_key == "lightgbm":
            scaler = preserve_dataframe_output(scaler)
        steps: list[tuple[str, Any]] = [("scaler", scaler)]
        selector = selector_for_method(feature_selection, task="classification") if model_key != "plsda" else None
        if selector is not None:
            if model_key == "lightgbm":
                selector = preserve_dataframe_output(selector)
            steps.append(("feature_selection", selector))
    else:
        selector = selector_for_method(feature_selection, task="classification") if model_key != "plsda" else None
        steps = [("preprocessor", RoleAwarePreprocessor(
            biomarker_columns=roles.biomarkers,
            numeric_covariates=roles.numeric_covariates,
            categorical_covariates=roles.categorical_covariates,
            imputation=imputation,
            selector=selector,
        ))]
    steps.append(("classifier", build_estimator(model_key, params, n_classes)))
    return Pipeline(steps)


def _suggest_params(trial, model_key: str, n_features: int, n_samples: int, n_classes: int) -> dict[str, Any]:
    if model_key == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 900, step=100),
            "max_depth": trial.suggest_categorical("max_depth", [None, 3, 5, 8, 12]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, max(2, min(10, n_samples // 8))),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
        }
    if model_key == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 20.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        }
    if model_key == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "num_leaves": trial.suggest_int("num_leaves", 7, 31),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 20.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 3, max(4, min(30, n_samples // 4))),
        }
    if model_key == "svm":
        return {
            "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            "kernel": trial.suggest_categorical("kernel", ["linear", "rbf"]),
        }
    if model_key == "logistic_regression":
        return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True), "l1_ratio": trial.suggest_float("l1_ratio", 0.05, 0.95)}
    if model_key == "knn":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, max(2, min(15, n_samples - 1))),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "metric": trial.suggest_categorical("metric", ["minkowski", "manhattan"]),
        }
    if model_key == "neural_network":
        return {
            "hidden_layer_sizes": trial.suggest_categorical(
                "hidden_layer_sizes", list(MLP_HIDDEN_LAYER_ARCHITECTURES)
            ),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 5e-3, log=True),
        }
    if model_key == "gaussiannb":
        return {"var_smoothing": trial.suggest_float("var_smoothing", 1e-12, 1e-6, log=True)}
    if model_key == "plsda":
        upper = max(1, min(n_features, n_samples - 1, 10))
        return {"n_components": trial.suggest_int("n_components", 1, upper)}
    return default_params(model_key, n_features, n_samples, n_classes)


def tune_params(
    model_key: str,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_selection: str,
    n_trials: int | None = None,
    roles: FeatureRoles | None = None,
    imputation: str = "median",
    *,
    study_scope: str = "configured_retraining",
    outer_fold: int | None = None,
    study_id: str | None = None,
    provenance_path: str | Path | None = None,
    tuning_reports: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    reject_legacy_n_trials(n_trials)
    inner_cv = _inner_cv(y)
    n_classes = len(np.unique(y))
    fallback = default_params(model_key, X.shape[1], X.shape[0], n_classes)
    if inner_cv is None:
        return fallback
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except Exception:
        return fallback

    def objective(trial):
        params = _suggest_params(trial, model_key, X.shape[1], X.shape[0], n_classes)
        pipe = build_pipeline(model_key, feature_selection, params, n_classes, roles, imputation)
        scores = cross_val_score(pipe, X, y, cv=inner_cv, scoring="f1_weighted", n_jobs=1, error_score=np.nan)
        value = float(np.nanmean(scores))
        return value

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    report = run_adaptive_study(
        study,
        objective,
        task="classification",
        model=model_key,
        study_scope=study_scope,
        objective_name="weighted_f1",
        direction="maximize",
        tolerance=tolerance_for_objective("weighted_f1"),
        seed=RANDOM_STATE,
        study_id=study_id or f"classification-{model_key}-{study_scope}",
        outer_fold=outer_fold,
        provenance_path=provenance_path,
    )
    if tuning_reports is not None:
        tuning_reports.append(report)
    return study.best_params if study.best_params else fallback


def _predict_proba_aligned(model: Pipeline, X: pd.DataFrame, classes: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X), dtype=float)
    else:
        pred = model.predict(X)
        proba = np.zeros((len(pred), len(classes)), dtype=float)
        for i, label in enumerate(pred):
            pos = int(np.where(classes == label)[0][0]) if label in classes else 0
            proba[i, pos] = 1.0
    if proba.ndim == 1:
        proba = np.column_stack([1 - proba, proba])
    if proba.shape[1] != len(classes):
        aligned = np.zeros((proba.shape[0], len(classes)), dtype=float)
        fitted_classes = getattr(model.named_steps.get("classifier"), "classes_", classes)
        for j, label in enumerate(fitted_classes):
            if label in classes and j < proba.shape[1]:
                aligned[:, int(np.where(classes == label)[0][0])] = proba[:, j]
        row_sum = aligned.sum(axis=1, keepdims=True)
        proba = np.divide(aligned, np.maximum(row_sum, 1e-12))
    return proba


def _auc_metrics(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> tuple[float, float]:
    try:
        if len(classes) == 2:
            value = roc_auc_score(y_true, proba[:, 1])
            return float(value), float(value)
        return (
            float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro")),
            float(roc_auc_score(y_true, proba, multi_class="ovr", average="micro")),
        )
    except Exception:
        return float("nan"), float("nan")


def _macro_roc_curve(class_fpr: list[np.ndarray], class_tpr: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    all_fpr = np.unique(np.concatenate(class_fpr))
    mean_tpr = np.zeros_like(all_fpr)
    for fpr, tpr in zip(class_fpr, class_tpr):
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= len(class_fpr)
    return all_fpr, mean_tpr


def _sensitivity_specificity(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> tuple[float, float]:
    return classification_sensitivity_specificity(y_true, y_pred, classes)


def _transformed_matrix(pipeline: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, list[str], str]:
    preprocessor = pipeline.named_steps.get("preprocessor")
    names = list(preprocessor.get_feature_names_out()) if preprocessor is not None else selected_feature_names_from_pipeline(pipeline, X.columns.tolist())
    feature_source = "selected_biomarkers_and_covariates" if preprocessor is not None else ("selected_proteins" if "feature_selection" in pipeline.named_steps else "original_proteins")
    current: Any = X
    for step_name, step in pipeline.steps[:-1]:
        current = step.transform(current)
    matrix = np.asarray(current)
    if len(names) != matrix.shape[1]:
        names = [f"Protein_{i+1}" for i in range(matrix.shape[1])]
    assert_protein_level_features(names, "Classification model explanation")
    return matrix, names, feature_source


def _plot_confusion(prefix: str, model_key: str, y_true: np.ndarray, y_pred: np.ndarray, encoder: LabelEncoder) -> None:
    labels = np.arange(len(encoder.classes_))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plot_confusion_matrix(cm, encoder.classes_, f"{MODEL_SPECS[model_key].public_name} Confusion Matrix", f"{prefix}_{model_key}_confusion_matrix.png")


def _plot_metrics(prefix: str, model_key: str, metrics: dict[str, float]) -> None:
    labels = ["Accuracy", "Weighted F1", "Macro AUC", "Micro AUC"]
    values = [metrics.get("Accuracy"), metrics.get("weighted_F1"), metrics.get("Macro_AUC"), metrics.get("Micro_AUC")]
    values = [0.0 if not np.isfinite(v) else float(v) for v in values]
    plot_metric_bars(labels, values, f"{MODEL_SPECS[model_key].public_name} Performance", f"{prefix}_{model_key}_metrics.png", ylim=(0.0, 1.12))


def _plot_roc(prefix: str, model_key: str, y_true: np.ndarray, proba: np.ndarray, encoder: LabelEncoder) -> None:
    classes = np.arange(len(encoder.classes_))
    roc_data: dict[str, Any] = {"fpr": {}, "tpr": {}, "roc_auc": {}}
    if len(classes) == 2:
        fpr, tpr, _ = roc_curve(y_true, proba[:, 1])
        roc_auc = auc(fpr, tpr)
        roc_data["fpr"][1] = fpr
        roc_data["tpr"][1] = tpr
        roc_data["roc_auc"][1] = roc_auc
        roc_data["roc_auc"]["macro"] = roc_auc
        roc_data["roc_auc"]["micro"] = roc_auc
        curves = [{"fpr": fpr, "tpr": tpr, "auc": roc_auc, "label": "ROC"}]
    else:
        y_bin = label_binarize(y_true, classes=classes)
        curves = []
        class_fpr = []
        class_tpr = []
        for idx, class_label in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, idx], proba[:, idx])
            roc_auc = auc(fpr, tpr)
            class_fpr.append(fpr)
            class_tpr.append(tpr)
            roc_data["fpr"][int(class_label)] = fpr
            roc_data["tpr"][int(class_label)] = tpr
            roc_data["roc_auc"][int(class_label)] = roc_auc
            curves.append({"fpr": fpr, "tpr": tpr, "auc": roc_auc, "label": str(encoder.classes_[idx])})
        roc_data["roc_auc"]["macro"], roc_data["roc_auc"]["micro"] = _auc_metrics(y_true, proba, classes)
        macro_fpr, macro_tpr = _macro_roc_curve(class_fpr, class_tpr)
        micro_fpr, micro_tpr, _ = roc_curve(y_bin.ravel(), proba.ravel())
        roc_data["fpr"]["macro"], roc_data["tpr"]["macro"] = macro_fpr, macro_tpr
        roc_data["fpr"]["micro"], roc_data["tpr"]["micro"] = micro_fpr, micro_tpr
        curves.extend(
            [
                {"fpr": macro_fpr, "tpr": macro_tpr, "auc": roc_data["roc_auc"]["macro"], "label": "Macro-average"},
                {"fpr": micro_fpr, "tpr": micro_tpr, "auc": roc_data["roc_auc"]["micro"], "label": "Micro-average"},
            ]
        )
    np.save(f"{prefix}_{model_key}_roc_data.npy", roc_data, allow_pickle=True)
    plot_roc_curves(curves, f"{MODEL_SPECS[model_key].public_name} ROC", f"{prefix}_{model_key}_roc_curve.png")


def _plot_nested(prefix: str, model_key: str, fold_metrics: pd.DataFrame) -> None:
    plot_classification_nested_cv(
        fold_metrics["Fold"],
        fold_metrics["weighted_F1"],
        fold_metrics["AUC"],
        f"{MODEL_SPECS[model_key].public_name} Nested CV",
        f"{prefix}_{model_key}_nested_cv_f1_auc.png",
    )


def train_classification_model(
    csv_path: str,
    prefix: str | None,
    model_key: str,
    feature_selection: str = "no",
    imputation: str = "median",
    covariates: list[str] | None = None,
    categorical_covariates: list[str] | None = None,
) -> Pipeline:
    set_seed()
    if model_key not in MODEL_SPECS:
        raise ValueError(f"Unsupported classification model: {model_key}")
    X, y, encoder, feature_names, sample_ids, roles = load_classification_data(
        csv_path, covariates, categorical_covariates, return_roles=True
    )
    prefix = prefix or Path(csv_path).stem
    feature_selection = resolve_model_feature_selection(feature_selection, model_key, "classification")
    if model_key == "plsda":
        feature_selection = "none"
    print(f"Running {MODEL_SPECS[model_key].public_name} with feature_selection={feature_selection}")

    classes = np.arange(len(encoder.classes_))
    outer = _outer_cv(y)
    fold_rows = []
    pred_frames = []
    oof_true = []
    oof_pred = []
    oof_proba = []
    best_params_by_fold = []
    tuning_reports: list[dict[str, Any]] = []
    missingness_folds = []

    for fold, (train_idx, test_idx) in enumerate(outer.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        params = tune_params(
            model_key,
            X_train,
            y_train,
            feature_selection,
            roles=roles,
            imputation=imputation,
            study_scope="outer_fold",
            outer_fold=fold,
            study_id=f"classification-{model_key}-outer-fold-{fold}",
            provenance_path=f"{prefix}_{model_key}_fold{fold}_optuna_study.json",
            tuning_reports=tuning_reports,
        )
        best_params_by_fold.append(params)
        model = build_pipeline(model_key, feature_selection, params, len(classes), roles, imputation)
        model.fit(X_train, y_train)
        missingness_folds.append(model.named_steps["preprocessor"].biomarker_diagnostics(fold))
        export_outer_fold_biomarkers(prefix, model_key, fold, model, feature_names)
        y_pred = model.predict(X_test)
        proba = _predict_proba_aligned(model, X_test, classes)
        macro_auc, micro_auc = _auc_metrics(y_test, proba, classes)
        sensitivity, specificity = _sensitivity_specificity(y_test, y_pred, classes)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        fold_rows.append(
            {
                "Model": model_key,
                "Fold": fold,
                "AUC": macro_auc,
                "weighted_F1": float(weighted_f1),
                "Accuracy": float(accuracy_score(y_test, y_pred)),
                "Sensitivity": sensitivity,
                "Specificity": specificity,
                "Macro_AUC": macro_auc,
                "Micro_AUC": micro_auc,
                "Number_of_Train_Samples": int(len(train_idx)),
                "Number_of_Test_Samples": int(len(test_idx)),
            }
        )
        fold_pred = pd.DataFrame(
            {
                "SampleID": sample_ids.iloc[test_idx].to_numpy(),
                "Original Label": encoder.inverse_transform(y_test),
                "Predicted Label": encoder.inverse_transform(y_pred.astype(int)),
                "Fold": fold,
            }
        )
        for idx, cls in enumerate(encoder.classes_):
            fold_pred[f"Probability_{cls}"] = proba[:, idx]
        fold_pred.to_csv(f"{prefix}_{model_key}_fold{fold}_predictions.csv", index=False)
        pred_frames.append(fold_pred)
        oof_true.append(y_test)
        oof_pred.append(y_pred)
        oof_proba.append(proba)

    y_true_all = np.concatenate(oof_true)
    y_pred_all = np.concatenate(oof_pred)
    proba_all = np.vstack(oof_proba)
    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(f"{prefix}_{model_key}_outer_fold_metrics.csv", index=False)
    predictions = pd.concat(pred_frames, ignore_index=True)
    predictions.to_csv(f"{prefix}_{model_key}_predictions.csv", index=False)

    pooled_macro_auc, pooled_micro_auc = _auc_metrics(y_true_all, proba_all, classes)
    summary_metrics = {
        "Model": model_key,
        "Accuracy": float(accuracy_score(y_true_all, y_pred_all)),
        "weighted_F1": float(f1_score(y_true_all, y_pred_all, average="weighted", zero_division=0)),
        "OOF_AUC": pooled_macro_auc,
        "Macro_AUC": pooled_macro_auc,
        "Micro_AUC": pooled_micro_auc,
        "MeanFold_AUC": float(np.nanmean(fold_metrics["Macro_AUC"])),
        "SDFold_AUC": float(np.nanstd(fold_metrics["Macro_AUC"], ddof=1)) if len(fold_metrics) > 1 else 0.0,
        "MetricScope": "Pooled outer-fold out-of-fold predictions",
    }
    pd.DataFrame([summary_metrics]).to_csv(f"{prefix}_{model_key}_final_validation_metrics.csv", index=False)
    _plot_confusion(prefix, model_key, y_true_all, y_pred_all, encoder)
    _plot_roc(prefix, model_key, y_true_all, proba_all, encoder)
    _plot_metrics(prefix, model_key, summary_metrics)
    _plot_nested(prefix, model_key, fold_metrics)

    write_missingness_qc(prefix + "_" + model_key, X, roles, missingness_folds)

    final_params = tune_params(
        model_key,
        X,
        y,
        feature_selection,
        roles=roles,
        imputation=imputation,
        study_scope="final_full_development",
        study_id=f"classification-{model_key}-final",
        provenance_path=f"{prefix}_{model_key}_final_optuna_study.json",
        tuning_reports=tuning_reports,
    )
    final_model = build_pipeline(model_key, feature_selection, final_params, len(classes), roles, imputation)
    final_model.fit(X, y)
    joblib.dump(final_model, f"{prefix}_{model_key}_model.pkl")
    joblib.dump((X, y, encoder), f"{prefix}_{model_key}_data.pkl")
    X_final, final_names, _ = _transformed_matrix(final_model, X)
    pd.DataFrame(X_final, columns=final_names).assign(Label=encoder.inverse_transform(y)).to_csv(f"{prefix}_{model_key}_transformed_features.csv", index=False)
    with open(f"{prefix}_{model_key}_best_params.json", "w", encoding="utf-8") as handle:
        json.dump({"final": final_params, "outer_folds": best_params_by_fold, "feature_selection": feature_selection, "imputation": imputation, "feature_roles": roles.__dict__, "adaptive_tuning": tuning_reports}, handle, indent=2)
    write_reproducibility_manifest(
        f"{prefix}_{model_key}_manifest.json",
        input_path=csv_path,
        task="classification",
        model=model_key,
        seed=RANDOM_STATE,
        roles=roles,
        imputation=imputation,
        final_hyperparameters=final_params,
        artifacts=[f"{prefix}_{model_key}_predictions.csv", f"{prefix}_{model_key}_model.pkl"],
        imputation_fallbacks=imputation_fallback_reasons(missingness_folds),
        tuning_studies=tuning_reports,
    )

    from engines.reliability.learning_curves import generate_learning_curves

    generate_learning_curves(
        prefix,
        task="classification",
        models=[model_key],
        estimator=final_model,
        X_train=X,
        y_train=y,
    )
    print(f"Finished {MODEL_SPECS[model_key].public_name}")
    return final_model


def _cli(model_key: str, function_name: str) -> None:
    parser = argparse.ArgumentParser(description=f"Run protein-level nested CV for {MODEL_SPECS[model_key].public_name}.")
    parser.add_argument("--input", "-i", "--csv", dest="csv", required=True)
    parser.add_argument("-p", "--prefix", default=None)
    parser.add_argument("-f", "--feature_selection", default="no", help="yes/no automatic protein feature selection")
    parser.add_argument("--outdir", default=".", help="Base output directory")
    parser.add_argument("--jobs", type=int, default=1, help="Accepted for CLI consistency")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--imputation", choices=["median", "knn"], default="median")
    parser.add_argument("--covariates", nargs="*", default=None)
    parser.add_argument("--categorical-covariates", nargs="*", default=None)
    args = parser.parse_args()
    set_global_seed(args.seed)
    with task_run_context(args.outdir, "classification"):
        train_classification_model(args.csv, args.prefix, model_key, args.feature_selection, args.imputation, args.covariates, args.categorical_covariates)


__all__ = ["PLSDAClassifier", "train_classification_model", "MODEL_SPECS"]
