from __future__ import annotations

import argparse
import glob
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, learning_curve, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline

from core.publication_utils import (
    DEFAULT_RANDOM_STATE,
    build_output_prefix,
    configure_publication_plots,
    plot_learning_progression,
    plot_training_size_curve,
    prediction_input_with_feature_names,
    set_global_seed,
    stage_task_inputs,
    task_output_dir,
    task_run_context,
    write_learning_progression,
    write_training_history,
)

configure_publication_plots()


CLASSIFICATION_MODEL_FILES = {
    "random_forest": "random_forest",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "svm": "svm",
    "logistic_regression": "logistic_regression",
    "knn": "knn",
    "neural_network": "neural_network",
    "plsda": "plsda",
    "gaussiannb": "gaussiannb",
}


def _validate_training_data(X, y, context: str):
    if X is None:
        raise ValueError(f"{context}: X_train is required for the training-size diagnostic.")
    if y is None:
        raise ValueError(f"{context}: y_train is required for the training-size diagnostic.")

    X_use = X if isinstance(X, pd.DataFrame) else np.asarray(X)
    y_use = np.asarray(y)
    if getattr(X_use, "ndim", None) != 2:
        raise ValueError(f"{context}: X_train must be a two-dimensional feature matrix.")
    if y_use.ndim == 2 and y_use.shape[1] == 1:
        y_use = y_use.reshape(-1)
    if y_use.ndim != 1:
        raise ValueError(f"{context}: y_train must be a one-dimensional target vector.")
    if X_use.shape[0] != y_use.shape[0]:
        raise ValueError(
            f"{context}: X_train and y_train contain different sample counts "
            f"({X_use.shape[0]} and {y_use.shape[0]})."
        )
    if X_use.shape[0] == 0:
        raise ValueError(f"{context}: training data cannot be empty.")
    return X_use, y_use


def _load_classification_artifact(prefix: str, model: str):
    model_path = f"{prefix}_{model}_model.pkl"
    data_path = f"{prefix}_{model}_data.pkl"
    if not os.path.exists(model_path) or not os.path.exists(data_path):
        return None
    estimator = joblib.load(model_path)
    data = joblib.load(data_path)
    if isinstance(data, tuple) and len(data) >= 2:
        X = data[0]
        y = data[1]
    else:
        return None
    X_use, y_use = _validate_training_data(X, y, f"{model} classification artifact")
    return estimator, X_use, y_use


def _load_regression_artifact(prefix: str, model: str):
    model_path = f"{prefix}_{model}_best_model.pkl"
    x_path = f"{prefix}_{model}_X.npy"
    y_path = f"{prefix}_{model}_y_true.npy"
    feature_path = f"{prefix}_{model}_feature_names.npy"
    if not all(os.path.exists(p) for p in [model_path, x_path, y_path, feature_path]):
        return None
    estimator = joblib.load(model_path)
    X = np.load(x_path)
    y = np.load(y_path)
    features = np.load(feature_path, allow_pickle=True)
    X_df = pd.DataFrame(X, columns=features.astype(str))
    transformed_path = f"{prefix}_{model}_transformed_features.csv"
    if os.path.exists(transformed_path):
        transformed = pd.read_csv(transformed_path, usecols=["Label"])
        y = transformed["Label"].to_numpy()
    X_use, y_use = _validate_training_data(X_df, y, f"{model} regression artifact")
    return estimator, X_use, y_use


def _compute_curve(estimator, X, y, task: str):
    if estimator is None:
        raise ValueError("Training-size diagnostic: estimator is required.")
    X, y = _validate_training_data(X, y, "Training-size diagnostic")
    if len(y) < 8:
        raise ValueError("Training-size diagnostic requires at least eight samples for cross-validation.")
    cv_splits = min(5, max(2, np.min(np.bincount(y.astype(int))) if task == "classification" and np.issubdtype(y.dtype, np.integer) else 5))
    if task == "classification":
        X, y = _interleave_classification_rows(X, y)
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
        scoring = "neg_log_loss" if hasattr(estimator, "predict_proba") else "accuracy"
        sizes = _classification_train_sizes(cv, X, y, estimator)
    else:
        cv = KFold(n_splits=min(5, len(y)), shuffle=True, random_state=DEFAULT_RANDOM_STATE)
        scoring = "neg_mean_squared_error"
        sizes = np.linspace(0.3, 1.0, 5)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=FitFailedWarning)
            train_sizes, train_scores, valid_scores = learning_curve(
                _clone_for_curve(estimator),
                X,
                y,
                train_sizes=sizes,
                cv=cv,
                scoring=scoring,
                n_jobs=1,
                shuffle=task != "classification",
                random_state=DEFAULT_RANDOM_STATE,
                error_score=np.nan,
            )
    except Exception as exc:
        raise RuntimeError(f"Training-size diagnostic fitting failed: {exc}") from exc
    valid_rows = ~(np.isnan(train_scores).all(axis=1) | np.isnan(valid_scores).all(axis=1))
    if not np.any(valid_rows):
        raise RuntimeError("Training-size diagnostic fitting failed for every training fraction.")
    train_sizes = train_sizes[valid_rows]
    train_scores = train_scores[valid_rows]
    valid_scores = valid_scores[valid_rows]
    if scoring.startswith("neg_"):
        train_loss = -np.nanmean(train_scores, axis=1)
        valid_loss = -np.nanmean(valid_scores, axis=1)
    else:
        train_loss = 1 - np.nanmean(train_scores, axis=1)
        valid_loss = 1 - np.nanmean(valid_scores, axis=1)
    return pd.DataFrame({"Training_Size": train_sizes, "Training_Score_Loss": train_loss, "Validation_Score_Loss": valid_loss})


def _interleave_classification_rows(X, y):
    class_indices = [list(np.flatnonzero(y == label)) for label in np.unique(y)]
    order = []
    while any(class_indices):
        for indices in class_indices:
            if indices:
                order.append(indices.pop(0))
    if hasattr(X, "iloc"):
        X = X.iloc[order].reset_index(drop=True)
    else:
        X = np.asarray(X)[order]
    return X, np.asarray(y)[order]


def _classification_train_sizes(cv, X, y, estimator, points: int = 5) -> np.ndarray:
    classes = np.unique(y)
    split_train_indices = [train_idx for train_idx, _ in cv.split(X, y)]
    maximum = min(len(indices) for indices in split_train_indices)
    final_estimator = _final_pipeline_estimator(estimator)
    minimum = max(len(classes), int(getattr(final_estimator, "n_neighbors", 1)))
    for indices in split_train_indices:
        for size in range(len(classes), len(indices) + 1):
            if np.unique(y[indices[:size]]).size == len(classes):
                minimum = max(minimum, size)
                break
        else:
            raise ValueError("Training-size diagnostic could not construct a training subset containing every class.")
    sizes = np.unique(np.rint(np.linspace(minimum, maximum, points)).astype(int))
    if len(sizes) < points and maximum - minimum + 1 >= points:
        sizes = np.arange(maximum - points + 1, maximum + 1, dtype=int)
    if len(sizes) < 2:
        raise ValueError("Training-size diagnostic requires at least two distinct stratified training sizes.")
    return sizes


def _clone_for_curve(estimator):
    diagnostic_estimator = clone(estimator)
    if isinstance(diagnostic_estimator, Pipeline):
        final_name, final_estimator = diagnostic_estimator.steps[-1]
        if isinstance(final_estimator, (MLPClassifier, MLPRegressor)):
            diagnostic_estimator.set_params(**{f"{final_name}__early_stopping": False})
    elif isinstance(diagnostic_estimator, (MLPClassifier, MLPRegressor)):
        diagnostic_estimator.set_params(early_stopping=False)
    return diagnostic_estimator


def _plot_training_size_curve(df: pd.DataFrame, title: str, out: str) -> None:
    plot_training_size_curve(df, title, "Cross-validation loss", out)


def _take_rows(X, indices):
    if hasattr(X, "iloc"):
        return X.iloc[indices]
    return np.asarray(X)[indices]


def _safe_stratify(y: np.ndarray):
    values, counts = np.unique(y, return_counts=True)
    return y if len(values) > 1 and np.min(counts) >= 2 else None


def _split_for_epoch_history(X, y, task: str):
    if len(y) < 8:
        return None
    stratify = _safe_stratify(y) if task == "classification" else None
    try:
        train_idx, valid_idx = train_test_split(
            np.arange(len(y)),
            test_size=0.25,
            random_state=DEFAULT_RANDOM_STATE,
            stratify=stratify,
        )
    except ValueError:
        return None
    return _take_rows(X, train_idx), _take_rows(X, valid_idx), y[train_idx], y[valid_idx]


def _final_pipeline_estimator(estimator):
    if isinstance(estimator, Pipeline):
        if not estimator.steps:
            raise ValueError("Training-size diagnostic received an empty pipeline.")
        return estimator.steps[-1][1]
    return estimator


def _fit_preprocessing(estimator, X_train, X_valid, y_train):
    if not isinstance(estimator, Pipeline):
        return np.asarray(X_train), np.asarray(X_valid), estimator
    if len(estimator.steps) == 1:
        return np.asarray(X_train), np.asarray(X_valid), estimator.steps[-1][1]
    preprocessing = Pipeline([(name, clone(step)) for name, step in estimator.steps[:-1]])
    X_train_t = preprocessing.fit_transform(X_train, y_train)
    X_valid_t = preprocessing.transform(X_valid)
    return X_train_t, X_valid_t, estimator.steps[-1][1]


def _compute_mlp_epoch_history(estimator, X, y, task: str, max_epochs: int = 80) -> list[dict]:
    final_estimator = _final_pipeline_estimator(estimator)
    if task == "classification" and not isinstance(final_estimator, MLPClassifier):
        return []
    if task == "regression" and not isinstance(final_estimator, MLPRegressor):
        return []

    split = _split_for_epoch_history(X, np.asarray(y), task)
    if split is None:
        return []
    X_train, X_valid, y_train, y_valid = split
    X_train_t, X_valid_t, final_estimator = _fit_preprocessing(estimator, X_train, X_valid, y_train)

    model = clone(final_estimator)
    model.set_params(warm_start=True, max_iter=1, early_stopping=False, tol=0.0)
    epochs = max(2, min(max_epochs, int(getattr(final_estimator, "max_iter", max_epochs) or max_epochs)))
    rows: list[dict] = []
    classes = np.unique(y_train) if task == "classification" else None

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for epoch in range(1, epochs + 1):
            model.fit(X_train_t, y_train)
            if task == "classification":
                train_loss = log_loss(y_train, model.predict_proba(X_train_t), labels=classes)
                valid_loss = log_loss(y_valid, model.predict_proba(X_valid_t), labels=classes)
            else:
                train_loss = mean_squared_error(y_train, model.predict(X_train_t))
                valid_loss = mean_squared_error(y_valid, model.predict(X_valid_t))
            rows.append(
                {
                    "Epoch": epoch,
                    "TrainingLoss": float(train_loss),
                    "ValidationLoss": float(valid_loss),
                    "LearningRate": float(getattr(model, "learning_rate_init", np.nan)),
                    "EarlyStoppingTriggered": False,
                    "BestEpoch": np.nan,
                }
            )
    if rows:
        best_epoch = min(rows, key=lambda row: row["ValidationLoss"])["Epoch"]
        for row in rows:
            row["BestEpoch"] = best_epoch
    return rows


def _write_epoch_history_if_supported(estimator, X, y, task: str, prefix: str, model: str) -> pd.DataFrame:
    rows = _compute_mlp_epoch_history(estimator, X, y, task)
    if rows:
        write_training_history(rows, prefix, model)
        learning = pd.DataFrame(rows).rename(columns={"Epoch": "ProgressionStep"})
        learning["ProgressionValue"] = learning["ProgressionStep"]
        learning["ProgressionLabel"] = "Epoch"
        learning["DiagnosticType"] = "Epoch-wise optimization"
        return learning[
            [
                "ProgressionStep",
                "ProgressionValue",
                "TrainingLoss",
                "ValidationLoss",
                "ProgressionLabel",
                "DiagnosticType",
            ]
        ]
    return pd.DataFrame()


def _progression_parameter(estimator, X_train) -> tuple[str, list[float | int], str] | None:
    final_estimator = _final_pipeline_estimator(estimator)
    parameter_prefix = ""
    if isinstance(estimator, Pipeline):
        parameter_prefix = f"{estimator.steps[-1][0]}__"

    # Calibrated SVC keeps the validated SVM hyperparameters on its base estimator.
    calibration_estimator = getattr(final_estimator, "estimator", None)
    if calibration_estimator is not None and hasattr(calibration_estimator, "C"):
        final_estimator = calibration_estimator
        parameter_prefix += "estimator__"

    if hasattr(final_estimator, "n_estimators"):
        selected = max(5, int(getattr(final_estimator, "n_estimators", 100) or 100))
        values = np.unique(np.rint(np.linspace(max(2, selected // 5), selected, 5)).astype(int)).tolist()
        return parameter_prefix + "n_estimators", values, "Number of estimators"
    if hasattr(final_estimator, "n_neighbors"):
        selected = max(1, int(getattr(final_estimator, "n_neighbors", 5) or 5))
        maximum = max(1, min(len(X_train) - 1, max(9, selected * 2 - 1)))
        values = np.unique(np.rint(np.linspace(1, maximum, 5)).astype(int)).tolist()
        return parameter_prefix + "n_neighbors", values, "Number of neighbors"
    if hasattr(final_estimator, "C"):
        selected = max(float(getattr(final_estimator, "C", 1.0) or 1.0), 1e-12)
        values = (selected * np.logspace(-1, 1, 5)).tolist()
        return parameter_prefix + "C", values, "Regularization parameter C"
    if hasattr(final_estimator, "var_smoothing"):
        selected = max(float(getattr(final_estimator, "var_smoothing", 1e-9) or 1e-9), 1e-15)
        values = (selected * np.logspace(-2, 2, 5)).tolist()
        return parameter_prefix + "var_smoothing", values, "Variance smoothing"
    if hasattr(final_estimator, "n_components"):
        selected = max(1, int(getattr(final_estimator, "n_components", 2) or 2))
        maximum = max(1, min(np.asarray(X_train).shape[1], len(X_train) - 1, max(5, selected * 2)))
        values = np.unique(np.rint(np.linspace(1, maximum, min(5, maximum))).astype(int)).tolist()
        return parameter_prefix + "n_components", values, "Number of components"
    return None


def _prediction_input_for_estimator(model, X):
    if isinstance(X, pd.DataFrame):
        names = X.columns.astype(str).tolist()
    else:
        fitted_names = getattr(model, "feature_name_", None)
        if fitted_names is None:
            fitted_names = getattr(model, "feature_names_in_", None)
        names = (
            list(fitted_names)
            if fitted_names is not None
            else [f"Feature_{i+1}" for i in range(np.asarray(X).shape[1])]
        )
    return prediction_input_with_feature_names(X, names, model)


def _classification_loss(model, X, y) -> float:
    X_predict = _prediction_input_for_estimator(model, X)
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(X_predict), dtype=float)
        labels = getattr(model, "classes_", np.unique(y))
        metric_probabilities = _probabilities_for_log_loss(probabilities, labels)
        return float(log_loss(y, metric_probabilities, labels=labels))
    return float(1.0 - accuracy_score(y, model.predict(X_predict)))


def _probabilities_for_log_loss(probabilities, labels):
    labels = np.asarray(labels)
    values = np.asarray(probabilities, dtype=float)
    if values.ndim == 1:
        if len(labels) != 2:
            raise ValueError("Multiclass log-loss requires a two-dimensional probability matrix.")
        if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("Binary log-loss received values outside the probability range [0, 1].")
        return values
    if values.ndim != 2:
        raise ValueError("Classification log-loss requires one- or two-dimensional probabilities.")
    if values.shape[1] != len(labels):
        raise ValueError(
            "Classification log-loss received a probability array whose class axis "
            "does not match the fitted class labels."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Classification log-loss received non-finite probability values.")
    if len(labels) == 2:
        positive = values[:, 1]
        if np.any((positive < 0.0) | (positive > 1.0)):
            raise ValueError("Binary log-loss received positive-class values outside [0, 1].")
        return positive
    row_sums = values.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError("Multiclass log-loss received probabilities whose rows do not sum to one.")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Multiclass log-loss received values outside the probability range [0, 1].")
    return values


def _compute_learning_progression(estimator, X, y, task: str) -> pd.DataFrame:
    """Compute a model-complexity validation curve from real deterministic refits."""
    X, y = _validate_training_data(X, y, "Learning curve")
    split = _split_for_epoch_history(X, y, task)
    if split is None:
        return pd.DataFrame()
    X_train, X_valid, y_train, y_valid = split
    progression = _progression_parameter(estimator, X_train)
    if progression is None:
        return pd.DataFrame()
    parameter, values, label = progression

    rows = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=FitFailedWarning)
        for step, value in enumerate(values, start=1):
            model = clone(estimator)
            model.set_params(**{parameter: value})
            model.fit(X_train, y_train)
            if task == "classification":
                training_loss = _classification_loss(model, X_train, y_train)
                validation_loss = _classification_loss(model, X_valid, y_valid)
            else:
                training_loss = float(
                    mean_squared_error(
                        y_train,
                        model.predict(_prediction_input_for_estimator(model, X_train)),
                    )
                )
                validation_loss = float(
                    mean_squared_error(
                        y_valid,
                        model.predict(_prediction_input_for_estimator(model, X_valid)),
                    )
                )
            rows.append(
                {
                    "ProgressionStep": float(value),
                    "ProgressionValue": float(value),
                    "TrainingLoss": training_loss,
                    "ValidationLoss": validation_loss,
                    "ProgressionLabel": label,
                    "DiagnosticType": "Refit-based model-complexity validation",
                    "Sequence": step,
                }
            )
    return pd.DataFrame(rows)


def generate_learning_curves(
    prefix: str,
    task: str = "classification",
    models: list[str] | None = None,
    *,
    estimator=None,
    X_train=None,
    y_train=None,
) -> pd.DataFrame:
    direct_values = (estimator, X_train, y_train)
    direct_artifact = any(value is not None for value in direct_values)
    if direct_artifact and not all(value is not None for value in direct_values):
        raise ValueError("Training-size diagnostic requires estimator, X_train, and y_train together.")

    rows = []
    learning_rows = []
    if task == "classification":
        candidate_models = models or list(CLASSIFICATION_MODEL_FILES)
        if direct_artifact and len(candidate_models) != 1:
            raise ValueError("Direct training data can be used with exactly one classification model.")
        for model in candidate_models:
            artifact = (estimator, *_validate_training_data(X_train, y_train, model)) if direct_artifact else _load_classification_artifact(prefix, model)
            if artifact is None:
                continue
            estimator, X, y = artifact
            curve = _compute_curve(estimator, X, y, task)
            if curve is None:
                continue
            curve["Model"] = model
            curve.to_csv(f"{prefix}_{model}_training_size_curve.csv", index=False)
            _plot_training_size_curve(curve, f"{model.replace('_', ' ').title()} Training Size Diagnostic", f"{prefix}_{model}_training_size_curve.png")
            learning = _write_epoch_history_if_supported(estimator, X, y, task, prefix, model)
            if learning.empty:
                learning = _compute_learning_progression(estimator, X, y, task)
                if not learning.empty:
                    write_learning_progression(learning, prefix, model)
            if not learning.empty:
                learning["Model"] = model
                learning_rows.append(learning)
            rows.append(curve)
    else:
        candidate_models = models or [Path(p).name.replace(f"{Path(prefix).name}_", "").replace("_best_model.pkl", "") for p in glob.glob(f"{prefix}_*_best_model.pkl")]
        if direct_artifact and len(candidate_models) != 1:
            raise ValueError("Direct training data can be used with exactly one regression model.")
        for model in candidate_models:
            artifact = (estimator, *_validate_training_data(X_train, y_train, model)) if direct_artifact else _load_regression_artifact(prefix, model)
            if artifact is None:
                continue
            estimator, X, y = artifact
            curve = _compute_curve(estimator, X, y, task)
            if curve is None:
                continue
            curve["Model"] = model
            curve.to_csv(f"{prefix}_{model}_training_size_curve.csv", index=False)
            _plot_training_size_curve(curve, f"{model.replace('_', ' ').title()} Training Size Diagnostic", f"{prefix}_{model}_training_size_curve.png")
            learning = _write_epoch_history_if_supported(estimator, X, y, task, prefix, model)
            if learning.empty:
                learning = _compute_learning_progression(estimator, X, y, task)
                if not learning.empty:
                    write_learning_progression(learning, prefix, model)
            if not learning.empty:
                learning["Model"] = model
                learning_rows.append(learning)
            rows.append(curve)

    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(f"{prefix}_training_size_curves.csv", index=False)
    summary = combined.groupby("Training_Size", as_index=False)[["Training_Score_Loss", "Validation_Score_Loss"]].mean()
    _plot_training_size_curve(summary, "Training Size Diagnostic", f"{prefix}_training_size_curve.png")
    if learning_rows:
        learning_combined = pd.concat(learning_rows, ignore_index=True)
        learning_combined.to_csv(f"{prefix}_learning_curves.csv", index=False)
        normalized = []
        for _, group in learning_combined.groupby("Model", sort=False):
            ordered = group.sort_values("ProgressionStep").copy()
            ordered["ProgressionStep"] = np.linspace(0.0, 100.0, len(ordered))
            ordered["ProgressionLabel"] = "Optimization progression (%)"
            normalized.append(ordered)
        learning_summary = pd.concat(normalized, ignore_index=True).groupby(
            "ProgressionStep", as_index=False
        )[["TrainingLoss", "ValidationLoss"]].mean()
        learning_summary["ProgressionValue"] = learning_summary["ProgressionStep"]
        learning_summary["ProgressionLabel"] = "Optimization progression (%)"
        learning_summary["DiagnosticType"] = "Cross-model optimization summary"
        plot_learning_progression(learning_summary, "Learning Curve", f"{prefix}_learning_curve.png")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication training-size validation curves from saved BiomarkerML model artifacts.")
    parser.add_argument("--input", default=None, help="Task input directory containing fitted model artifacts.")
    parser.add_argument("-p", "--prefix", default=None)
    parser.add_argument("--outdir", default=".", help="Base output directory.")
    parser.add_argument("--feature_selection", choices=["yes", "no"], default="no", help="Accepted for CLI consistency.")
    parser.add_argument("--jobs", type=int, default=1, help="Accepted for CLI consistency.")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("-t", "--task", default="classification", choices=["classification", "regression"])
    parser.add_argument("-m", "--models", nargs="*", default=None)
    args = parser.parse_args()
    prefix = build_output_prefix(args.input, args.prefix)
    set_global_seed(args.seed)
    task_dir = "classification" if args.task == "classification" else "regression"
    outdir = task_output_dir(args.outdir, task_dir)
    stage_task_inputs(args.input, outdir, prefix)
    with task_run_context(args.outdir, task_dir):
        generate_learning_curves(prefix, args.task, args.models)


if __name__ == "__main__":
    main()
