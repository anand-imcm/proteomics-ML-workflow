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

from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

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
    MLP_HIDDEN_LAYER_ARCHITECTURES,
    decode_mlp_hidden_layer_sizes,
    estimator_importance,
    export_outer_fold_biomarkers,
    make_elasticnet_selector,
    plot_metric_bars,
    plot_observed_predicted,
    plot_regression_nested_cv,
    plot_residuals,
    resolve_model_feature_selection,
    selected_feature_names_from_pipeline,
    set_global_seed,
    task_run_context,
)

configure_publication_plots()
warnings.filterwarnings("ignore")

RANDOM_STATE = DEFAULT_RANDOM_STATE


@dataclass(frozen=True)
class RegressionSpec:
    public_name: str
    model_key: str


MODEL_SPECS = {
    "Neural_Network_reg": RegressionSpec("Neural Network Regression", "Neural_Network_reg"),
    "Random_Forest_reg": RegressionSpec("Random Forest Regression", "Random_Forest_reg"),
    "SVM_reg": RegressionSpec("Support Vector Regression", "SVM_reg"),
    "XGBoost_reg": RegressionSpec("XGBoost Regression", "XGBoost_reg"),
    "LightGBM_reg": RegressionSpec("LightGBM Regression", "LightGBM_reg"),
    "KNN_reg": RegressionSpec("K-Nearest Neighbors Regression", "KNN_reg"),
    "PLS_reg": RegressionSpec("PLS Regression", "PLS_reg"),
}


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def load_regression_data(
    csv_path: str,
    covariates: list[str] | None = None,
    categorical_covariates: list[str] | None = None,
    *,
    return_roles: bool = False,
):
    df = pd.read_csv(csv_path)
    if "Label" not in df.columns:
        raise ValueError("Input CSV must contain a Label column.")
    roles = resolve_feature_roles(df, target="Label", identifier="SampleID", covariates=covariates, categorical_covariates=categorical_covariates)
    sample_ids = df["SampleID"].astype(str) if "SampleID" in df.columns else pd.Series([f"Sample_{i+1}" for i in range(len(df))])
    X = modeling_frame(df, roles)
    y = pd.to_numeric(df["Label"], errors="coerce")
    if y.isna().any():
        raise ValueError("Regression Label column must be numeric.")
    assert_protein_level_features(roles.biomarkers, "Regression training")
    result = (X, y.to_numpy(dtype=float), list(roles.biomarkers), sample_ids)
    return (*result, roles) if return_roles else result


def _outer_cv(n_samples: int):
    if n_samples < 3:
        raise ValueError("At least three samples are required for regression cross-validation.")
    return KFold(n_splits=min(5, n_samples), shuffle=True, random_state=RANDOM_STATE)


def _inner_cv(n_samples: int):
    if n_samples < 6:
        return None
    return KFold(n_splits=min(3, n_samples), shuffle=True, random_state=RANDOM_STATE)


def _fit_import_xgboost(params: dict[str, Any]):
    try:
        from xgboost import XGBRegressor

        return XGBRegressor(random_state=RANDOM_STATE, n_jobs=1, tree_method="hist", objective="reg:squarederror", **params)
    except Exception:
        return HistGradientBoostingRegressor(random_state=RANDOM_STATE, **{k: v for k, v in params.items() if k in {"learning_rate", "max_iter", "max_leaf_nodes", "l2_regularization"}})


def _fit_import_lightgbm(params: dict[str, Any]):
    try:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(random_state=RANDOM_STATE, n_jobs=1, verbosity=-1, **params)
    except Exception:
        return HistGradientBoostingRegressor(random_state=RANDOM_STATE, **{k: v for k, v in params.items() if k in {"learning_rate", "max_iter", "max_leaf_nodes", "l2_regularization"}})


def default_params(model_key: str, n_features: int, n_samples: int) -> dict[str, Any]:
    if model_key == "Random_Forest_reg":
        return {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 2, "max_features": "sqrt"}
    if model_key == "XGBoost_reg":
        return {"n_estimators": 250, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.7, "reg_lambda": 5.0, "reg_alpha": 0.1}
    if model_key == "LightGBM_reg":
        return {"n_estimators": 250, "num_leaves": 15, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.7, "reg_lambda": 5.0, "min_child_samples": max(3, min(20, n_samples // 10))}
    if model_key == "SVM_reg":
        return {"C": 1.0, "epsilon": 0.1, "gamma": "scale", "kernel": "rbf"}
    if model_key == "KNN_reg":
        return {"n_neighbors": max(1, min(7, n_samples - 1)), "weights": "distance", "metric": "minkowski"}
    if model_key == "Neural_Network_reg":
        return {"hidden_layer_sizes": (64,), "alpha": 0.01, "learning_rate_init": 0.001}
    if model_key == "PLS_reg":
        return {"n_components": max(1, min(2, n_features, n_samples - 1))}
    return {}


def build_regressor(model_key: str, params: dict[str, Any]):
    if model_key == "Random_Forest_reg":
        return RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1, **params)
    if model_key == "XGBoost_reg":
        return _fit_import_xgboost(params)
    if model_key == "LightGBM_reg":
        return _fit_import_lightgbm(params)
    if model_key == "SVM_reg":
        return SVR(**params)
    if model_key == "KNN_reg":
        params = dict(params)
        params["n_neighbors"] = max(1, int(params.get("n_neighbors", 5)))
        return KNeighborsRegressor(**params)
    if model_key == "Neural_Network_reg":
        params = dict(params)
        params["hidden_layer_sizes"] = decode_mlp_hidden_layer_sizes(params["hidden_layer_sizes"])
        return MLPRegressor(
            activation="relu",
            solver="adam",
            early_stopping=True,
            validation_fraction=0.2,
            max_iter=900,
            random_state=RANDOM_STATE,
            **params,
        )
    if model_key == "PLS_reg":
        return PLSRegression(**params)
    raise ValueError(f"Unsupported regression model: {model_key}")


def build_pipeline(
    model_key: str,
    feature_selection: str,
    params: dict[str, Any],
    roles: FeatureRoles | None = None,
    imputation: str = "median",
) -> Pipeline:
    if roles is None:
        scaler = StandardScaler()
        if model_key == "LightGBM_reg":
            scaler = preserve_dataframe_output(scaler)
        steps: list[tuple[str, Any]] = [("scaler", scaler)]
        selector = selector_for_method(feature_selection, task="regression") if model_key != "PLS_reg" else None
        if selector is not None:
            if model_key == "LightGBM_reg":
                selector = preserve_dataframe_output(selector)
            steps.append(("feature_selection", selector))
    else:
        selector = selector_for_method(feature_selection, task="regression") if model_key != "PLS_reg" else None
        steps = [("preprocessor", RoleAwarePreprocessor(
            biomarker_columns=roles.biomarkers,
            numeric_covariates=roles.numeric_covariates,
            categorical_covariates=roles.categorical_covariates,
            imputation=imputation,
            selector=selector,
        ))]
    steps.append(("regressor", build_regressor(model_key, params)))
    return Pipeline(steps)


def _suggest_params(trial, model_key: str, n_features: int, n_samples: int) -> dict[str, Any]:
    if model_key == "Random_Forest_reg":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 900, step=100),
            "max_depth": trial.suggest_categorical("max_depth", [None, 3, 5, 8, 12]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, max(2, min(10, n_samples // 8))),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
        }
    if model_key == "XGBoost_reg":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 20.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        }
    if model_key == "LightGBM_reg":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "num_leaves": trial.suggest_int("num_leaves", 7, 31),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 20.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 3, max(4, min(30, n_samples // 4))),
        }
    if model_key == "SVM_reg":
        return {
            "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
            "epsilon": trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            "kernel": trial.suggest_categorical("kernel", ["linear", "rbf"]),
        }
    if model_key == "KNN_reg":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, max(2, min(15, n_samples - 1))),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "metric": trial.suggest_categorical("metric", ["minkowski", "manhattan"]),
        }
    if model_key == "Neural_Network_reg":
        return {
            "hidden_layer_sizes": trial.suggest_categorical(
                "hidden_layer_sizes", list(MLP_HIDDEN_LAYER_ARCHITECTURES)
            ),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 5e-3, log=True),
        }
    if model_key == "PLS_reg":
        upper = max(1, min(n_features, n_samples - 1, 10))
        return {"n_components": trial.suggest_int("n_components", 1, upper)}
    return default_params(model_key, n_features, n_samples)


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
    inner_cv = _inner_cv(len(y))
    fallback = default_params(model_key, X.shape[1], X.shape[0])
    if inner_cv is None:
        return fallback
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except Exception:
        return fallback

    def objective(trial):
        params = _suggest_params(trial, model_key, X.shape[1], X.shape[0])
        pipe = build_pipeline(model_key, feature_selection, params, roles, imputation)
        scores = cross_val_score(pipe, X, y, cv=inner_cv, scoring="neg_root_mean_squared_error", n_jobs=1, error_score=np.nan)
        value = float(np.nanmean(scores))
        return value

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    report = run_adaptive_study(
        study,
        objective,
        task="regression",
        model=model_key,
        study_scope=study_scope,
        objective_name="neg_root_mean_squared_error",
        direction="maximize",
        tolerance=tolerance_for_objective(
            "neg_root_mean_squared_error",
            scale_floor=max(float(np.nanstd(y)), np.finfo(float).eps),
        ),
        seed=RANDOM_STATE,
        study_id=study_id or f"regression-{model_key}-{study_scope}",
        outer_fold=outer_fold,
        provenance_path=provenance_path,
    )
    if tuning_reports is not None:
        tuning_reports.append(report)
    return study.best_params if study.best_params else fallback


def _r2(y_true, y_pred):
    return float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan")


def _transformed_matrix(pipeline: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, list[str], str]:
    preprocessor = pipeline.named_steps.get("preprocessor")
    names = list(preprocessor.get_feature_names_out()) if preprocessor is not None else selected_feature_names_from_pipeline(pipeline, X.columns.tolist())
    feature_source = "selected_biomarkers_and_covariates" if preprocessor is not None else ("selected_proteins" if "feature_selection" in pipeline.named_steps else "original_proteins")
    current: Any = X
    for _, step in pipeline.steps[:-1]:
        current = step.transform(current)
    matrix = np.asarray(current)
    if len(names) != matrix.shape[1]:
        names = [f"Protein_{i+1}" for i in range(matrix.shape[1])]
    assert_protein_level_features(names, "Regression model explanation")
    return matrix, names, feature_source


def _plot_prediction(prefix: str, model_key: str, y_true, y_pred, suffix: str) -> None:
    plot_observed_predicted(y_true, y_pred, f"{MODEL_SPECS[model_key].public_name}: Observed vs Predicted", f"{prefix}_{model_key}_{suffix}.png")


def _plot_residuals(prefix: str, model_key: str, y_true, y_pred, suffix: str) -> None:
    plot_residuals(y_true, y_pred, f"{MODEL_SPECS[model_key].public_name}: Residuals", f"{prefix}_{model_key}_{suffix}.png")


def _plot_metrics(prefix: str, model_key: str, fold_metrics: pd.DataFrame) -> None:
    plot_regression_nested_cv(fold_metrics, f"{MODEL_SPECS[model_key].public_name} Nested CV", f"{prefix}_{model_key}_metrics_line_plot.png")
    metrics = ["RMSE", "MAE", "R2"]
    values = [float(pd.to_numeric(fold_metrics[metric], errors="coerce").mean()) for metric in metrics]
    plot_metric_bars(metrics, values, f"{MODEL_SPECS[model_key].public_name} Evaluation Metrics", f"{prefix}_{model_key}_metrics.png")


def train_regression_model(
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
        raise ValueError(f"Unsupported regression model: {model_key}")
    X, y, feature_names, sample_ids, roles = load_regression_data(csv_path, covariates, categorical_covariates, return_roles=True)
    prefix = prefix or Path(csv_path).stem
    feature_selection = resolve_model_feature_selection(feature_selection, model_key, "regression")
    if model_key == "PLS_reg":
        feature_selection = "none"
    print(f"Running {MODEL_SPECS[model_key].public_name} with feature_selection={feature_selection}")

    outer = _outer_cv(len(y))
    fold_rows = []
    pred_frames = []
    all_true = []
    all_pred = []
    best_params_by_fold = []
    tuning_reports: list[dict[str, Any]] = []
    missingness_folds = []

    for fold, (train_idx, test_idx) in enumerate(outer.split(X), start=1):
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
            study_id=f"regression-{model_key}-outer-fold-{fold}",
            provenance_path=f"{prefix}_{model_key}_fold{fold}_optuna_study.json",
            tuning_reports=tuning_reports,
        )
        best_params_by_fold.append(params)
        model = build_pipeline(model_key, feature_selection, params, roles, imputation)
        model.fit(X_train, y_train)
        missingness_folds.append(model.named_steps["preprocessor"].biomarker_diagnostics(fold))
        export_outer_fold_biomarkers(prefix, model_key, fold, model, feature_names)
        y_pred = np.asarray(model.predict(X_test)).ravel()
        mse = mean_squared_error(y_test, y_pred)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = _r2(y_test, y_pred)
        fold_rows.append(
            {
                "Model": model_key,
                "Fold": fold,
                "MSE": float(mse),
                "RMSE": float(rmse),
                "MAE": float(mae),
                "R2": r2,
                "Number_of_Train_Samples": int(len(train_idx)),
                "Number_of_Test_Samples": int(len(test_idx)),
            }
        )
        fold_pred = pd.DataFrame(
            {
                "SampleID": sample_ids.iloc[test_idx].to_numpy(),
                "Original Label": y_test,
                "Predicted Label": y_pred,
                "Fold": fold,
            }
        )
        fold_pred.to_csv(f"{prefix}_{model_key}_fold{fold}_predictions.csv", index=False)
        pred_frames.append(fold_pred)
        _plot_prediction(prefix, model_key, y_test, y_pred, f"fold{fold}_prediction")
        _plot_residuals(prefix, model_key, y_test, y_pred, f"fold{fold}_residuals")
        all_true.append(y_test)
        all_pred.append(y_pred)

    y_true_all = np.concatenate(all_true)
    y_pred_all = np.concatenate(all_pred)
    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(f"{prefix}_{model_key}_outer_fold_metrics.csv", index=False)
    fold_metrics.to_csv(f"{prefix}_{model_key}_model_performance_metrics.csv", index=False)
    predictions = pd.concat(pred_frames, ignore_index=True)
    predictions.to_csv(f"{prefix}_{model_key}_predictions.csv", index=False)
    np.save(f"{prefix}_{model_key}_X.npy", X.values)
    np.save(f"{prefix}_{model_key}_y_true.npy", y_true_all)
    np.save(f"{prefix}_{model_key}_y_pred.npy", y_pred_all)
    np.save(f"{prefix}_{model_key}_feature_names.npy", np.asarray(feature_names, dtype=object))
    _plot_metrics(prefix, model_key, fold_metrics)
    _plot_prediction(prefix, model_key, y_true_all, y_pred_all, "overall_prediction")
    _plot_residuals(prefix, model_key, y_true_all, y_pred_all, "overall_residuals")

    write_missingness_qc(prefix + "_" + model_key, X, roles, missingness_folds)

    final_params = tune_params(
        model_key,
        X,
        y,
        feature_selection,
        roles=roles,
        imputation=imputation,
        study_scope="final_full_development",
        study_id=f"regression-{model_key}-final",
        provenance_path=f"{prefix}_{model_key}_final_optuna_study.json",
        tuning_reports=tuning_reports,
    )
    final_model = build_pipeline(model_key, feature_selection, final_params, roles, imputation)
    final_model.fit(X, y)
    joblib.dump(final_model, f"{prefix}_{model_key}_best_model.pkl")
    joblib.dump((X, y, roles), f"{prefix}_{model_key}_data.pkl")
    X_final, final_names, _ = _transformed_matrix(final_model, X)
    pd.DataFrame(X_final, columns=final_names).assign(Label=y).to_csv(f"{prefix}_{model_key}_transformed_features.csv", index=False)
    with open(f"{prefix}_{model_key}_best_params.json", "w", encoding="utf-8") as handle:
        json.dump({"final": final_params, "outer_folds": best_params_by_fold, "feature_selection": feature_selection, "imputation": imputation, "feature_roles": roles.__dict__, "adaptive_tuning": tuning_reports}, handle, indent=2)
    write_reproducibility_manifest(
        f"{prefix}_{model_key}_manifest.json", input_path=csv_path, task="regression", model=model_key,
        seed=RANDOM_STATE, roles=roles, imputation=imputation, final_hyperparameters=final_params,
        artifacts=[f"{prefix}_{model_key}_predictions.csv", f"{prefix}_{model_key}_best_model.pkl"],
        imputation_fallbacks=imputation_fallback_reasons(missingness_folds),
        tuning_studies=tuning_reports,
    )
    pd.DataFrame([
        {
            "Model": model_key,
            "MSE": float(mean_squared_error(y_true_all, y_pred_all)),
            "RMSE": float(math.sqrt(mean_squared_error(y_true_all, y_pred_all))),
            "MAE": float(mean_absolute_error(y_true_all, y_pred_all)),
            "R2": _r2(y_true_all, y_pred_all),
            "MetricScope": "Pooled outer-fold out-of-fold predictions",
        }
    ]).to_csv(f"{prefix}_{model_key}_final_validation_metrics.csv", index=False)

    from engines.reliability.learning_curves import generate_learning_curves

    generate_learning_curves(
        prefix,
        task="regression",
        models=[model_key],
        estimator=final_model,
        X_train=X,
        y_train=y,
    )
    print(f"Finished {MODEL_SPECS[model_key].public_name}")
    return final_model


def _cli(model_key: str) -> None:
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
    with task_run_context(args.outdir, "regression"):
        train_regression_model(args.csv, args.prefix, model_key, args.feature_selection, args.imputation, args.covariates, args.categorical_covariates)


__all__ = ["train_regression_model", "MODEL_SPECS"]
