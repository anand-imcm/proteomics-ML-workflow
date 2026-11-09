"""Classification engine adapters."""

from __future__ import annotations

from importlib import import_module

from core.publication_utils import resolve_model_feature_selection

MODEL_MODULES = {
    "KNN": ("engines.classification.Step2_KNN", "knn_nested_cv", "knn"),
    "RF": ("engines.classification.Step2_RF", "random_forest_nested_cv", "random_forest"),
    "NN": ("engines.classification.Step2_NN", "neural_network_nested_cv", "neural_network"),
    "SVM": ("engines.classification.Step2_SVM", "svm_nested_cv", "svm"),
    "XGB": ("engines.classification.Step2_XGBOOST", "xgboost_nested_cv", "xgboost"),
    "PLSDA": ("engines.classification.Step2_PLSDA", "plsda_nested_cv", "plsda"),
    "LGBM": ("engines.classification.Step2_Light_GBM", "lightgbm_nested_cv", "lightgbm"),
    "LR": ("engines.classification.Step2_LR", "logistic_regression_nested_cv", "logistic_regression"),
    "GNB": ("engines.classification.Step2_GNB", "gaussian_nb_nested_cv", "gaussiannb"),
}


def run_model(model: str, csv: str, prefix: str, feature_selection: str, imputation: str = "median", covariates=None, categorical_covariates=None) -> None:
    if model == "VAE":
        from engines.deep_learning.Step2_VAE_MLP import vae_script

        vae_script(csv, prefix, imputation=imputation, covariates=covariates, categorical_covariates=categorical_covariates)
        return
    if model == "MLPVAE":
        from engines.deep_learning.Step2_MLP_in_VAE import vae

        vae(csv, prefix, imputation=imputation, covariates=covariates, categorical_covariates=categorical_covariates)
        return
    module_name, function_name, model_key = MODEL_MODULES[model]
    resolved = resolve_model_feature_selection(feature_selection, model_key, "classification")
    module = import_module(module_name)
    getattr(module, function_name)(csv, prefix, resolved, imputation=imputation, covariates=covariates, categorical_covariates=categorical_covariates)
