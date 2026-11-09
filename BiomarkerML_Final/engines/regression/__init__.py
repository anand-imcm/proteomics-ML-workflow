"""Regression engine adapters."""

from __future__ import annotations

from core.publication_utils import resolve_model_feature_selection

MODEL_ALIASES = {
    "NN_reg": "Neural_Network_reg",
    "RF_reg": "Random_Forest_reg",
    "SVM_reg": "SVM_reg",
    "XGB_reg": "XGBoost_reg",
    "PLS_reg": "PLS_reg",
    "KNN_reg": "KNN_reg",
    "LGBM_reg": "LightGBM_reg",
}
VAE_MODELS = {"VAE_reg", "MLPVAE_reg"}


def run_model(model: str, csv: str, prefix: str, feature_selection: str, imputation: str = "median", covariates=None, categorical_covariates=None) -> None:
    normalized = next((choice for choice in list(MODEL_ALIASES) + list(VAE_MODELS) if model.casefold() == choice.casefold()), model)
    if normalized == "VAE_reg":
        from engines.deep_learning.Step2_VAE_MLP_reg import vae

        vae(csv, prefix, imputation=imputation, covariates=covariates, categorical_covariates=categorical_covariates)
        return
    if normalized == "MLPVAE_reg":
        from engines.deep_learning.Step2_MLP_in_VAE_reg import vae_regression

        vae_regression(csv, prefix, imputation=imputation, covariates=covariates, categorical_covariates=categorical_covariates)
        return
    if normalized not in MODEL_ALIASES:
        raise ValueError(f"Unsupported regression model: {model}")
    from engines.regression.protein_regression import train_regression_model

    model_key = MODEL_ALIASES[normalized]
    resolved = resolve_model_feature_selection(feature_selection, model_key, "regression")
    train_regression_model(csv, prefix, model_key, resolved, imputation, covariates, categorical_covariates)
