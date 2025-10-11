import numpy as np
import pandas as pd
import shap
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from joblib import Parallel, delayed, load
import argparse
import warnings
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, KernelPCA

# Try to import wrappers used in regression for type checks; fall back to None if unavailable
try:
    from wrappers import UMAPPipelineWrapper, TSNEPipelineWrapper
except Exception:
    UMAPPipelineWrapper = None
    TSNEPipelineWrapper = None

# Import custom feature selectors for proper unpickling
from feature_selectors import PLSFeatureSelector, ElasticNetFeatureSelector, TSNETransformer

# Global plotting style
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})

# Be conservative: silence only the specific FutureWarning text from sklearn Pipeline if any slip through.
warnings.filterwarnings(
    "ignore",
    message="This Pipeline instance is not fitted yet. Call 'fit' with appropriate arguments.*",
    category=FutureWarning
)
warnings.filterwarnings("ignore", message="The figure layout has changed to tight")


def _apply_fitted_steps_sequentially(X, steps):
    """
    Apply each fitted transform step sequentially without wrapping them in a new Pipeline.
    This avoids sklearn Pipeline's 'not fitted yet' FutureWarning.
    """
    X_cur = X
    for name, step in steps:
        # Some steps expect pandas with column names; others accept numpy
        # Try DataFrame-based transform first; fallback to ndarray if needed.
        try:
            X_cur = step.transform(X_cur)
        except Exception:
            X_cur = step.transform(np.asarray(X_cur))
    return X_cur


def calculate_shap_values(model_name, num_features, output_prefix):
    try:
        # Paths
        X_path = f"{output_prefix}_{model_name}_X.npy"
        y_true_path = f"{output_prefix}_{model_name}_y_true.npy"
        feature_names_path = f"{output_prefix}_{model_name}_feature_names.npy"
        model_path = f"{output_prefix}_{model_name}_best_model.pkl"

        required_files = [X_path, y_true_path, feature_names_path, model_path]
        missing_files = [file for file in required_files if not os.path.exists(file)]
        if missing_files:
            print(f"Missing files for model '{model_name}': {missing_files}. Skipping.")
            return

        # Load data
        X = np.load(X_path)
        _ = np.load(y_true_path)  # y_true not strictly needed for SHAP
        feature_names = np.load(feature_names_path, allow_pickle=True)
        if isinstance(feature_names[0], bytes):
            feature_names = feature_names.astype(str)

        # Create DataFrame from original features (keep names)
        X_df = pd.DataFrame(X, columns=feature_names)

        # Load trained pipeline (fitted)
        model_pipeline = load(model_path)
        if not isinstance(model_pipeline, Pipeline):
            print(f"Error: The loaded model '{model_name}' is not a Pipeline object.")
            return

        if "regressor" not in model_pipeline.named_steps:
            print(f"Error: No 'regressor' step found in pipeline for '{model_name}'.")
            return

        final_regressor = model_pipeline.named_steps["regressor"]

        # All steps except the final regressor
        feature_transform_steps = [(n, s) for (n, s) in model_pipeline.steps if n != "regressor"]

        # Transform X by all non-regressor steps sequentially (no new Pipeline object)
        if len(feature_transform_steps) == 0:
            X_transformed_array = X_df.values
        else:
            X_transformed_array = _apply_fitted_steps_sequentially(X_df, feature_transform_steps)
            # Ensure ndarray
            if isinstance(X_transformed_array, pd.DataFrame):
                X_transformed_array = X_transformed_array.values
            else:
                X_transformed_array = np.asarray(X_transformed_array)

        num_transformed_features = X_transformed_array.shape[1]

        # Determine SHAP feature names based on the feature_selection step
        selection_step = model_pipeline.named_steps.get("feature_selection", None)

        if selection_step is None:
            shap_feature_names = list(feature_names)
        elif isinstance(selection_step, ElasticNetFeatureSelector):
            mask = selection_step.selector.get_support()
            shap_feature_names = list(feature_names[mask])
        elif isinstance(selection_step, PLSFeatureSelector):
            shap_feature_names = [f"PLS_Component_{i+1}" for i in range(num_transformed_features)]
        elif isinstance(selection_step, PCA):
            shap_feature_names = [f"PCA_Component_{i+1}" for i in range(num_transformed_features)]
        elif isinstance(selection_step, KernelPCA):
            shap_feature_names = [f"KPCA_Component_{i+1}" for i in range(num_transformed_features)]
        elif (UMAPPipelineWrapper is not None) and isinstance(selection_step, UMAPPipelineWrapper):
            shap_feature_names = [f"UMAP_Component_{i+1}" for i in range(num_transformed_features)]
        elif (TSNEPipelineWrapper is not None) and isinstance(selection_step, TSNEPipelineWrapper):
            shap_feature_names = [f"TSNE_Component_{i+1}" for i in range(num_transformed_features)]
        elif isinstance(selection_step, TSNETransformer):
            shap_feature_names = [f"TSNE_Component_{i+1}" for i in range(num_transformed_features)]
        else:
            shap_feature_names = [f"Component_{i+1}" for i in range(num_transformed_features)]

        # Final consistency check: match names length to transformed feature count
        if len(shap_feature_names) != num_transformed_features:
            shap_feature_names = [f"Feature_{i+1}" for i in range(num_transformed_features)]

        # Final regressor prediction wrapper
        def predict_fs(array_input):
            if hasattr(final_regressor, "feature_names_in_"):
                # Align columns to what regressor expects
                df_in = pd.DataFrame(array_input, columns=final_regressor.feature_names_in_)
                return final_regressor.predict(df_in)
            else:
                return final_regressor.predict(array_input)

        # Background sampling (smaller for KNN)
        rng = np.random.default_rng(42)
        if model_name == "KNN_reg":
            background_size = min(10, X_transformed_array.shape[0])
        else:
            background_size = min(100, X_transformed_array.shape[0])
        background_indices = rng.choice(X_transformed_array.shape[0], background_size, replace=False)
        background_array = X_transformed_array[background_indices]

        # Build explainer background in the right type (DataFrame if regressor wants named columns)
        if hasattr(final_regressor, "feature_names_in_"):
            background_df = pd.DataFrame(background_array, columns=final_regressor.feature_names_in_)
            explainer_data = background_df
        else:
            explainer_data = background_array

        # PermutationExplainer evals
        required_max_evals = 2 * num_transformed_features + 1
        explainer = shap.PermutationExplainer(
            predict_fs,
            explainer_data,
            max_evals=required_max_evals
        )

        # SHAP for full transformed X
        if hasattr(final_regressor, "feature_names_in_"):
            X_full_for_pred = pd.DataFrame(
                X_transformed_array,
                columns=final_regressor.feature_names_in_
            )
        else:
            X_full_for_pred = X_transformed_array

        shap_values_obj = explainer(X_full_for_pred)
        shap_values = shap_values_obj.values  # shape: (n_samples, n_features)

        # Save raw SHAP values
        shap_values_path = f"{output_prefix}_shap_values_{model_name}.npy"
        np.save(shap_values_path, shap_values)

        # Summary DataFrame (mean |SHAP|)
        abs_shap_vals = np.abs(shap_values)
        mean_shap_vals = np.mean(abs_shap_vals, axis=0)
        shap_df = pd.DataFrame({
            "Feature": shap_feature_names,
            "Mean SHAP Value": mean_shap_vals
        }).sort_values(by="Mean SHAP Value", ascending=False)

        shap_csv_path = f"{output_prefix}_{model_name}_shap_values.csv"
        shap_df.to_csv(shap_csv_path, index=False)

        # Bar plot (top-k)
        top_shap_df = shap_df.head(num_features)
        plt.figure(figsize=(15, 10))
        ax = sns.barplot(x="Mean SHAP Value", y="Feature", data=top_shap_df)
        ax.set_title(f"SHAP Mean Summary Plot for {model_name}", fontsize=18)
        ax.set_xlabel("Mean SHAP Value", fontsize=16)
        ax.set_ylabel("Feature", fontsize=16)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_{model_name}_shap_summary_bar.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Beeswarm (summary dot) plot with title
        plt.figure(figsize=(15, 10))
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The NumPy global RNG was seeded by calling `np.random.seed`",
                category=FutureWarning
            )
            shap.summary_plot(
                shap_values,
                features=X_transformed_array,
                feature_names=shap_feature_names,
                show=False,
                max_display=num_features
            )
        plt.title(f"SHAP Summary Dot Plot for {model_name}", fontsize=18)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_{model_name}_shap_summary_dot.png", dpi=300, bbox_inches="tight")
        plt.close()

        print(f"SHAP analysis completed for model '{model_name}'.")

    except Exception as e:
        print(f"Error in model '{model_name}': {e}")
        return


def main():
    parser = argparse.ArgumentParser(description="Calculate SHAP values for regression models.")
    parser.add_argument("--p", type=str, required=True,
                        help="Prefix for the output filenames from regression models.")
    parser.add_argument("--m", type=str, nargs="+", required=True,
                        help='List of model names to analyze, e.g. "NN_reg".')
    parser.add_argument("--f", type=int, default=20,
                        help="Number of top components (features) to display in SHAP plots.")

    args = parser.parse_args()
    output_prefix = args.p
    num_features = args.f

    model_aliases = {
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

    user_choices = args.m
    model_names = list(model_aliases.keys())
    for i, m in enumerate(user_choices):
        for choice in model_names:
            if m.casefold() == choice.casefold():
                user_choices[i] = model_aliases[choice]

    Parallel(n_jobs=-1)(
        delayed(calculate_shap_values)(model_name, num_features, output_prefix)
        for model_name in user_choices
    )
    print("All SHAP analyses completed.")


if __name__ == "__main__":
    main()
