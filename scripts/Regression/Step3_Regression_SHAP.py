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

# Import custom feature selectors for proper unpickling
from feature_selectors import PLSFeatureSelector, ElasticNetFeatureSelector, TSNETransformer

warnings.filterwarnings("ignore", message="The figure layout has changed to tight")


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

        # Create DataFrame from original features
        X_df = pd.DataFrame(X, columns=feature_names)

        # Load trained pipeline
        model_pipeline = load(model_path)
        if not isinstance(model_pipeline, Pipeline):
            print(f"Error: The loaded model '{model_name}' is not a Pipeline object.")
            return

        if 'regressor' not in model_pipeline.named_steps:
            print(f"Error: No 'regressor' step found in pipeline for '{model_name}'.")
            return

        final_regressor = model_pipeline.named_steps['regressor']

        # Build sub-pipeline for steps except the final regressor
        feature_transform_steps = []
        for step_name, step_obj in model_pipeline.steps:
            if step_name == 'regressor':
                continue
            feature_transform_steps.append((step_name, step_obj))

        # If no transform steps, pass original X
        if len(feature_transform_steps) == 0:
            X_transformed_array = X_df.values
            num_transformed_features = X_transformed_array.shape[1]
            shap_feature_names = list(feature_names)
        else:
            # We have transforms in the pipeline
            feature_transform_pipeline = Pipeline(feature_transform_steps)
            X_transformed_array = feature_transform_pipeline.transform(X_df)
            num_transformed_features = X_transformed_array.shape[1]

            # Determine feature names for SHAP
            if 'feature_selection' not in model_pipeline.named_steps:
                # e.g., PCA, KPCA, UMAP, etc.
                shap_feature_names = [f"Component_{i+1}" for i in range(num_transformed_features)]
            else:
                selection_step = model_pipeline.named_steps['feature_selection']
                if isinstance(selection_step, ElasticNetFeatureSelector):
                    mask = selection_step.selector.get_support()
                    shap_feature_names = list(feature_names[mask])
                else:
                    # PCA, KPCA, UMAP, t-SNE, PLSFeatureSelector, etc.
                    shap_feature_names = [f"Component_{i+1}" for i in range(num_transformed_features)]

        # Depending on how the final regressor was fitted, it might expect columns
        # that match feature_names_in_. We create a function that handles arrays or DataFrames.
        def predict_fs(array_input):
            # If the final regressor was fitted with named columns, we build a DataFrame
            # with matching columns. Otherwise, we pass the array directly.
            if hasattr(final_regressor, "feature_names_in_"):
                return final_regressor.predict(
                    pd.DataFrame(array_input, columns=final_regressor.feature_names_in_)
                )
            else:
                return final_regressor.predict(array_input)

        # Adjust background size for certain regressors
        if model_name == 'KNN_reg':
            background_size = min(10, X_transformed_array.shape[0])
        else:
            background_size = min(100, X_transformed_array.shape[0])

        background_indices = np.random.choice(X_transformed_array.shape[0], background_size, replace=False)
        background_array = X_transformed_array[background_indices]

        # If the final regressor needs columns, convert background_array to DataFrame
        if hasattr(final_regressor, "feature_names_in_"):
            background_df = pd.DataFrame(
                background_array,
                columns=final_regressor.feature_names_in_
            )
            explainer_data = background_df
        else:
            explainer_data = background_array

        required_max_evals = 2 * num_transformed_features + 1
        explainer = shap.PermutationExplainer(
            predict_fs,
            explainer_data,
            max_evals=required_max_evals
        )

        # SHAP for each row
        def compute_shap_one_sample(sample_idx):
            single_sample_array = X_transformed_array[[sample_idx]]
            if hasattr(final_regressor, "feature_names_in_"):
                single_sample_df = pd.DataFrame(
                    single_sample_array,
                    columns=final_regressor.feature_names_in_
                )
                shap_values_single = explainer(single_sample_df).values.flatten()
            else:
                shap_values_single = explainer(single_sample_array).values.flatten()
            return shap_values_single

        shap_values_list = Parallel(n_jobs=-1)(
            delayed(compute_shap_one_sample)(i) for i in range(X_transformed_array.shape[0])
        )
        shap_values = np.array(shap_values_list)

        # Save SHAP values
        shap_values_path = f"{output_prefix}_shap_values_{model_name}.npy"
        np.save(shap_values_path, shap_values)

        # Build summary DataFrame
        abs_shap_vals = np.abs(shap_values)
        mean_shap_vals = np.mean(abs_shap_vals, axis=0)
        shap_df = pd.DataFrame({
            "Feature": shap_feature_names,
            "Mean SHAP Value": mean_shap_vals
        }).sort_values(by="Mean SHAP Value", ascending=False)

        shap_csv_path = f"{output_prefix}_{model_name}_shap_values.csv"
        shap_df.to_csv(shap_csv_path, index=False)

        # Bar plot
        top_shap_df = shap_df.head(num_features)
        plt.figure(figsize=(15, 10))
        sns.barplot(x="Mean SHAP Value", y="Feature", data=top_shap_df)
        plt.title(f"SHAP Mean Summary Plot for {model_name}")
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_{model_name}_shap_summary_bar.png", dpi=300)
        plt.close()

        # Summary dot plot
        shap_summary_path = f"{output_prefix}_{model_name}_shap_summary_dot.png"
        plt.figure(figsize=(15, 10))
        shap.summary_plot(
            shap_values,
            features=X_transformed_array,
            feature_names=shap_feature_names,
            show=False,
            max_display=num_features
        )
        plt.title(f"SHAP Summary Dot Plot for {model_name}")
        plt.savefig(shap_summary_path, dpi=300)
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
                        help='List of model names to analyze, e.g. "Neural_Network_reg".')
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
