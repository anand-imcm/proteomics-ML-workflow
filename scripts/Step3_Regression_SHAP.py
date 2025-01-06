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

# Import custom feature selectors for proper unpickling
from feature_selectors import PLSFeatureSelector, ElasticNetFeatureSelector, TSNETransformer

# Suppress specific matplotlib warnings
warnings.filterwarnings("ignore", message="The figure layout has changed to tight")


def calculate_shap_values(model_name, num_features, output_prefix):
    """
    Calculate and save SHAP values and plots for a given model.

    Parameters:
    - model_name (str): Name of the model (e.g., 'Neural_Network_reg').
    - num_features (int): Number of top features to display in plots.
    - output_prefix (str): Prefix for the output filenames.
    """
    try:
        # Define paths to necessary files using prefix and model name
        X_path = f"{output_prefix}_{model_name}_X.npy"
        y_true_path = f"{output_prefix}_{model_name}_y_true.npy"
        feature_names_path = f"{output_prefix}_{model_name}_feature_names.npy"
        model_path = f"{output_prefix}_{model_name}_best_model.pkl"

        # Check if all required files exist
        required_files = [X_path, y_true_path, feature_names_path, model_path]
        missing_files = [file for file in required_files if not os.path.exists(file)]
        if missing_files:
            print(f"Missing files for model '{model_name}': {missing_files}. Skipping.")
            return

        # Load data
        X = np.load(X_path)
        y_true = np.load(y_true_path)

        # Load feature names, ensuring they are strings
        feature_names = np.load(feature_names_path, allow_pickle=True)
        if isinstance(feature_names[0], bytes):
            feature_names = feature_names.astype(str)

        # Convert X to pandas DataFrame with feature names
        X_df = pd.DataFrame(X, columns=feature_names)

        # Load the trained model (Pipeline)
        model = load(model_path)

        # Ensure model has feature names for certain models like MLPRegressor
        if hasattr(model, 'feature_names_in_'):
            X_df = X_df[model.feature_names_in_]

        # To avoid "Model type not yet supported by TreeExplainer" for pipelines,
        # we will use PermutationExplainer for all models.

        # Some KNN configurations can be very slow or large-memory
        # so we can optionally reduce the background size for KNN_reg.
        if model_name == 'KNN_reg':
            background_size = min(10, X_df.shape[0])
        else:
            background_size = min(100, X_df.shape[0])

        background_indices = np.random.choice(X_df.shape[0], background_size, replace=False)
        background = X_df.iloc[background_indices]

        # Calculate required max_evals based on number of features
        num_features_in_model = X_df.shape[1]
        required_max_evals = 2 * num_features_in_model + 1

        # Create a PermutationExplainer
        explainer = shap.PermutationExplainer(model.predict, background, max_evals=required_max_evals)

        # Define a function to compute SHAP values for one sample
        def compute_shap(row):
            return explainer(row).values.flatten()

        # Parallel computation over samples
        shap_values = np.array(
            Parallel(n_jobs=-1)(
                delayed(compute_shap)(X_df.iloc[[i]]) for i in range(X_df.shape[0])
            )
        )

        # Ensure SHAP values are in the correct format
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values).mean(axis=0)

        # Save SHAP values
        shap_values_path = f"{output_prefix}_shap_values_{model_name}.npy"
        np.save(shap_values_path, shap_values)

        # Create SHAP dataframe for all features and save CSV
        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'Mean SHAP Value': np.mean(np.abs(shap_values), axis=0)
        })
        shap_csv_path = f"{output_prefix}_{model_name}_shap_values.csv"
        shap_df.to_csv(shap_csv_path, index=False)

        # Generate and save SHAP bar plot for top N features
        top_shap_df = shap_df.sort_values(by='Mean SHAP Value', ascending=False).head(num_features)
        plt.figure(figsize=(15, 10))
        sns.barplot(x='Mean SHAP Value', y='Feature', data=top_shap_df)
        plt.title(f'SHAP Mean Summary Plot for {model_name}')
        plt.savefig(f"{output_prefix}_{model_name}_shap_summary_bar.png", dpi=300)
        plt.close()

        # Generate and save SHAP summary dot plot for top N features
        shap_summary_path = f"{output_prefix}_{model_name}_shap_summary_dot.png"
        plt.figure(figsize=(15, 10))
        shap.summary_plot(shap_values, features=X_df, feature_names=feature_names, show=False, max_display=num_features)
        plt.title(f'SHAP Summary Dot Plot for {model_name}')
        plt.savefig(shap_summary_path, dpi=300)
        plt.close()

        print(f"SHAP analysis completed for model '{model_name}'.")

    except Exception as e:
        print(f"Error in model '{model_name}': {e}")
        return


def main():
    # Define argument parser
    parser = argparse.ArgumentParser(description='Calculate SHAP values for regression models.')
    parser.add_argument('--p', type=str, required=True, help='Prefix for the output filenames from regression models.')
    parser.add_argument('--m', type=str, nargs='+', required=True, help='List of model names to analyze, e.g. "Neural_Network_reg".')
    parser.add_argument('--f', type=int, default=20, help='Number of top features to display in SHAP plots.')

    # Parse arguments
    args = parser.parse_args()

    output_prefix = args.p
    model_names = args.m
    num_features = args.f

    # Run SHAP calculations in parallel for each model
    Parallel(n_jobs=-1)(
        delayed(calculate_shap_values)(model_name, num_features, output_prefix)
        for model_name in model_names
    )
    print("All SHAP analyses completed.")


if __name__ == "__main__":
    main()
