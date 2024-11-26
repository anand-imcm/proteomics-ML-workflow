import pandas as pd
import numpy as np
import argparse
import pickle
import shap
import os
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import warnings
from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import label_binarize
from sklearn.cross_decomposition import PLSRegression
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib
import multiprocessing
from joblib import Parallel, delayed

# Model mapping
model_map = {
    "RF": "random_forest",
    "KNN": "knn",
    "NN": "neural_network",
    "SVM": "svm",
    "XGB": "xgboost",
    "PLSDA": "plsda",
    "VAE": "vae",
    "LGBM": "lightgbm",
    "LogisticRegression": "logistic_regression",
    "GaussianNB": "gaussiannb",
    "MLP-VAE": "vaemlp"
}

# Suppress all warnings
warnings.filterwarnings('ignore')

# Define custom PLSDA classifier
class PLSDAClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.pls = PLSRegression(n_components=self.n_components)
        self.classes_ = None

    def fit(self, X, y):
        # Save class labels
        self.classes_ = np.unique(y)
        y_onehot = label_binarize(y, classes=self.classes_)
        if y_onehot.ndim == 1:
            y_onehot = np.vstack([1 - y_onehot, y_onehot]).T
        self.pls.fit(X, y_onehot)
        return self

    def predict(self, X):
        y_pred_continuous = self.pls.predict(X)
        if y_pred_continuous.ndim == 2 and y_pred_continuous.shape[1] > 1:
            y_pred = self.classes_[np.argmax(y_pred_continuous, axis=1)]
        else:
            y_pred = (y_pred_continuous >= 0.5).astype(int).ravel()
        return y_pred

    def predict_proba(self, X):
        y_pred_continuous = self.pls.predict(X)
        y_pred_proba = np.maximum(y_pred_continuous, 0)
        if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] > 1:
            y_pred_proba_sum = y_pred_proba.sum(axis=1, keepdims=True)
            y_pred_proba = y_pred_proba / y_pred_proba_sum
        else:
            y_pred_proba = np.hstack([1 - y_pred_proba, y_pred_proba])
        return y_pred_proba

def calculate_shap_values(best_model, X, num_classes, model_type, n_jobs_explainer):
    """
    Calculate SHAP values based on the model type.
    """
    if model_type == "tree":
        explainer = shap.TreeExplainer(best_model)
    elif model_type == "linear":
        explainer = shap.LinearExplainer(best_model, X, feature_perturbation="interventional")
    elif model_type == "kernel":
        explainer = shap.KernelExplainer(best_model.predict_proba, shap.kmeans(X, 10))
    elif model_type == "permutation":
        # Use permutation explainer with controlled parallelism
        sample_size = min(100, X.shape[0])
        sample_X = X if X.shape[0] <= sample_size else shap.utils.sample(X, sample_size)
        explainer = shap.PermutationExplainer(best_model.predict_proba, sample_X, n_jobs=n_jobs_explainer)
    else:
        raise ValueError("Unknown model_type")

    # Calculate SHAP values
    try:
        shap_values = explainer.shap_values(X)
    except Exception as e:
        print(f"Error in SHAP explainer for model_type {model_type}: {e}")
        return np.zeros(X.shape[1])

    # Debug statements
    print(f"Type of shap_values: {type(shap_values)}")
    if isinstance(shap_values, list):
        print(f"Length of shap_values list: {len(shap_values)}")
    else:
        print(f"Shape of shap_values array: {shap_values.shape}")

    n_features = X.shape[1]

    if num_classes == 2:
        # For binary classification
        if isinstance(shap_values, list):
            if len(shap_values) == 1:
                # Use the single array
                shap_values = shap_values[0]
            elif len(shap_values) == 2:
                # Use the positive class
                shap_values = shap_values[1]
            else:
                raise ValueError(f"Unexpected shap_values list length for binary classification: {len(shap_values)}")
            shap_values = np.array(shap_values).reshape(-1, n_features)
        elif isinstance(shap_values, np.ndarray):
            # shap_values is an array
            if shap_values.ndim == 3:
                if shap_values.shape[2] == num_classes:
                    # shap_values has shape (n_samples, n_features, n_classes)
                    # Use the positive class
                    shap_values = shap_values[:, :, 1]
                elif shap_values.shape[1] == num_classes and shap_values.shape[2] == n_features:
                    # shap_values has shape (n_samples, n_classes, n_features)
                    # Transpose to (n_samples, n_features, n_classes)
                    shap_values = np.transpose(shap_values, (0, 2, 1))
                    shap_values = shap_values[:, :, 1]
                else:
                    raise ValueError("Unexpected shape of shap_values in binary classification.")
            elif shap_values.ndim == 2 and shap_values.shape[1] == n_features:
                # Correct shape
                pass
            else:
                raise ValueError("Unexpected shape of shap_values in binary classification.")
        else:
            raise ValueError("Unexpected type of shap_values in binary classification.")
        shap_values_mean = np.mean(shap_values, axis=0)
    else:
        # For multi-class classification
        if isinstance(shap_values, list):
            num_shap_classes = len(shap_values)
            # Initialize an array to hold per-class mean SHAP values
            shap_values_mean = np.zeros((num_shap_classes, n_features))
            for class_idx in range(num_shap_classes):
                class_shap_values = np.array(shap_values[class_idx]).reshape(-1, n_features)
                # Compute mean SHAP value per feature for this class
                shap_values_mean[class_idx, :] = np.mean(class_shap_values, axis=0)
        elif isinstance(shap_values, np.ndarray):
            # shap_values is an array
            if shap_values.ndim == 3:
                if shap_values.shape[1] == n_features and shap_values.shape[2] == num_classes:
                    # shap_values has shape (n_samples, n_features, n_classes)
                    # Transpose to (n_samples, n_classes, n_features)
                    shap_values = np.transpose(shap_values, (0, 2, 1))  # Now shape is (n_samples, n_classes, n_features)
                    shap_values_mean = np.mean(shap_values, axis=0)  # Shape (n_classes, n_features)
                elif shap_values.shape[1] == num_classes and shap_values.shape[2] == n_features:
                    # shap_values has shape (n_samples, n_classes, n_features)
                    shap_values_mean = np.mean(shap_values, axis=0)  # Shape (n_classes, n_features)
                else:
                    raise ValueError("Unexpected shape for shap_values in multi-class classification.")
            else:
                raise ValueError("Unexpected shape for shap_values in multi-class classification.")
        else:
            raise ValueError("Unexpected type for shap_values in multi-class classification.")

    # Print shapes for debugging
    print(f"SHAP input X shape: {X.shape}")
    if shap_values_mean.ndim == 1:
        print(f"Computed SHAP values mean shape: {shap_values_mean.shape}")
    else:
        print(f"Computed SHAP values mean shape: {shap_values_mean.shape}")

    return shap_values_mean

def load_model_and_data(model_name, prefix):
    """
    Load pre-trained model and data.
    """
    model_path = f"{prefix}_{model_name}_model.pkl"
    data_path = f"{prefix}_{model_name}_data.pkl"

    # Load model
    best_model = joblib.load(model_path)
    data = joblib.load(data_path)

    if model_name == "knn":
        if len(data) == 4:
            X, y_encoded, le, feature_names = data
        else:
            raise ValueError("Expected four elements (X, y_encoded, le, feature_names) in KNN data.")
    else:
        if len(data) == 3:
            X, y_encoded, le = data
        else:
            raise ValueError(f"Expected three elements (X, y_encoded, le) in {model_name} data.")
        # Get feature names
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X = X.to_numpy()
        elif isinstance(X, pd.Series):
            feature_names = [X.name]
            X = X.values
        else:
            # If X is a NumPy array without feature names
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            n_features = X.shape[1]
            feature_names = [f"Feature_{i}" for i in range(n_features)]

    # Ensure X is a C-contiguous NumPy array
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    elif isinstance(X, pd.Series):
        X = X.values
    else:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
    X = np.ascontiguousarray(X)

    num_classes = len(np.unique(y_encoded))

    # Get actual class names
    if model_name == "knn":
        class_names = le.inverse_transform(np.unique(y_encoded))
    elif hasattr(best_model, 'classes_'):
        class_names = best_model.classes_
        if hasattr(le, 'inverse_transform'):
            class_names = le.inverse_transform(class_names)
    else:
        class_names = [str(i) for i in range(num_classes)]
    
    return best_model, X, y_encoded, num_classes, feature_names, class_names

def plot_shap_radar(model, shap_df, num_features, prefix, class_names):
    """
    Plot SHAP radar chart for the top features.
    """
    feature_names = shap_df['Feature'].tolist()
    value_columns = [col for col in shap_df.columns if col != 'Feature']

    if len(value_columns) == 1:
        # Binary classification or regression
        shap_values_mean = shap_df[value_columns[0]].values
        shap_df['Total Mean SHAP Value'] = np.abs(shap_values_mean)
        shap_df_top = shap_df.sort_values(by='Total Mean SHAP Value', ascending=False).head(num_features)

        labels = shap_df_top['Feature'].values
        values = shap_df_top[value_columns[0]].values
        num_vars = len(labels)

        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        values = np.concatenate((values, [values[0]]))
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

        # Set the coordinate limits
        min_value = min(values)
        max_value = max(values)
        ax.set_ylim(min_value, max_value)

        # Set radial ticks
        num_ticks = 5  # You can adjust the number of ticks as needed
        ticks = np.linspace(min_value, max_value, num_ticks)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{tick:.2f}" for tick in ticks], fontsize=10)

        ax.plot(angles, values, color='orange', linewidth=2)
        ax.fill(angles, values, color='orange', alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=12)

        plt.title(f'SHAP Values for {model}', size=20, color='black', weight='bold')

        # Save figure
        plt.savefig(f"{prefix}_{model}_shap_radar.png", dpi=300, bbox_inches='tight')
        plt.close()

    else:
        # Multi-class classification
        num_classes = len(value_columns)
        # Compute total mean SHAP value across classes for each feature
        shap_df['Total Mean SHAP Value'] = shap_df[value_columns].abs().sum(axis=1)
        shap_df_top = shap_df.sort_values(by='Total Mean SHAP Value', ascending=False).head(num_features)

        labels = shap_df_top['Feature'].values
        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

        # Set the coordinate limits based on min and max SHAP values of the selected features
        min_value = shap_df_top[value_columns].values.min()
        max_value = shap_df_top[value_columns].values.max()
        ax.set_ylim(min_value, max_value)

        # Set radial ticks
        num_ticks = 5
        ticks = np.linspace(min_value, max_value, num_ticks)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{tick:.2f}" for tick in ticks], fontsize=10)

        # Plot per-class SHAP values
        for class_name in value_columns:
            values = shap_df_top[class_name].values
            values = np.concatenate((values, [values[0]]))
            ax.plot(angles, values, linewidth=2, label=class_name)
            ax.fill(angles, values, alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=12)

        # Place legend at the bottom center
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=min(len(class_names), 5))

        plt.title(f'SHAP Values for {model}', size=20, color='black', weight='bold')

        # Adjust layout to make space for the legend
        plt.subplots_adjust(bottom=0.25)

        # Save figure
        plt.savefig(f"{prefix}_{model}_shap_radar.png", dpi=300, bbox_inches='tight')
        plt.close()

def get_model_type(model_name):
    """
    Get the SHAP explainer type based on the model name.
    """
    if model_name in ['random_forest', 'xgboost', 'lightgbm']:
        return "tree"
    elif model_name in ['logistic_regression', 'plsda']:
        return "permutation"  # Use PermutationExplainer for these models
    elif model_name == "neural_network":
        return "permutation"  # Use PermutationExplainer for neural networks with parallelization
    else:
        return "permutation"

def process_model(model_name, prefix, num_features, n_jobs_explainer):
    """
    Process a single model to calculate and plot SHAP values.
    """
    try:
        if model_name in ['vae', 'vaemlp']:
            # Directly read VAE or VAE-MLP SHAP CSV file
            shap_file = f"{prefix}_{model_name}_shap_values.csv"
            if os.path.exists(shap_file):
                shap_df = pd.read_csv(shap_file)
                print(f"{model_name.upper()} SHAP values read from {shap_file}")
                feature_names = shap_df['Feature'].tolist()
                # Check columns
                value_columns = [col for col in shap_df.columns if col != 'Feature']
                # Determine class names
                if len(value_columns) == 1:
                    class_names = ['']
                else:
                    class_names = value_columns  # Use the column names as class names
            else:
                print(f"{model_name.upper()} SHAP values file {shap_file} does not exist. Skipping {model_name.upper()}.")
                return
        else:
            print(f"Processing {model_name}...")
            best_model, X, y_encoded, num_classes, feature_names, class_names = load_model_and_data(model_name, prefix)

            model_type = get_model_type(model_name)
            shap_values_mean = calculate_shap_values(best_model, X, num_classes, model_type, n_jobs_explainer)

            # Ensure feature_names and shap_values_mean have compatible shapes
            if shap_values_mean.ndim == 1:
                # Binary classification
                if len(feature_names) != len(shap_values_mean):
                    print(f"Error: feature_names length ({len(feature_names)}) does not match shap_values_mean length ({len(shap_values_mean)}).")
                    return
                # Save SHAP values
                shap_df = pd.DataFrame({
                    'Feature': feature_names,
                    f'Mean SHAP Value for {class_names[1]}': shap_values_mean
                })
            elif shap_values_mean.ndim == 2:
                # Multi-class classification
                num_classes_in_shap, num_features_in_shap = shap_values_mean.shape
                if num_features_in_shap != len(feature_names):
                    print(f"Error: feature_names length ({len(feature_names)}) does not match number of features in shap_values_mean ({num_features_in_shap}).")
                    return
                # Create DataFrame
                shap_df = pd.DataFrame({'Feature': feature_names})
                for idx, class_name in enumerate(class_names):
                    shap_df[f'Mean SHAP Value for {class_name}'] = shap_values_mean[idx, :]
            else:
                print(f"Unexpected shap_values_mean dimensions: {shap_values_mean.ndim}")
                return

            shap_df.to_csv(f"{prefix}_{model_name}_shap_values.csv", index=False)

            print(f'SHAP values calculated and saved to {prefix}_{model_name}_shap_values.csv')

        # Plot radar chart
        plot_shap_radar(model_name, shap_df, num_features, prefix, class_names)
        print(f"SHAP radar plot saved to {prefix}_{model_name}_shap_radar.png")
    except Exception as e:
        print(f"Error processing model {model_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Calculate SHAP values for specified models')
    parser.add_argument('-m', '--models', type=str, nargs='+', choices=list(model_map.keys()), help='Name of the model(s)', required=True)
    parser.add_argument('-p', '--prefix', type=str, required=True, help='Output prefix')
    parser.add_argument('-f', '--num_features', type=int, required=True, help='Number of top features to display in radar chart')
    args = parser.parse_args()

    models = [model_map[model] for model in args.models]
    prefix = args.prefix
    num_features = args.num_features

    # Determine the number of available CPU cores
    total_cores = multiprocessing.cpu_count()
    print(f"Total CPU cores available: {total_cores}")

    # Decide on the number of processes based on CPU cores
    # Reserve some cores for SHAP's internal parallelism
    reserved_cores = 2  # Adjust based on your system
    max_workers = max(1, total_cores - reserved_cores)
    print(f"Using {max_workers} parallel processes for model SHAP calculations.")

    # Calculate n_jobs for SHAP explainer to avoid overloading
    n_jobs_explainer = max(1, reserved_cores)

    # Process all selected models in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_model, model_name, prefix, num_features, n_jobs_explainer) for model_name in models]
        for future in futures:
            future.result()  # Ensure all models are processed

if __name__ == '__main__':
    main()