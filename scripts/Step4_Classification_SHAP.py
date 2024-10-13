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
    "MLP-VAE":"vaemlp"
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
            y_pred_proba_sum = y_pred_proba.sum(axis=1, keepdims=True)
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
    
    if num_classes == 2:
        # For binary classification, use only the positive class SHAP values
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]
        shap_values = np.array(shap_values).reshape(-1, X.shape[1])
        shap_values = np.mean(np.abs(shap_values), axis=0)
    else:
        # 对于多分类，将每个类别的 SHAP 值并行计算
        def compute_class_shap(shap_values, class_idx):
            return np.mean(np.abs(shap_values[class_idx]), axis=0)
        
        shap_values = Parallel(n_jobs=n_jobs_explainer)(
            delayed(compute_class_shap)(shap_values, class_idx) for class_idx in range(num_classes)
        )
        shap_values = np.mean(shap_values, axis=0)
    
    return shap_values

def load_model_and_data(model_name, prefix):
    """
    Load pre-trained model and data.
    """
    model_path = f"{prefix}_{model_name}_model.pkl"
    data_path = f"{prefix}_{model_name}_data.pkl"
    
    if model_name == "neural_network":
        # Use pickle to load neural_network model
        with open(model_path, 'rb') as f:
            best_model = pickle.load(f)
        with open(data_path, 'rb') as f:
            X, y_encoded, le = pickle.load(f)
    else:
        # For other models, use joblib
        best_model = joblib.load(model_path)
        X, y_encoded, le = joblib.load(data_path)
    
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X = X.to_numpy()
    elif isinstance(X, pd.Series):
        feature_names = [X.name]
        X = X.values
    else:
        # If X is a NumPy array without feature names
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    
    # Ensure X is a C-contiguous NumPy array
    X = np.ascontiguousarray(X)
    
    num_classes = len(np.unique(y_encoded))
    return best_model, X, y_encoded, num_classes, feature_names

def plot_shap_radar(model, shap_values_mean, feature_names, num_features, prefix):
    """
    Plot SHAP radar chart for the top features.
    """
    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean SHAP Value': shap_values_mean
    })
    shap_df = shap_df.sort_values(by='Mean SHAP Value', ascending=False).head(num_features)

    labels = shap_df['Feature'].values
    values = shap_df['Mean SHAP Value'].values
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values = np.concatenate((values, [values[0]]))
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='orange', alpha=0.25)
    ax.plot(angles, values, color='orange', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)

    plt.title(f'SHAP Values for {model}', size=20, color='black', weight='bold')
    plt.savefig(f"{prefix}_{model}_shap_radar.png",dpi=300)
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
                feature_names = shap_df['Feature']
                shap_values_mean = shap_df['Mean SHAP Value']
                print(f"{model_name.upper()} SHAP values read from {shap_file}")
            else:
                print(f"{model_name.upper()} SHAP values file {shap_file} does not exist. Skipping {model_name.upper()}.")
                return
        else:
            print(f"Processing {model_name}...")
            best_model, X, y_encoded, num_classes, feature_names = load_model_and_data(model_name, prefix)
        
            model_type = get_model_type(model_name)
            shap_values_mean = calculate_shap_values(best_model, X, num_classes, model_type, n_jobs_explainer)

            # Save SHAP values
            shap_df = pd.DataFrame({
                'Feature': feature_names,
                'Mean SHAP Value': shap_values_mean
            })
            shap_df.to_csv(f"{prefix}_{model_name}_shap_values.csv", index=False)

            print(f'SHAP values calculated and saved to {prefix}_{model_name}_shap_values.csv')

        # Plot radar chart
        plot_shap_radar(model_name, shap_values_mean, feature_names, num_features, prefix)
        print(f"SHAP radar plot saved to {prefix}_{model_name}_shap_radar.png",dpi=300)
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
