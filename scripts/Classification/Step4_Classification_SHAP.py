import pandas as pd
import numpy as np
import argparse
import shap
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
    QuantileTransformer,
    PowerTransformer,
)
from sklearn.cross_decomposition import PLSRegression
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
import joblib
import multiprocessing
from sklearn.linear_model import ElasticNet
from sklearn.decomposition import PCA, KernelPCA
import umap
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.naive_bayes import GaussianNB
import concurrent.futures
# Suppress all warnings
warnings.filterwarnings("ignore")

# Custom transformer for PLS
class PLSFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Modified PLSFeatureSelector to return a DataFrame with columns named
    'PLS_Component_1', 'PLS_Component_2', etc. This ensures SHAP can correctly
    handle the transformed data in a pipeline.
    """
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.pls = None
        self.feature_names_ = None

    def fit(self, X, y):
        # Store feature names if X is a DataFrame
        if hasattr(X, "columns"):
            self.feature_names_ = X.columns.tolist()
        else:
            self.feature_names_ = [f"Feature_{i}" for i in range(X.shape[1])]
        self.pls = PLSRegression(n_components=self.n_components)
        self.pls.fit(X, y)
        return self

    def transform(self, X):
        X_pls = self.pls.transform(X)
        columns = [f"PLS_Component_{i+1}" for i in range(X_pls.shape[1])]
        return pd.DataFrame(X_pls, index=range(X_pls.shape[0]), columns=columns)

# Custom transformer for t-SNE
class TSNETransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_components=2,
        perplexity=30,
        learning_rate=200,
        max_iter=1000,
        random_state=1234,
    ):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.tsne = TSNE(
            n_components=self.n_components,
            perplexity=self.perplexity,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.X_transformed_ = None

    def fit(self, X, y=None):
        self.X_transformed_ = self.tsne.fit_transform(X)
        return self

    def transform(self, X):
        # t-SNE does not support transforming new data
        if (
            self.X_transformed_ is not None
            and X.shape[0] == self.X_transformed_.shape[0]
        ):
            return self.X_transformed_
        else:
            raise NotImplementedError(
                "TSNETransformer does not support transforming new data."
            )

# Custom ElasticNet Feature Selector
class ElasticNetFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=10000, tol=1e-4):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.selector = SelectFromModel(
            ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=self.max_iter, tol=self.tol, random_state=1234)
        )

    def fit(self, X, y):
        self.selector.fit(X, y)
        return self

    def transform(self, X):
        return self.selector.transform(X)

# Define custom PLSDA classifier
class PLSDAClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.pls = PLSRegression(n_components=self.n_components)
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        y_onehot = label_binarize(y, classes=self.classes_)
        if y_onehot.ndim == 1:
            # Binary
            y_onehot = np.vstack([1 - y_onehot, y_onehot]).T
        self.pls.fit(X, y_onehot)
        return self

    def predict(self, X):
        y_pred_continuous = self.pls.predict(X)
        if y_pred_continuous.ndim == 2 and y_pred_continuous.shape[1] > 1:
            # Multi-class
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

# Model mapping
model_map = {
    "KNN": "knn",
    "RF": "random_forest",
    "NN": "neural_network",
    "SVM": "svm",
    "XGB": "xgboost",
    "PLSDA": "plsda",
    "VAE": "vae",
    "LGBM": "lightgbm",
    "LR": "logistic_regression",
    "MLPVAE": "vaemlp",
    "GNB": "gaussiannb",
}

# Define all transformers that alter feature counts
feature_selection_transformers = (
    TSNETransformer,
    PLSFeatureSelector,
    ElasticNetFeatureSelector,
    PCA,
    KernelPCA,
    umap.UMAP,
    SelectFromModel
)

# Define all scaler transformers
scaler_transformers = (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
    QuantileTransformer,
    PowerTransformer,
)

def calculate_shap_values(
    best_model, X, num_classes, model_type, n_jobs_explainer
):
    """
    Calculate SHAP values based on the model type.
    """
    if model_type == "tree":
        explainer = shap.TreeExplainer(best_model)
    elif model_type == "linear":
        explainer = shap.LinearExplainer(
            best_model, X, feature_perturbation="interventional"
        )
    elif model_type == "kernel":
        explainer = shap.KernelExplainer(
            best_model.predict_proba, shap.kmeans(X, 10)
        )
    elif model_type == "permutation":
        # Use permutation explainer with parallelism
        sample_size = min(100, X.shape[0])
        sample_X = X if X.shape[0] <= sample_size else shap.utils.sample(X, sample_size)
        explainer = shap.PermutationExplainer(
            best_model.predict_proba, sample_X, n_jobs=n_jobs_explainer
        )
    else:
        raise ValueError("Unknown model_type")

    try:
        shap_values = explainer.shap_values(X)
    except Exception as e:
        print(f"Error in SHAP explainer for model_type {model_type}: {e}")
        return np.zeros(X.shape[1])

    print(f"Type of shap_values: {type(shap_values)}")
    if isinstance(shap_values, list):
        print(f"Length of shap_values list: {len(shap_values)}")
    else:
        print(f"Shape of shap_values array: {shap_values.shape}")

    n_features = X.shape[1]

    if num_classes == 2:
        # Binary classification
        if isinstance(shap_values, list):
            if len(shap_values) == 1:
                shap_values = shap_values[0]
            elif len(shap_values) == 2:
                shap_values = shap_values[1]
            else:
                raise ValueError(
                    f"Unexpected shap_values list length for binary classification: {len(shap_values)}"
                )
            shap_values = np.array(shap_values).reshape(-1, n_features)
        elif isinstance(shap_values, np.ndarray):
            if shap_values.ndim == 3:
                if shap_values.shape[2] == num_classes:
                    shap_values = shap_values[:, :, 1]
                elif shap_values.shape[1] == num_classes and shap_values.shape[2] == n_features:
                    shap_values = np.transpose(shap_values, (0, 2, 1))
                    shap_values = shap_values[:, :, 1]
                else:
                    raise ValueError(
                        "Unexpected shape of shap_values in binary classification."
                    )
            elif shap_values.ndim == 2 and shap_values.shape[1] == n_features:
                pass
            else:
                raise ValueError(
                    "Unexpected shape of shap_values in binary classification."
                )
        else:
            raise ValueError("Unexpected type of shap_values in binary classification.")

        shap_values_mean = np.mean(shap_values, axis=0)
    else:
        # Multi-class classification
        if isinstance(shap_values, list):
            num_shap_classes = len(shap_values)
            shap_values_mean = np.zeros((num_shap_classes, n_features))
            for class_idx in range(num_shap_classes):
                class_shap_values = np.array(shap_values[class_idx]).reshape(
                    -1, n_features
                )
                shap_values_mean[class_idx, :] = np.mean(class_shap_values, axis=0)
        elif isinstance(shap_values, np.ndarray):
            if shap_values.ndim == 3:
                if shap_values.shape[1] == n_features and shap_values.shape[2] == num_classes:
                    shap_values = np.transpose(shap_values, (0, 2, 1))
                    shap_values_mean = np.mean(shap_values, axis=0)
                elif shap_values.shape[1] == num_classes and shap_values.shape[2] == n_features:
                    shap_values_mean = np.mean(shap_values, axis=0)
                else:
                    raise ValueError(
                        "Unexpected shape for shap_values in multi-class classification."
                    )
            else:
                raise ValueError(
                    "Unexpected shape for shap_values in multi-class classification."
                )
        else:
            raise ValueError(
                "Unexpected type for shap_values in multi-class classification."
            )

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

    best_model = joblib.load(model_path)
    data = joblib.load(data_path)

    if model_name == "knn":
        if len(data) != 4:
            raise ValueError(
                "Expected four elements (X, y_encoded, le, feature_names) in KNN data."
            )
        X, y_encoded, le, feature_names = data
    else:
        if len(data) != 3:
            raise ValueError(
                f"Expected three elements (X, y_encoded, le) in {model_name} data."
            )
        X, y_encoded, le = data
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X = X.to_numpy()
        elif isinstance(X, pd.Series):
            feature_names = [X.name]
            X = X.values
        else:
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            n_features = X.shape[1]
            feature_names = [f"Feature_{i}" for i in range(n_features)]

    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    elif isinstance(X, pd.Series):
        X = X.values
    else:
        if X.ndim == 1:
            X = X.reshape(-1, 1)

    X = np.ascontiguousarray(X)
    num_classes = len(np.unique(y_encoded))

    if model_name == "knn":
        class_names = le.inverse_transform(np.unique(y_encoded))
    elif hasattr(best_model, "classes_"):
        class_labels = best_model.classes_
        if hasattr(le, "inverse_transform"):
            class_names = le.inverse_transform(class_labels)
        else:
            class_names = class_labels
    else:
        class_names = [str(i) for i in range(num_classes)]

    return best_model, X, y_encoded, num_classes, feature_names, class_names

def plot_shap_radar(model, shap_df, num_features, prefix, class_names):
    """
    Plot SHAP radar chart for the top features.
    """
    feature_names = shap_df["Feature"].tolist()
    value_columns = [col for col in shap_df.columns if col != "Feature"]

    if len(value_columns) == 1:
        # Binary classification or single-value SHAP
        shap_df["Total Mean SHAP Value"] = np.abs(
            shap_df[value_columns[0]].values
        )
        shap_df_top = shap_df.sort_values(
            by="Total Mean SHAP Value", ascending=False
        ).head(num_features)

        labels = shap_df_top["Feature"].values
        values = shap_df_top[value_columns[0]].values
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        values = np.concatenate((values, [values[0]]))
        angles += angles[:1]

        fig, ax = plt.subplots(
            figsize=(10, 8), subplot_kw=dict(polar=True)
        )
        min_value = min(values)
        max_value = max(values)
        ax.set_ylim(min_value, max_value)
        num_ticks = 5
        ticks = np.linspace(min_value, max_value, num_ticks)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{tick:.2f}" for tick in ticks], fontsize=10)

        ax.plot(angles, values, color="orange", linewidth=2)
        ax.fill(angles, values, color="orange", alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=12)

        plt.title(f"SHAP Values for {model}", size=20)
        plt.savefig(
            f"{prefix}_{model}_shap_radar.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    else:
        # Multi-class classification
        shap_df["Total Mean SHAP Value"] = shap_df[value_columns].abs().sum(
            axis=1
        )
        shap_df_top = shap_df.sort_values(
            by="Total Mean SHAP Value", ascending=False
        ).head(num_features)

        labels = shap_df_top["Feature"].values
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(
            figsize=(10, 8), subplot_kw=dict(polar=True)
        )
        min_value = shap_df_top[value_columns].values.min()
        max_value = shap_df_top[value_columns].values.max()
        ax.set_ylim(min_value, max_value)
        num_ticks = 5
        ticks = np.linspace(min_value, max_value, num_ticks)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{tick:.2f}" for tick in ticks], fontsize=10)

        for class_name in value_columns:
            vals = shap_df_top[class_name].values
            vals = np.concatenate((vals, [vals[0]]))
            ax.plot(angles, vals, linewidth=2, label=class_name)
            ax.fill(angles, vals, alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=12)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=min(len(class_names), 5),
        )
        plt.title(f"SHAP Values for {model}", size=20)
        plt.subplots_adjust(bottom=0.25)
        plt.savefig(
            f"{prefix}_{model}_shap_radar.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

def plot_shap_bar(model, shap_df, num_features, prefix, class_names):
    """
    Plot SHAP bar chart for the top features in multi-class classification.
    """
    feature_names = shap_df["Feature"].tolist()
    value_columns = [
        col for col in shap_df.columns if col not in ["Feature", "Total Mean SHAP Value"]
    ]

    # Compute total mean SHAP value across classes for each feature
    shap_df["Total Mean SHAP Value"] = shap_df[value_columns].abs().sum(axis=1)
    shap_df_top = shap_df.sort_values(
        by="Total Mean SHAP Value", ascending=False
    ).head(num_features)

    labels = shap_df_top["Feature"].values
    x = np.arange(len(labels))  # the label locations
    width = 0.8 / len(value_columns)  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 8))
    for idx, class_name in enumerate(value_columns):
        vals = shap_df_top[class_name].values
        ax.bar(x + idx * width, vals, width, label=class_name)

    ax.set_xticks(x + width * (len(value_columns) - 1) / 2)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Mean SHAP Value")
    ax.set_title(f"SHAP Values Bar Chart for {model}")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=min(len(class_names), 5),
    )
    plt.tight_layout()
    plt.savefig(
        f"{prefix}_{model}_shap_bar.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

def get_model_type(model_name):
    """
    Get the SHAP explainer type based on the model name.
    """
    if model_name in ["random_forest", "xgboost", "lightgbm"]:
        return "tree"
    elif model_name in ["logistic_regression", "plsda"]:
        return "permutation"  # Use PermutationExplainer for these models
    elif model_name == "neural_network":
        return "permutation"  # Use PermutationExplainer for neural networks with parallelization
    elif model_name == "knn":
        return "permutation"  # KNN is not a tree or linear model
    else:
        return "permutation"

def determine_feature_prefix(transformer):
    """
    Returns a string prefix for feature names depending on the transformer's type.
    """
    if isinstance(transformer, TSNETransformer):
        return "TSNE_Component"
    elif isinstance(transformer, PCA):
        return "PCA_Component"
    elif isinstance(transformer, KernelPCA):
        return "KPCA_Component"
    elif isinstance(transformer, umap.UMAP):
        return "UMAP_Component"
    elif isinstance(transformer, PLSFeatureSelector):
        return "PLS_Component"
    elif isinstance(transformer, ElasticNetFeatureSelector):
        # For ElasticNetFeatureSelector, we will use actual feature names, so no prefix needed
        return ""
    else:
        # For any other or no feature selection, default to "Feature"
        return "Feature"

def process_model(model_name, prefix, num_features, n_jobs_explainer):
    """
    Process a single model to calculate and plot SHAP values.
    """
    try:
        if model_name in ["vae", "vaemlp"]:
            # Directly read VAE or VAE-MLP SHAP CSV file
            shap_file = f"{prefix}_{model_name}_shap_values.csv"
            if os.path.exists(shap_file):
                shap_df = pd.read_csv(shap_file)
                print(f"{model_name.upper()} SHAP values read from {shap_file}")
                feature_names = shap_df["Feature"].tolist()
                value_columns = [col for col in shap_df.columns if col != "Feature"]
                if len(value_columns) == 1:
                    class_names = [""]
                else:
                    class_names = value_columns
            else:
                print(
                    f"{model_name.upper()} SHAP values file {shap_file} does not exist. Skipping {model_name.upper()}."
                )
                return
        else:
            print(f"Processing {model_name}...")
            best_model, X, y_encoded, num_classes, feature_names, class_names = load_model_and_data(
                model_name, prefix
            )

            # Check if the pipeline includes any feature selection or scaler transformers
            if hasattr(best_model, "named_steps"):
                steps = best_model.named_steps
                # Identify feature selection transformers
                feature_selection_steps = [
                    step_name
                    for step_name, step_transformer in steps.items()
                    if isinstance(step_transformer, feature_selection_transformers)
                ]
                # Identify scalers
                scaler_steps = [
                    step_name
                    for step_name, step_transformer in steps.items()
                    if isinstance(step_transformer, scaler_transformers)
                ]

                # Determine if the model is neural_network with ElasticNetFeatureSelector
                is_nn_with_elasticnet = False
                for step_name in feature_selection_steps:
                    transformer = steps[step_name]
                    if (
                        isinstance(transformer, ElasticNetFeatureSelector)
                        or (isinstance(transformer, SelectFromModel) and isinstance(transformer.estimator, ElasticNet))
                    ) and model_name == "neural_network":
                        is_nn_with_elasticnet = True
                        break

                # Decide which transformers to remove
                if is_nn_with_elasticnet:
                    transformers_to_remove = feature_selection_steps  # Do not remove scalers
                else:
                    transformers_to_remove = feature_selection_steps + scaler_steps

                selected_transformer = None
                for step_name in feature_selection_steps:
                    transformer = steps[step_name]
                    selected_transformer = transformer
                    break  # Assuming only one feature selection transformer is used

                if transformers_to_remove:
                    print(
                        f"Removing transformers {transformers_to_remove} from the pipeline for SHAP explanations."
                    )
                    # Create a new pipeline without the feature selection transformers and scalers
                    new_steps = [
                        (name, transformer)
                        for name, transformer in best_model.named_steps.items()
                        if name not in transformers_to_remove
                    ]
                    if not new_steps:
                        raise ValueError(
                            "All transformers were removed from the pipeline. Cannot perform SHAP explanations."
                        )
                    modified_model = Pipeline(new_steps)
                else:
                    modified_model = best_model
                    selected_transformer = None

                # Load transformed data only if a feature selection transformer was removed
                if feature_selection_steps:
                    transformed_csv = f"{prefix}_{model_name}_transformed_X.csv"
                    if os.path.exists(transformed_csv):
                        transformed_data = pd.read_csv(transformed_csv)
                        # Drop 'SampleID' and 'Label' columns if they exist
                        for col_to_drop in ["SampleID", "Label"]:
                            if col_to_drop in transformed_data.columns:
                                transformed_data.drop(columns=[col_to_drop], inplace=True)
                        X_transformed = transformed_data.values
                        print(
                            f"Loaded transformed data from {transformed_csv} for SHAP explanations."
                        )

                        # Special handling for TSNE + GaussianNB
                        if isinstance(modified_model.steps[-1][1], GaussianNB) and isinstance(selected_transformer, TSNETransformer):
                            print("Warning: TSNE with GaussianNB is not supported for SHAP. Using original data instead.")
                            X_transformed = X
                            feature_names_transformed = feature_names
                        else:
                            if isinstance(selected_transformer, ElasticNetFeatureSelector) or (
                                isinstance(selected_transformer, SelectFromModel)
                                and isinstance(selected_transformer.estimator, ElasticNet)
                            ):
                                # Use actual feature names from the transformed CSV
                                feature_names_transformed = transformed_data.columns.tolist()
                            else:
                                prefix_for_features = determine_feature_prefix(selected_transformer)
                                if prefix_for_features:
                                    num_transformed_features = X_transformed.shape[1]
                                    feature_names_transformed = [
                                        f"{prefix_for_features}_{i+1}"
                                        for i in range(num_transformed_features)
                                    ]
                                else:
                                    feature_names_transformed = transformed_data.columns.tolist()
                    else:
                        raise FileNotFoundError(
                            f"Transformed data file {transformed_csv} does not exist."
                        )
                else:
                    # No feature selection transformer was used; use original data
                    X_transformed = X
                    feature_names_transformed = feature_names
            else:
                # If the model is not a Pipeline
                modified_model = best_model
                selected_transformer = None
                X_transformed = X
                feature_names_transformed = feature_names

            # Extract the final estimator from the modified pipeline if it's a Pipeline
            if isinstance(modified_model, Pipeline):
                final_estimator = modified_model.steps[-1][1]
            else:
                final_estimator = modified_model

            model_type = get_model_type(model_name)
            shap_values_mean = calculate_shap_values(
                final_estimator, X_transformed, num_classes, model_type, n_jobs_explainer
            )

            # Build final SHAP dataframe
            if shap_values_mean.ndim == 1:
                # Binary classification
                if len(feature_names_transformed) != len(shap_values_mean):
                    print(
                        f"Error: feature_names length ({len(feature_names_transformed)}) does not match shap_values_mean length ({len(shap_values_mean)})."
                    )
                    return
                shap_df = pd.DataFrame(
                    {
                        "Feature": feature_names_transformed,
                        f"Mean SHAP Value for {class_names[1]}": shap_values_mean,
                    }
                )
            elif shap_values_mean.ndim == 2:
                # Multi-class classification
                num_shap_classes, num_feats_shap = shap_values_mean.shape
                if num_feats_shap != len(feature_names_transformed):
                    print(
                        f"Error: feature_names length ({len(feature_names_transformed)}) does not match number of features in shap_values_mean ({num_feats_shap})."
                    )
                    return
                shap_df = pd.DataFrame({"Feature": feature_names_transformed})
                for idx, class_name in enumerate(class_names):
                    shap_df[f"Mean SHAP Value for {class_name}"] = shap_values_mean[
                        idx, :
                    ]
            else:
                print(
                    f"Unexpected shap_values_mean dimensions: {shap_values_mean.ndim}"
                )
                return

            shap_df.to_csv(f"{prefix}_{model_name}_shap_values.csv", index=False)
            print(
                f"SHAP values saved to {prefix}_{model_name}_shap_values.csv"
            )

        # Create plots
        plot_shap_radar(model_name, shap_df, num_features, prefix, class_names)
        print(
            f"SHAP radar plot saved to {prefix}_{model_name}_shap_radar.png"
        )

        if len(class_names) > 1:
            plot_shap_bar(model_name, shap_df, num_features, prefix, class_names)
            print(
                f"SHAP bar chart saved to {prefix}_{model_name}_shap_bar.png"
            )

    except Exception as e:
        print(f"Error processing model {model_name}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate SHAP values for specified models"
    )
    parser.add_argument(
        "-m", "--models", type=str, nargs="+", choices=list(model_map.keys()), required=True
    )
    parser.add_argument("-p", "--prefix", type=str, required=True)
    parser.add_argument("-f", "--num_features", type=int, required=True)
    args = parser.parse_args()

    models = [model_map[m] for m in args.models]
    prefix = args.prefix
    num_features = args.num_features

    total_cores = multiprocessing.cpu_count()
    max_parallel_models = 4
    n_jobs_explainer = max(1, total_cores // max_parallel_models)

    print(f"Total CPU cores: {total_cores}")
    print(f"Max models in parallel: {max_parallel_models}")
    print(f"Each model will use {n_jobs_explainer} threads for SHAP")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_parallel_models) as executor:
        futures = [
            executor.submit(process_model, model_name, prefix, num_features, n_jobs_explainer)
            for model_name in models
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
