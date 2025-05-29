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
    def __init__(self, n_components=2, max_iter=1000, tol=1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.pls = None

    def fit(self, X, y):
        self.index_ = X.index if isinstance(X, pd.DataFrame) else None
        self.pls = PLSRegression(n_components=self.n_components, max_iter=self.max_iter, tol=self.tol)
        self.pls.fit(X, y)
        return self

    def transform(self, X):
        X_new = self.pls.transform(X)
        return pd.DataFrame(
            X_new,
            index=self.index_,
            columns=[f"PLS_Component_{i+1}" for i in range(self.n_components)]
        )


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

    fig, ax = plt.subplots(figsize=(12, 8))  
    for idx, class_name in enumerate(value_columns):
        vals = shap_df_top[class_name].values
        ax.bar(x + idx * width, vals, width, label=class_name)
    
    ax.set_xticks(x + width * (len(value_columns) - 1) / 2)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    
    ax.set_xlabel("Features", fontsize=12)
    ax.set_ylabel("SHAP values", fontsize=12)
    
    ax.set_title(f"SHAP Values Bar Chart for {model}", fontsize=16)
    
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        fontsize=10,
    )
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])  
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

# ---------------------------------------------------------------------- #
# helper – figure out the feature names after the transformer pipeline   #
# ---------------------------------------------------------------------- #
def _derive_feature_names_after_transform(
    transformer: Pipeline | None,
    original_names: list[str],
) -> list[str]:
    """
    Return the column names *after* the fitted preprocessing / feature-
    selection pipeline has been applied.

    ── What it handles ───────────────────────────────────────────────────
      • SelectFromModel / ElasticNet selector  → keeps real names
      • PCA / KernelPCA / UMAP / TSNE          → Component_#
      • PLSFeatureSelector                     → PLS_Component_#
      • scalers                                → no change

    If you add new transformers later, extend the `elif` chain.
    """
    names = original_names[:]                 # start with raw names
    if transformer is None or len(transformer) == 0:
        return names                          # nothing to change

    for step_name, step_obj in transformer.steps:

        # ------------------------------------------------------------------
        # 1) dimensionality-reducing steps (checked *first* – order matters)
        # ------------------------------------------------------------------
        if isinstance(step_obj, PCA):
            names = [f"PCA_Component_{i+1}"   for i in range(step_obj.n_components_)]
            continue

        if isinstance(step_obj, KernelPCA):
            names = [f"KPCA_Component_{i+1}"  for i in range(step_obj.n_components)]
            continue

        if isinstance(step_obj, umap.UMAP):
            names = [f"UMAP_Component_{i+1}"  for i in range(step_obj.n_components)]
            continue

        if isinstance(step_obj, TSNE):
            names = [f"TSNE_Component_{i+1}"  for i in range(step_obj.n_components)]
            continue

        # your custom PLS selector
        if hasattr(step_obj, "pls"):
            n_comp = step_obj.n_components
            names  = [f"PLS_Component_{i+1}"  for i in range(n_comp)]
            continue

        # ------------------------------------------------------------------
        # 2) feature *selection* steps (keep subset of original names)
        # ------------------------------------------------------------------
        if isinstance(step_obj, SelectFromModel):
            support = step_obj.get_support()
            names   = [n for n, keep in zip(names, support) if keep]
            continue

        if hasattr(step_obj, "selector") and isinstance(step_obj.selector, ElasticNet):
            support = step_obj.selector.get_support()
            names   = [n for n, keep in zip(names, support) if keep]
            continue

        # ------------------------------------------------------------------
        # 3) scalers or anything else – do NOTHING to the name list
        # ------------------------------------------------------------------

    return names
# ---------------------------------------------------------------------- #

def process_model(model_name, prefix, num_features, n_jobs_explainer):
    """
    Process one model end-to-end: load it, apply any scaler+feature-selection,
    build the post-FS feature matrix with real names, compute SHAP on that,
    save CSV + radar/bar plots.
    """
    try:
        print(f"Processing {model_name}...")

        # 1) VAE / VAE-MLP: just read precomputed CSV if present
        if model_name in ("vae", "vaemlp"):
            shap_file = f"{prefix}_{model_name}_shap_values.csv"
            if os.path.exists(shap_file):
                shap_df = pd.read_csv(shap_file)
                feature_names = shap_df["Feature"].tolist()
                value_cols    = [c for c in shap_df.columns if c != "Feature"]
                class_names   = value_cols if len(value_cols)>1 else [""]
                print(f"  - loaded SHAP CSV for {model_name}")

                # ADD THIS BLOCK: Call plot functions explicitly!
                plot_shap_radar(model_name, shap_df, num_features, prefix, class_names)
                if len(class_names) > 1:
                    plot_shap_bar(model_name, shap_df, num_features, prefix, class_names)
                print(f"  - finished processing {model_name} (from CSV)")    
                return

            else:
                print(f"  SKIPPING {model_name}: no precomputed SHAP")
                return

        else:
            # 2) Load model pipeline + data
            best_model, X_raw, y_enc, n_classes, orig_names, class_names = \
                load_model_and_data(model_name, prefix)
            Xdf = pd.DataFrame(X_raw, columns=orig_names)

            # 3) Pull out scaler & feature_selection steps if they exist
            scaler = None
            fs     = None
            if isinstance(best_model, Pipeline):
                steps = best_model.named_steps
                scaler = steps.get("scaler", None)
                fs     = steps.get("feature_selection", None)

            # 4) Build X_shap = raw → scaler → feature_selection
            if fs is not None:
                # 4a) scaling
                X_scaled = scaler.transform(Xdf) if scaler is not None else Xdf.values

                # 4b) feature‐selection transform
                try:
                    X_fs = fs.transform(X_scaled)
                except NotImplementedError:
                    # TSNETransformer: fallback to the embedding computed at fit()
                    X_fs = fs.X_transformed_

                # 4c) derive real feature‐names
                transformer_class = fs.__class__.__name__

                # ---- ElasticNet / SelectFromModel ----
                if hasattr(fs, "get_support"):
                    # e.g. if fs is a bare SelectFromModel
                    mask = fs.get_support()
                    feat_names = [orig_names[i] for i,m in enumerate(mask) if m]
                elif hasattr(fs, "selector") and hasattr(fs.selector, "get_support"):
                    # our ElasticNetFeatureSelector
                    mask = fs.selector.get_support()
                    feat_names = [orig_names[i] for i,m in enumerate(mask) if m]

                # ---- PCA / KPCA / UMAP / PLS / TSNE ----
                elif transformer_class == "PCA":
                    feat_names = [f"PCA_Component_{i+1}" for i in range(X_fs.shape[1])]
                elif transformer_class == "KernelPCA":
                    feat_names = [f"KPCA_Component_{i+1}" for i in range(X_fs.shape[1])]
                elif "UMAP" in transformer_class:
                    feat_names = [f"UMAP_Component_{i+1}" for i in range(X_fs.shape[1])]
                elif transformer_class == "PLSFeatureSelector":
                    feat_names = [f"PLS_Component_{i+1}" for i in range(X_fs.shape[1])]
                elif transformer_class == "TSNETransformer":
                    feat_names = [f"TSNE_Component_{i+1}" for i in range(X_fs.shape[1])]

                # ---- fallback ----
                else:
                    feat_names = [f"Feature_{i+1}" for i in range(X_fs.shape[1])]

                print(f"  - applied {transformer_class}, now {X_fs.shape[1]} features")
                X_shap = np.asarray(X_fs)
                feature_names = feat_names

            else:
                # no feature selection
                print("  - no feature-selection")
                if scaler is not None:
                    X_shap = scaler.transform(Xdf)
                    print("  - applied scaler only")
                else:
                    X_shap = X_raw
                feature_names = orig_names

            # 5) strip off scaler/fs, keep just the final estimator
            final_est = best_model.steps[-1][1] if isinstance(best_model, Pipeline) else best_model

            # 6) ONLY for true tree models, if n_features_in_ mismatches, revert to raw X
            model_type = get_model_type(model_name)
            if model_type == "tree" and hasattr(final_est, "n_features_in_"):
                expected = final_est.n_features_in_
                if expected != X_shap.shape[1]:
                    print(f"  - TREE expects {expected} but we have {X_shap.shape[1]}; reverting to raw")
                    X_shap = X_raw
                    feature_names = orig_names

            # 7) compute SHAP means
            print(f"  - using SHAP explainer '{model_type}' on {X_shap.shape[1]} features")
            shap_mean = calculate_shap_values(
                final_est, X_shap, n_classes, model_type, n_jobs_explainer
            )

            # ────────────────────────────────────────────────────────────────
            #  (NEW PATCH) For GaussianNB + TSNETransformer only
            # ────────────────────────────────────────────────────────────────
            fs_classname = fs.__class__.__name__.lower() if fs is not None else ""
            if model_name == "gaussiannb" and "tsne" in fs_classname:
                print(f"  - [PATCH] Detected TSNE for GNB, truncating SHAP values")
                if shap_mean.ndim == 1:
                    shap_mean = shap_mean[:2]
                else:
                    shap_mean = shap_mean[:, :2]
                feature_names = feature_names[:2]

            # 8) build DataFrame of feature vs mean SHAP
            if shap_mean.ndim == 1:
                col = f"Mean SHAP Value for {class_names[1] if len(class_names)>1 else ''}"
                shap_df = pd.DataFrame({
                    "Feature": feature_names,
                    col: shap_mean
                })
            else:
                shap_df = pd.DataFrame({"Feature": feature_names})
                for idx, cname in enumerate(class_names):
                    shap_df[f"Mean SHAP Value for {cname}"] = shap_mean[idx, :]

            # 9) save CSV + plots
            out_csv = f"{prefix}_{model_name}_shap_values.csv"
            shap_df.to_csv(out_csv, index=False)
            print(f"  - saved SHAP values to {out_csv}")

            plot_shap_radar(model_name, shap_df, num_features, prefix, class_names)
            if len(class_names) > 1:
                plot_shap_bar(model_name, shap_df, num_features, prefix, class_names)

    except Exception as e:
        print(f"Error processing model {model_name}: {e}")
        
def _aggregate_shap(shap_vals, n_features, num_classes):
    """
    Given shap_values output by an explainer, reduce to
    mean absolute shap per feature:
      - binary: pick the positive class array or collapse appropriately
      - multiclass: mean per class per feature
    """
    # identical logic to your calculate_shap_values but only for aggregation
    if num_classes == 2:
        # shap_vals may be list or array
        if isinstance(shap_vals, list):
            arr = np.array(shap_vals[-1])
        else:
            arr = np.array(shap_vals)
            if arr.ndim == 3:
                # shape (n_samples, n_features, 2)
                arr = arr[:, :, 1]
        return np.mean(arr, axis=0)
    else:
        if isinstance(shap_vals, list):
            mats = [np.array(m).reshape(-1, n_features) for m in shap_vals]
            out = np.vstack([m.mean(axis=0) for m in mats])
        else:
            arr = np.array(shap_vals)
            if arr.ndim == 3:
                # could be (n_s, n_feat, n_classes) or (n_s, n_classes, n_feat)
                if arr.shape[1] == n_features:
                    arr = np.transpose(arr, (0, 2, 1))
                out = arr.mean(axis=0)
            else:
                raise ValueError("Unexpected shap_vals shape")
        return out


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
