import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import ElasticNet
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    f1_score,
    accuracy_score,
    ConfusionMatrixDisplay,
)
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import warnings
import sys
import contextlib
import joblib
import optuna
from optuna.samplers import TPESampler
from sklearn.exceptions import ConvergenceWarning
import umap
from scipy.sparse.linalg import ArpackError

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# Suppress output
class SuppressOutput(contextlib.AbstractContextManager):
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = open('/dev/null', 'w')
        sys.stderr = open('/dev/null', 'w')
    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._stdout
        sys.stderr = self._stderr

# Custom transformer for PLS
class PLSFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.pls = None  # Initialize in fit

    def fit(self, X, y):
        self.pls = PLSRegression(n_components=self.n_components)
        self.pls.fit(X, y)
        return self

    def transform(self, X):
        return self.pls.transform(X)[0]

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
        if self.X_transformed_ is not None and X.shape[0] == self.X_transformed_.shape[0]:
            return self.X_transformed_
        else:
            raise NotImplementedError("TSNETransformer does not support transforming new data.")

def knn_nested_cv(inp, prefix, feature_selection_method):
    # Read data
    data = pd.read_csv(inp)

    # Ensure 'SampleID' and 'Label' columns exist
    if 'SampleID' not in data.columns or 'Label' not in data.columns:
        raise ValueError("Input data must contain 'SampleID' and 'Label' columns.")

    # Extract SampleID
    sample_ids = data['SampleID']

    # Data processing
    X = data.drop(columns=['SampleID', 'Label']).copy()
    y = data['Label']

    # Apply data standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    # Convert target variable to categorical
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_binarized = pd.get_dummies(y_encoded).values
    num_classes = len(np.unique(y_encoded))

    # Save all possible principal component information
    if feature_selection_method == 'pca':
        full_pca = PCA(n_components=X.shape[1], random_state=1234)
        X_pca_full = full_pca.fit_transform(X)
        explained_variance = full_pca.explained_variance_ratio_
        explained_variance_df = pd.DataFrame({
            'Component': range(1, len(explained_variance)+1),
            'Explained Variance Ratio': explained_variance
        })
        explained_variance_df.to_csv(f"{prefix}_knn_pca_explained_variance_full.csv", index=False)
        # Save all components' transformed data
        X_pca_full_df = pd.DataFrame(X_pca_full, columns=[f"PCA_Component_{i+1}" for i in range(X_pca_full.shape[1])])
        X_pca_full_df.insert(0, 'SampleID', sample_ids)
        X_pca_full_df['Label'] = y
        X_pca_full_df.to_csv(f"{prefix}_knn_pca_all_components.csv", index=False)
    elif feature_selection_method == 'pls':
        pls = PLSRegression(n_components=min(X.shape[0]-1, X.shape[1]))
        with SuppressOutput():
            X_pls_full = pls.fit_transform(X, y_encoded)[0]
        explained_variance = np.var(X_pls_full, axis=0) / np.var(X, axis=0).sum()
        explained_variance_df = pd.DataFrame({
            'Component': range(1, len(explained_variance)+1),
            'Explained Variance Ratio': explained_variance
        })
        explained_variance_df.to_csv(f"{prefix}_knn_pls_explained_variance_full.csv", index=False)
        # Save all components' transformed data
        X_pls_full_df = pd.DataFrame(X_pls_full, columns=[f"PLS_Component_{i+1}" for i in range(X_pls_full.shape[1])])
        X_pls_full_df.insert(0, 'SampleID', sample_ids)
        X_pls_full_df['Label'] = y
        X_pls_full_df.to_csv(f"{prefix}_knn_pls_all_components.csv", index=False)
    elif feature_selection_method == 'kpca':
        n_samples = X.shape[0]
        n_features = X.shape[1]
        max_kpca_components = min(n_features, n_samples -1)
        kpca = KernelPCA(n_components=max_kpca_components, kernel='rbf', gamma=1.0, random_state=1234, eigen_solver='arpack', max_iter=5000)
        with SuppressOutput():
            try:
                X_kpca_full = kpca.fit_transform(X)
            except ArpackError as e:
                print(f"KernelPCA fitting failed: {e}")
                X_kpca_full = np.zeros((X.shape[0], max_kpca_components))
        # Save transformed data
        X_kpca_full_df = pd.DataFrame(X_kpca_full, columns=[f"KPCA_Component_{i+1}" for i in range(X_kpca_full.shape[1])])
        X_kpca_full_df.insert(0, 'SampleID', sample_ids)
        X_kpca_full_df['Label'] = y
        X_kpca_full_df.to_csv(f"{prefix}_knn_kpca_all_components.csv", index=False)
    elif feature_selection_method == 'umap':
        umap_full = umap.UMAP(n_components=min(X.shape[1], 100), n_neighbors=15, min_dist=0.1, random_state=1234)
        with SuppressOutput():
            X_umap_full = umap_full.fit_transform(X)
        # Save transformed data
        X_umap_full_df = pd.DataFrame(X_umap_full, columns=[f"UMAP_Component_{i+1}" for i in range(X_umap_full.shape[1])])
        X_umap_full_df.insert(0, 'SampleID', sample_ids)
        X_umap_full_df['Label'] = y
        X_umap_full_df.to_csv(f"{prefix}_knn_umap_all_components.csv", index=False)
    elif feature_selection_method == 'tsne':
        tsne_full = TSNE(n_components=2, perplexity=30, learning_rate=200, max_iter=1000, random_state=1234)
        with SuppressOutput():
            X_tsne_full = tsne_full.fit_transform(X)
        # Save transformed data
        X_tsne_full_df = pd.DataFrame(X_tsne_full, columns=[f"TSNE_Component_{i+1}" for i in range(X_tsne_full.shape[1])])
        X_tsne_full_df.insert(0, 'SampleID', sample_ids)
        X_tsne_full_df['Label'] = y
        X_tsne_full_df.to_csv(f"{prefix}_knn_tsne_all_components.csv", index=False)

    # Define outer cross-validation strategy
    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

    # Lists to store outer fold metrics
    outer_f1_scores = []
    outer_auc_scores = []

    # Iterate over each outer fold
    fold_idx = 1
    for train_idx, test_idx in cv_outer.split(X, y_encoded):
        print(f"Processing outer fold {fold_idx}...")
        X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
        y_train_outer, y_test_outer = y_encoded[train_idx], y_encoded[test_idx]

        # Define inner cross-validation strategy
        cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=1234)

        # Define the objective function for Optuna within the outer fold
        def objective_inner(trial):
            # Start building the steps list
            steps = [('scaler', StandardScaler())]

            # Depending on the feature selection method, add steps and hyperparameters
            if feature_selection_method == 'elasticnet':
                # Suggest hyperparameters for ElasticNet
                elasticnet_alpha = trial.suggest_loguniform('elasticnet_alpha', 1e-4, 1e1)
                l1_ratio = trial.suggest_uniform('elasticnet_l1_ratio', 0.0, 1.0)
                steps.append(('feature_selection', SelectFromModel(
                    ElasticNet(alpha=elasticnet_alpha, l1_ratio=l1_ratio, max_iter=10000, tol=1e-4, random_state=1234)
                )))
            elif feature_selection_method == 'pca':
                max_pca_components = min(X_train_outer.shape[1], X_train_outer.shape[0]-1)
                n_components = trial.suggest_int('pca_n_components', 1, max_pca_components)
                steps.append(('feature_selection', PCA(n_components=n_components, random_state=1234)))
            elif feature_selection_method == 'kpca':
                n_samples_train = X_train_outer.shape[0]
                max_kpca_components = min(X_train_outer.shape[1], n_samples_train -1)
                max_kpca_components = max(1, max_kpca_components)
                n_components = trial.suggest_int('kpca_n_components', 1, max_kpca_components)
                kernel = trial.suggest_categorical('kpca_kernel', ['linear', 'poly', 'rbf', 'sigmoid', 'cosine'])
                kpca_params = {'n_components': n_components, 'kernel': kernel, 'random_state': 1234, 'eigen_solver': 'arpack', 'max_iter': 5000}
                if kernel in ['poly', 'rbf', 'sigmoid']:
                    kpca_gamma = trial.suggest_loguniform('kpca_gamma', 1e-4, 1e1)
                    kpca_params['gamma'] = kpca_gamma
                if kernel in ['poly', 'sigmoid']:
                    kpca_coef0 = trial.suggest_uniform('kpca_coef0', 0.0, 1.0)
                    kpca_params['coef0'] = kpca_coef0
                if kernel == 'poly':
                    kpca_degree = trial.suggest_int('kpca_degree', 2, 5)
                    kpca_params['degree'] = kpca_degree
                steps.append(('feature_selection', KernelPCA(**kpca_params)))
            elif feature_selection_method == 'umap':
                max_umap_components = min(X_train_outer.shape[1], 100)
                n_components = trial.suggest_int('umap_n_components', 2, max_umap_components)
                umap_n_neighbors = trial.suggest_int('umap_n_neighbors', 5, min(50, X_train_outer.shape[0]-1))
                min_dist = trial.suggest_uniform('umap_min_dist', 0.0, 0.99)
                steps.append(('feature_selection', umap.UMAP(n_components=n_components, n_neighbors=umap_n_neighbors, min_dist=min_dist, random_state=1234)))
            elif feature_selection_method == 'pls':
                max_pls_components = min(X_train_outer.shape[1], X_train_outer.shape[0]-1)
                max_pls_components = max(1, max_pls_components)
                n_components = trial.suggest_int('pls_n_components', 1, max_pls_components)
                steps.append(('feature_selection', PLSFeatureSelector(n_components=n_components)))
            elif feature_selection_method == 'tsne':
                # t-SNE requires n_components <=3
                n_components = trial.suggest_int('tsne_n_components', 2, 3)
                perplexity = trial.suggest_int('tsne_perplexity', 5, min(50, X_train_outer.shape[0]-1))
                learning_rate = trial.suggest_loguniform('tsne_learning_rate', 10, 1000)
                max_iter = trial.suggest_int('tsne_max_iter', 250, 2000)
                steps.append(('feature_selection', TSNETransformer(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, max_iter=max_iter, random_state=1234)))
            else:
                # No feature selection
                pass

            # Suggest hyperparameters for KNN with unique name
            knn_n_neighbors = trial.suggest_int('knn_n_neighbors', 1, min(30, len(X_train_outer) - 1))

            # Add KNN to pipeline
            steps.append(('knn', KNeighborsClassifier(n_neighbors=knn_n_neighbors)))

            # Create pipeline
            pipeline = Pipeline(steps)

            # Perform inner cross-validation
            with SuppressOutput():
                f1_scores = []
                for inner_train_idx, inner_valid_idx in cv_inner.split(X_train_outer, y_train_outer):
                    X_train_inner, X_valid_inner = X_train_outer.iloc[inner_train_idx], X_train_outer.iloc[inner_valid_idx]
                    y_train_inner, y_valid_inner = y_train_outer[inner_train_idx], y_train_outer[inner_valid_idx]
                    try:
                        pipeline.fit(X_train_inner, y_train_inner)
                        y_pred_inner = pipeline.predict(X_valid_inner)
                        f1 = f1_score(y_valid_inner, y_pred_inner, average='weighted')
                        f1_scores.append(f1)
                    except (NotImplementedError, ArpackError, ValueError):
                        # If feature selection fails
                        return 0.0
                return np.mean(f1_scores)

        # Create an Optuna study for the inner fold
        study_inner = optuna.create_study(direction='maximize', sampler=TPESampler(seed=1234))
        study_inner.optimize(objective_inner, n_trials=50, show_progress_bar=False)

        # Best hyperparameters from the inner fold
        best_params_inner = study_inner.best_params

        # Initialize the best model with the best hyperparameters
        steps = [('scaler', StandardScaler())]

        if feature_selection_method == 'elasticnet':
            best_elasticnet_alpha = best_params_inner['elasticnet_alpha']
            best_elasticnet_l1_ratio = best_params_inner['elasticnet_l1_ratio']
            steps.append(('feature_selection', SelectFromModel(
                ElasticNet(alpha=best_elasticnet_alpha, l1_ratio=best_elasticnet_l1_ratio, max_iter=10000, tol=1e-4, random_state=1234)
            )))
        elif feature_selection_method == 'pca':
            best_pca_n_components = best_params_inner['pca_n_components']
            steps.append(('feature_selection', PCA(n_components=best_pca_n_components, random_state=1234)))
        elif feature_selection_method == 'kpca':
            best_kpca_n_components = best_params_inner['kpca_n_components']
            best_kpca_kernel = best_params_inner['kpca_kernel']
            kpca_params = {'n_components': best_kpca_n_components, 'kernel': best_kpca_kernel, 'random_state': 1234, 'eigen_solver': 'arpack', 'max_iter': 5000}
            if best_kpca_kernel in ['poly', 'rbf', 'sigmoid']:
                kpca_params['gamma'] = best_params_inner['kpca_gamma']
            if best_kpca_kernel in ['poly', 'sigmoid']:
                kpca_params['coef0'] = best_params_inner['kpca_coef0']
            if best_kpca_kernel == 'poly':
                kpca_params['degree'] = best_params_inner['kpca_degree']
            steps.append(('feature_selection', KernelPCA(**kpca_params)))
        elif feature_selection_method == 'umap':
            best_umap_n_components = best_params_inner['umap_n_components']
            best_umap_n_neighbors = best_params_inner['umap_n_neighbors']
            best_umap_min_dist = best_params_inner['umap_min_dist']
            steps.append(('feature_selection', umap.UMAP(n_components=best_umap_n_components, n_neighbors=best_umap_n_neighbors, min_dist=best_umap_min_dist, random_state=1234)))
        elif feature_selection_method == 'pls':
            best_pls_n_components = best_params_inner['pls_n_components']
            steps.append(('feature_selection', PLSFeatureSelector(n_components=best_pls_n_components)))
        elif feature_selection_method == 'tsne':
            best_tsne_n_components = best_params_inner['tsne_n_components']
            best_tsne_perplexity = best_params_inner['tsne_perplexity']
            best_tsne_learning_rate = best_params_inner['tsne_learning_rate']
            best_tsne_max_iter = best_params_inner['tsne_max_iter']
            steps.append(('feature_selection', TSNETransformer(n_components=best_tsne_n_components, perplexity=best_tsne_perplexity, learning_rate=best_tsne_learning_rate, max_iter=best_tsne_max_iter, random_state=1234)))
        else:
            # No feature selection
            pass

        best_knn_n_neighbors = best_params_inner['knn_n_neighbors']
        steps.append(('knn', KNeighborsClassifier(n_neighbors=best_knn_n_neighbors)))

        best_model_inner = Pipeline(steps)

        # Fit the model on the outer training set
        with SuppressOutput():
            try:
                best_model_inner.fit(X_train_outer, y_train_outer)
            except (NotImplementedError, ArpackError, ValueError) as e:
                outer_f1_scores.append(0.0)
                outer_auc_scores.append(0.0)
                print(f"Fold {fold_idx} - F1 Score: 0.0000, AUC: 0.0000 ({str(e)})")
                fold_idx += 1
                continue

        # Predict probabilities on the outer test set
        try:
            y_pred_prob_outer = best_model_inner.predict_proba(X_test_outer)
            y_pred_class_outer = best_model_inner.predict(X_test_outer)
        except (NotImplementedError, ArpackError, ValueError) as e:
            y_pred_prob_outer = np.zeros((X_test_outer.shape[0], num_classes))
            y_pred_class_outer = np.zeros(X_test_outer.shape[0])
            print(f"Prediction failed for fold {fold_idx} due to: {str(e)}")

        # Compute F1 score
        f1_outer = f1_score(y_test_outer, y_pred_class_outer, average='weighted')
        outer_f1_scores.append(f1_outer)

        # Compute AUC
        if num_classes == 2 and y_pred_prob_outer.shape[1] == 2:
            try:
                fpr_val, tpr_val, _ = roc_curve(y_test_outer, y_pred_prob_outer[:, 1])
                auc_outer = auc(fpr_val, tpr_val)
            except ValueError:
                auc_outer = 0.0
        else:
            # Compute micro-average ROC AUC for multi-class
            try:
                fpr_val, tpr_val, _ = roc_curve(y_binarized[test_idx].ravel(), y_pred_prob_outer.ravel())
                auc_outer = auc(fpr_val, tpr_val)
            except ValueError:
                auc_outer = 0.0
        outer_auc_scores.append(auc_outer)

        print(f"Fold {fold_idx} - F1 Score: {f1_outer:.4f}, AUC: {auc_outer:.4f}")
        fold_idx += 1

    # Plot and save the F1 and AUC scores per outer fold
    plt.figure(figsize=(10, 6))
    folds = range(1, cv_outer.get_n_splits() + 1)
    plt.plot(folds, outer_f1_scores, marker='o', label='F1 Score')
    plt.plot(folds, outer_auc_scores, marker='s', label='AUC')
    plt.xlabel('Outer Fold')
    plt.ylabel('Score')
    plt.title('F1 and AUC Scores per Outer Fold')
    plt.xticks(folds)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{prefix}_knn_nested_cv_f1_auc.png", dpi=300)
    plt.close()

    print("Nested cross-validation completed.")
    print(f"Average F1 Score: {np.mean(outer_f1_scores):.4f} ± {np.std(outer_f1_scores):.4f}")
    print(f"Average AUC: {np.mean(outer_auc_scores):.4f} ± {np.std(outer_auc_scores):.4f}")

    # After nested CV, perform hyperparameter tuning on the entire dataset
    print("Starting hyperparameter tuning on the entire dataset...")

    # Define the objective function for Optuna on the entire dataset
    def objective_full(trial):
        # Start building the steps list
        steps = [('scaler', StandardScaler())]

        # Depending on the feature selection method, add steps and hyperparameters
        if feature_selection_method == 'elasticnet':
            # Suggest hyperparameters for ElasticNet
            elasticnet_alpha = trial.suggest_loguniform('elasticnet_alpha', 1e-4, 1e1)
            l1_ratio = trial.suggest_uniform('elasticnet_l1_ratio', 0.0, 1.0)
            steps.append(('feature_selection', SelectFromModel(
                ElasticNet(alpha=elasticnet_alpha, l1_ratio=l1_ratio, max_iter=10000, tol=1e-4, random_state=1234)
            )))
        elif feature_selection_method == 'pca':
            max_pca_components = min(X.shape[1], X.shape[0]-1)
            max_pca_components = max(1, max_pca_components)
            n_components = trial.suggest_int('pca_n_components', 1, max_pca_components)
            steps.append(('feature_selection', PCA(n_components=n_components, random_state=1234)))
        elif feature_selection_method == 'kpca':
            n_samples = X.shape[0]
            n_features = X.shape[1]
            max_kpca_components = min(n_features, n_samples -1)
            max_kpca_components = max(1, max_kpca_components)
            n_components = trial.suggest_int('kpca_n_components', 1, max_kpca_components)
            kernel = trial.suggest_categorical('kpca_kernel', ['linear', 'poly', 'rbf', 'sigmoid', 'cosine'])
            kpca_params = {'n_components': n_components, 'kernel': kernel, 'random_state': 1234, 'eigen_solver': 'arpack', 'max_iter': 5000}
            if kernel in ['poly', 'rbf', 'sigmoid']:
                kpca_gamma = trial.suggest_loguniform('kpca_gamma', 1e-4, 1e1)
                kpca_params['gamma'] = kpca_gamma
            if kernel in ['poly', 'sigmoid']:
                kpca_coef0 = trial.suggest_uniform('kpca_coef0', 0.0, 1.0)
                kpca_params['coef0'] = kpca_coef0
            if kernel == 'poly':
                kpca_degree = trial.suggest_int('kpca_degree', 2, 5)
                kpca_params['degree'] = kpca_degree
            steps.append(('feature_selection', KernelPCA(**kpca_params)))
        elif feature_selection_method == 'umap':
            max_umap_components = min(X.shape[1], 100)
            n_components = trial.suggest_int('umap_n_components', 2, max_umap_components)
            umap_n_neighbors = trial.suggest_int('umap_n_neighbors', 5, min(50, X.shape[0]-1))
            min_dist = trial.suggest_uniform('umap_min_dist', 0.0, 0.99)
            steps.append(('feature_selection', umap.UMAP(n_components=n_components, n_neighbors=umap_n_neighbors, min_dist=min_dist, random_state=1234)))
        elif feature_selection_method == 'pls':
            max_pls_components = min(X.shape[1], X.shape[0]-1)
            max_pls_components = max(1, max_pls_components)
            n_components = trial.suggest_int('pls_n_components', 1, max_pls_components)
            steps.append(('feature_selection', PLSFeatureSelector(n_components=n_components)))
        elif feature_selection_method == 'tsne':
            # t-SNE requires n_components <=3
            n_components = trial.suggest_int('tsne_n_components', 2, 3)
            perplexity = trial.suggest_int('tsne_perplexity', 5, min(50, X.shape[0]-1))
            learning_rate = trial.suggest_loguniform('tsne_learning_rate', 10, 1000)
            max_iter = trial.suggest_int('tsne_max_iter', 250, 2000)
            steps.append(('feature_selection', TSNETransformer(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, max_iter=max_iter, random_state=1234)))
        else:
            # No feature selection
            pass

        # Suggest hyperparameters for KNN with unique name
        knn_n_neighbors = trial.suggest_int('knn_n_neighbors', 1, min(30, len(X) - 1))

        # Add KNN to pipeline
        steps.append(('knn', KNeighborsClassifier(n_neighbors=knn_n_neighbors)))

        # Create pipeline
        pipeline = Pipeline(steps)

        # Perform cross-validation
        with SuppressOutput():
            f1_scores = []
            for train_idx_full, valid_idx_full in cv_outer.split(X, y_encoded):
                X_train_full, X_valid_full = X.iloc[train_idx_full], X.iloc[valid_idx_full]
                y_train_full, y_valid_full = y_encoded[train_idx_full], y_encoded[valid_idx_full]
                try:
                    pipeline.fit(X_train_full, y_train_full)
                    y_pred_full = pipeline.predict(X_valid_full)
                    f1 = f1_score(y_valid_full, y_pred_full, average='weighted')
                    f1_scores.append(f1)
                except (NotImplementedError, ArpackError, ValueError):
                    # If feature selection fails
                    f1_scores.append(0.0)
            return np.mean(f1_scores)

    # Create an Optuna study for the entire dataset
    study_full = optuna.create_study(direction='maximize', sampler=TPESampler(seed=1234))
    study_full.optimize(objective_full, n_trials=50, show_progress_bar=True)

    # Best hyperparameters from the entire dataset
    best_params_full = study_full.best_params
    print(f"Best parameters for KNN: {best_params_full}")

    # Initialize the best model with the best hyperparameters
    steps = [('scaler', StandardScaler())]

    if feature_selection_method == 'elasticnet':
        best_elasticnet_alpha_full = best_params_full['elasticnet_alpha']
        best_elasticnet_l1_ratio_full = best_params_full['elasticnet_l1_ratio']
        steps.append(('feature_selection', SelectFromModel(
            ElasticNet(alpha=best_elasticnet_alpha_full, l1_ratio=best_elasticnet_l1_ratio_full, max_iter=10000, tol=1e-4, random_state=1234)
        )))
    elif feature_selection_method == 'pca':
        best_pca_n_components_full = best_params_full['pca_n_components']
        steps.append(('feature_selection', PCA(n_components=best_pca_n_components_full, random_state=1234)))
    elif feature_selection_method == 'kpca':
        best_kpca_n_components_full = best_params_full['kpca_n_components']
        best_kpca_kernel_full = best_params_full['kpca_kernel']
        kpca_params = {'n_components': best_kpca_n_components_full, 'kernel': best_kpca_kernel_full, 'random_state': 1234, 'eigen_solver': 'arpack', 'max_iter': 5000}
        if best_kpca_kernel_full in ['poly', 'rbf', 'sigmoid']:
            kpca_params['gamma'] = best_params_full['kpca_gamma']
        if best_kpca_kernel_full in ['poly', 'sigmoid']:
            kpca_params['coef0'] = best_params_full['kpca_coef0']
        if best_kpca_kernel_full == 'poly':
            kpca_params['degree'] = best_params_full['kpca_degree']
        steps.append(('feature_selection', KernelPCA(**kpca_params)))
    elif feature_selection_method == 'umap':
        best_umap_n_components_full = best_params_full['umap_n_components']
        best_umap_n_neighbors_full = best_params_full['umap_n_neighbors']
        best_umap_min_dist_full = best_params_full['umap_min_dist']
        steps.append(('feature_selection', umap.UMAP(n_components=best_umap_n_components_full, n_neighbors=best_umap_n_neighbors_full, min_dist=best_umap_min_dist_full, random_state=1234)))
    elif feature_selection_method == 'pls':
        best_pls_n_components_full = best_params_full['pls_n_components']
        steps.append(('feature_selection', PLSFeatureSelector(n_components=best_pls_n_components_full)))
    elif feature_selection_method == 'tsne':
        best_tsne_n_components_full = best_params_full['tsne_n_components']
        best_tsne_perplexity_full = best_params_full['tsne_perplexity']
        best_tsne_learning_rate_full = best_params_full['tsne_learning_rate']
        best_tsne_max_iter_full = best_params_full['tsne_max_iter']
        steps.append(('feature_selection', TSNETransformer(n_components=best_tsne_n_components_full, perplexity=best_tsne_perplexity_full, learning_rate=best_tsne_learning_rate_full, max_iter=best_tsne_max_iter_full, random_state=1234)))
    else:
        # No feature selection
        pass

    best_knn_n_neighbors_full = best_params_full['knn_n_neighbors']
    steps.append(('knn', KNeighborsClassifier(n_neighbors=best_knn_n_neighbors_full)))

    best_model = Pipeline(steps)

    # Fit the model on the entire dataset
    with SuppressOutput():
        try:
            best_model.fit(X, y_encoded)
        except (NotImplementedError, ArpackError, ValueError):
            print("Feature selection method failed on the entire dataset. Skipping feature selection.")
            # Remove feature selection step and fit KNN only
            steps = [('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=best_knn_n_neighbors_full))]
            best_model = Pipeline(steps)
            best_model.fit(X, y_encoded)

    # Save the best model and data along with feature names
    joblib.dump(best_model, f"{prefix}_knn_model.pkl")
    joblib.dump((X, y_encoded, le), f"{prefix}_knn_data.pkl")

    # Output the best parameters
    print(f"Best parameters for KNN: {best_params_full}")

    # If feature selection is used, save the transformed data and variance information
    if feature_selection_method != 'none':
        if 'feature_selection' in best_model.named_steps:
            try:
                # Transform the entire dataset
                X_transformed = best_model.named_steps['feature_selection'].transform(X)
            except (NotImplementedError, ArpackError, ValueError):
                # If feature selection fails, skip saving transformed data
                X_transformed = None
        else:
            print("Feature selection step is not present in the pipeline.")
            X_transformed = None

        if X_transformed is not None:
            # Create a DataFrame for the transformed data
            if feature_selection_method in ['pca', 'kpca', 'umap', 'pls']:
                n_components = X_transformed.shape[1]
                transformed_columns = [f"{feature_selection_method.upper()}_Component_{i+1}" for i in range(n_components)]
                X_transformed_df = pd.DataFrame(X_transformed, columns=transformed_columns)
            elif feature_selection_method == 'tsne':
                # t-SNE with optimized n_components
                transformed_columns = [f"{feature_selection_method.upper()}_Component_{i+1}" for i in range(X_transformed.shape[1])]
                X_transformed_df = pd.DataFrame(X_transformed, columns=transformed_columns)
            elif feature_selection_method == 'elasticnet':
                selected_features = X.columns[best_model.named_steps['feature_selection'].get_support()]
                X_transformed_df = X[selected_features].copy()
            else:
                # For other methods or no feature selection
                X_transformed_df = pd.DataFrame(X_transformed)

            # Add SampleID and Label
            X_transformed_df.insert(0, 'SampleID', sample_ids)
            X_transformed_df['Label'] = y

            # Save the transformed data
            transformed_csv_path = f"{prefix}_knn_transformed_X.csv"
            X_transformed_df.to_csv(transformed_csv_path, index=False)
            print(f"Transformed data saved to {transformed_csv_path}")

            # Save variance information if applicable
            variance_csv_path = f"{prefix}_knn_variance.csv"
            if feature_selection_method == 'pca':
                pca_step = best_model.named_steps['feature_selection']
                explained_variance = pca_step.explained_variance_ratio_
                explained_variance_df = pd.DataFrame({
                    'Component': range(1, len(explained_variance)+1),
                    'Explained Variance Ratio': explained_variance
                })
                explained_variance_df.to_csv(variance_csv_path, index=False)
                print(f"PCA explained variance ratios saved to {variance_csv_path}")
            elif feature_selection_method == 'pls':
                pls_step = best_model.named_steps['feature_selection'].pls
                x_scores = pls_step.x_scores_
                explained_variance = np.var(x_scores, axis=0) / np.var(X, axis=0).sum()
                explained_variance_df = pd.DataFrame({
                    'Component': range(1, len(explained_variance)+1),
                    'Explained Variance Ratio': explained_variance
                })
                explained_variance_df.to_csv(variance_csv_path, index=False)
                print(f"PLS explained variance ratios saved to {variance_csv_path}")
            elif feature_selection_method == 'tsne':
                # t-SNE does not provide variance information
                with open(variance_csv_path, 'w') as f:
                    f.write("t-SNE does not provide variance information.\n")
                print(f"No variance information available for t-SNE. File created at {variance_csv_path}")
            elif feature_selection_method == 'elasticnet':
                # ElasticNet does not provide variance information
                with open(variance_csv_path, 'w') as f:
                    f.write("ElasticNet does not provide variance information.\n")
                print(f"No variance information available for ElasticNet. File created at {variance_csv_path}")
            else:
                # For KernelPCA, UMAP, etc., variance information is not directly available
                with open(variance_csv_path, 'w') as f:
                    f.write(f"{feature_selection_method.upper()} does not provide variance information.\n")
                print(f"No variance information available for {feature_selection_method.upper()}. File created at {variance_csv_path}")
        else:
            print("Transformed data is not available.")
    else:
        print("No feature selection method selected. Skipping transformed data and variance information saving.")

    # Prediction using cross_val_predict
    try:
        y_pred_prob = cross_val_predict(best_model, X, y_encoded, cv=cv_outer, method='predict_proba', n_jobs=-1)
        y_pred_class = np.argmax(y_pred_prob, axis=1)
    except (NotImplementedError, ArpackError, ValueError):
        y_pred_prob = np.zeros((X.shape[0], num_classes))
        y_pred_class = np.zeros(X.shape[0])

    # Compute metrics
    acc = accuracy_score(y_encoded, y_pred_class)
    f1 = f1_score(y_encoded, y_pred_class, average='weighted')
    cm = confusion_matrix(y_encoded, y_pred_class)

    # Compute sensitivity and specificity
    if num_classes == 2:
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            sensitivity = 0
            specificity = 0
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            sensitivity = np.mean(np.divide(np.diag(cm), np.sum(cm, axis=1),
                                            out=np.zeros_like(np.diag(cm), dtype=float),
                                            where=np.sum(cm, axis=1) != 0))
            specificity = np.mean(np.divide(np.diag(cm), np.sum(cm, axis=0),
                                            out=np.zeros_like(np.diag(cm), dtype=float),
                                            where=np.sum(cm, axis=0) != 0))

    # Confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for KNN')
    plt.savefig(f"{prefix}_knn_confusion_matrix.png", dpi=300)
    plt.close()

    # ROC and AUC
    fpr = {}
    tpr = {}
    roc_auc = {}

    # Different handling for binary and multi-class cases
    if num_classes == 2 and y_pred_prob.shape[1] == 2:
        try:
            fpr[0], tpr[0], _ = roc_curve(y_binarized[:, 0], y_pred_prob[:, 0])
            roc_auc[0] = auc(fpr[0], tpr[0])
        except ValueError:
            roc_auc[0] = 0.0
    else:
        for i in range(y_binarized.shape[1]):
            if np.sum(y_binarized[:, i]) == 0:
                fpr[i] = np.array([0, 1])
                tpr[i] = np.array([0, 1])
                roc_auc[i] = 0.0
            else:
                try:
                    fpr[i], tpr[i], _ = roc_curve(y_binarized[:, i], y_pred_prob[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                except ValueError:
                    fpr[i], tpr[i], roc_auc[i] = np.array([0, 1]), np.array([0, 1]), 0.0

        # Compute overall ROC AUC for multi-class
        try:
            fpr["micro"], tpr["micro"], _ = roc_curve(y_binarized.ravel(), y_pred_prob.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        except ValueError:
            fpr["micro"], tpr["micro"], roc_auc["micro"] = np.array([0, 1]), np.array([0, 1]), 0.0

    # Save ROC data
    roc_data = {
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc
    }
    np.save(f"{prefix}_knn_roc_data.npy", roc_data)

    # Plot and save ROC curve
    plt.figure(figsize=(10, 8))

    if num_classes == 2 and y_pred_prob.shape[1] == 2:
        plt.plot(fpr[0], tpr[0], label=f'AUC = {roc_auc[0]:.2f}')
    else:
        for i in range(len(le.classes_)):
            if roc_auc.get(i, 0.0) > 0.0:
                plt.plot(fpr[i], tpr[i], label=f'{le.inverse_transform([i])[0]} (AUC = {roc_auc[i]:.2f})')
        if roc_auc.get("micro", 0.0) > 0.0:
            plt.plot(fpr["micro"], tpr["micro"], label=f'Overall (AUC = {roc_auc["micro"]:.2f})', linestyle='--')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for KNN')
    plt.legend(loc="lower right")
    plt.savefig(f"{prefix}_knn_roc_curve.png", dpi=300)
    plt.close()

    # Output performance metrics as a bar chart
    metrics = {'Accuracy': acc, 'F1 Score': f1, 'Sensitivity': sensitivity, 'Specificity': specificity}
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    ax = metrics_df.plot(kind='bar', x='Metric', y='Value', legend=False)
    plt.title('Performance Metrics for KNN')
    plt.ylabel('Value')
    plt.ylim(0, 1)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{prefix}_knn_metrics.png", dpi=300)
    plt.close()

    # Create a DataFrame for predictions
    predictions_df = pd.DataFrame({
        'SampleID': sample_ids,
        'Original Label': y,
        'Predicted Label': le.inverse_transform(y_pred_class.astype(int))
    })

    # Save predictions to CSV
    predictions_df.to_csv(f"{prefix}_knn_predictions.csv", index=False)

    print(f"Predictions saved to {prefix}_knn_predictions.csv")

if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Run KNN model with Nested Cross-Validation, Optional Feature Selection, and Optuna hyperparameter optimization.')
    parser.add_argument('-i', '--csv', type=str, required=True, help='Input file in CSV format.')
    parser.add_argument('-p', '--prefix', type=str, required=True, help='Output prefix.')
    parser.add_argument('-f', '--feature_selection', type=str, choices=['none', 'elasticnet', 'pca', 'kpca', 'umap', 'pls', 'tsne'], default='none', help='Feature selection method to use.')

    # Parse the arguments
    args = parser.parse_args()

    # Run the KNN nested cross-validation function
    knn_nested_cv(args.csv, args.prefix, args.feature_selection)
