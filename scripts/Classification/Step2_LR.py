import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, label_binarize, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    f1_score,
    accuracy_score,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import warnings
import sys
import contextlib
import joblib
import optuna
from optuna.samplers import TPESampler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
import umap
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import ConvergenceWarning
from scipy.sparse.linalg import ArpackError

# Suppress all warnings except ConvergenceWarning from models
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# Suppress output during model fitting and evaluation
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

# Custom PLS Feature Selector without random_state
class PLSFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, max_iter=1000, tol=1e-06):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.pls = None

    def fit(self, X, y):
        self.pls = PLSRegression(n_components=self.n_components, max_iter=self.max_iter, tol=self.tol)
        self.pls.fit(X, y)
        return self

    def transform(self, X):
        return self.pls.transform(X)

# Custom t-SNE Transformer
class TSNETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, perplexity=30, learning_rate=200, max_iter=1000):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tsne = TSNE(n_components=self.n_components, perplexity=self.perplexity,
                         learning_rate=self.learning_rate, max_iter=self.max_iter,
                         random_state=1234)
        self.X_transformed_ = None

    def fit(self, X, y=None):
        self.X_transformed_ = self.tsne.fit_transform(X)
        return self

    def transform(self, X):
        # t-SNE does not support transforming new data
        # During cross-validation, bypass transformation for validation data
        if self.X_transformed_ is not None and X.shape[0] == self.X_transformed_.shape[0]:
            return self.X_transformed_
        else:
            # Return the data as-is for new data to prevent errors
            return X

# ElasticNet Feature Selector
class ElasticNetFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=10000, tol=1e-4):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.selector = SelectFromModel(
            LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=self.l1_ratio,
                               C=1.0 / self.alpha, max_iter=self.max_iter, tol=self.tol, random_state=1234)
        )

    def fit(self, X, y):
        self.selector.fit(X, y)
        return self

    def transform(self, X):
        return self.selector.transform(X)

def logistic_regression_nested_cv(inp, prefix, feature_selection_method):
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

    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_binarized = label_binarize(y_encoded, classes=np.unique(y_encoded))
    num_classes = len(np.unique(y_encoded))

    # Define outer cross-validation strategy
    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

    # Lists to store outer fold metrics
    outer_f1_scores = []
    outer_auc_scores = []

    # Check if t-SNE is selected
    tsne_selected = (feature_selection_method == 'tsne')

    if tsne_selected:
        print("Applying t-SNE transformation to the entire dataset before cross-validation...")
        # Perform hyperparameter tuning for t-SNE
        def tsne_objective(trial):
            n_components = trial.suggest_int('tsne_n_components', 2, 3)
            perplexity = trial.suggest_int('tsne_perplexity', 5, min(50, X.shape[0]-1))
            learning_rate = trial.suggest_loguniform('tsne_learning_rate', 10, 1000)
            max_iter = trial.suggest_int('tsne_max_iter', 250, 2000)
            tsne = TSNE(n_components=n_components, perplexity=perplexity,
                        learning_rate=learning_rate, max_iter=max_iter,
                        random_state=1234)
            with SuppressOutput():
                X_tsne = tsne.fit_transform(X)
                # Use variance as a proxy metric for optimization
                variance = np.var(X_tsne)
            return variance

        # Create Optuna study for t-SNE
        study_tsne = optuna.create_study(direction='maximize', sampler=TPESampler(seed=1234))
        study_tsne.optimize(tsne_objective, n_trials=20, show_progress_bar=False)

        # Get best t-SNE parameters
        best_tsne_params = study_tsne.best_params
        print(f"Best t-SNE parameters: {best_tsne_params}")

        # Apply t-SNE with best parameters
        tsne_final = TSNE(n_components=best_tsne_params['tsne_n_components'],
                          perplexity=best_tsne_params['tsne_perplexity'],
                          learning_rate=best_tsne_params['tsne_learning_rate'],
                          max_iter=best_tsne_params['tsne_max_iter'],
                          random_state=1234)
        with SuppressOutput():
            X_tsne_final = tsne_final.fit_transform(X)
        X_transformed = pd.DataFrame(X_tsne_final, columns=[f"TSNE_Component_{i+1}" for i in range(X_tsne_final.shape[1])])

        # Assign transformed data to X_transformed_final
        X_transformed_final = X_transformed.reset_index(drop=True)

    # Iterate over each outer fold
    fold_idx = 1
    for train_idx, test_idx in cv_outer.split(X, y_encoded):
        print(f"Processing outer fold {fold_idx}...")
        X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
        y_train_outer, y_test_outer = y_encoded[train_idx], y_encoded[test_idx]

        if tsne_selected:
            # Use the already transformed data
            X_train_outer_fold = X_transformed_final.iloc[train_idx].reset_index(drop=True)
            X_test_outer_fold = X_transformed_final.iloc[test_idx].reset_index(drop=True)
        else:
            X_train_outer_fold = X_train_outer
            X_test_outer_fold = X_test_outer

        # Define inner cross-validation strategy
        cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

        if not tsne_selected:
            # Define the objective function for Optuna within the outer fold
            def objective_inner(trial):
                steps = []

                # Add feature selection based on the method
                if feature_selection_method != 'none':
                    if feature_selection_method == 'pca':
                        pca_n_components = trial.suggest_int('pca_n_components', 1, min(X_train_outer_fold.shape[1], X_train_outer_fold.shape[0]-1))
                        steps.append(('feature_selection', PCA(n_components=pca_n_components, random_state=1234)))
                    elif feature_selection_method == 'kpca':
                        kpca_kernel = trial.suggest_categorical('kpca_kernel', ['linear', 'poly', 'rbf', 'sigmoid', 'cosine'])
                        kpca_n_components = trial.suggest_int('kpca_n_components', 1, min(X_train_outer_fold.shape[1], X_train_outer_fold.shape[0]-1))
                        kpca_params = {
                            'n_components': kpca_n_components,
                            'kernel': kpca_kernel,
                            'random_state': 1234,
                            'eigen_solver': 'arpack',
                            'max_iter': 5000
                        }
                        if kpca_kernel in ['poly', 'rbf', 'sigmoid']:
                            kpca_gamma = trial.suggest_loguniform('kpca_gamma', 1e-4, 1e1)
                            kpca_params['gamma'] = kpca_gamma
                        if kpca_kernel in ['poly', 'sigmoid']:
                            kpca_coef0 = trial.suggest_uniform('kpca_coef0', 0.0, 1.0)
                            kpca_params['coef0'] = kpca_coef0
                        if kpca_kernel == 'poly':
                            kpca_degree = trial.suggest_int('kpca_degree', 2, 5)
                            kpca_params['degree'] = kpca_degree
                        steps.append(('feature_selection', KernelPCA(**kpca_params)))
                    elif feature_selection_method == 'umap':
                        umap_n_components = trial.suggest_int('umap_n_components', 2, min(100, X_train_outer_fold.shape[1]))
                        umap_n_neighbors = trial.suggest_int('umap_n_neighbors', 5, min(50, X_train_outer_fold.shape[0]-1))
                        umap_min_dist = trial.suggest_uniform('umap_min_dist', 0.0, 0.99)
                        steps.append(('feature_selection', umap.UMAP(
                            n_components=umap_n_components,
                            n_neighbors=umap_n_neighbors,
                            min_dist=umap_min_dist,
                            random_state=1234
                        )))
                    elif feature_selection_method == 'pls':
                        pls_n_components = trial.suggest_int('pls_n_components', 1, min(X_train_outer_fold.shape[1], X_train_outer_fold.shape[0]-1))
                        steps.append(('feature_selection', PLSFeatureSelector(
                            n_components=pls_n_components,
                            max_iter=1000,
                            tol=1e-06
                        )))
                    elif feature_selection_method == 'elasticnet':
                        elasticnet_alpha = trial.suggest_loguniform('elasticnet_alpha', 1e-4, 1e1)
                        l1_ratio = trial.suggest_uniform('elasticnet_l1_ratio', 0.0, 1.0)
                        steps.append(('feature_selection', ElasticNetFeatureSelector(
                            alpha=elasticnet_alpha,
                            l1_ratio=l1_ratio,
                            max_iter=10000,
                            tol=1e-4
                        )))

                # Add Logistic Regression to the pipeline
                C = trial.suggest_loguniform('C', 1e-2, 100)
                steps.append(('logreg', LogisticRegression(C=C, random_state=1234, max_iter=1000)))

                pipeline = Pipeline(steps)

                # Perform inner cross-validation
                with SuppressOutput():
                    f1_scores = []
                    for inner_train_idx, inner_valid_idx in cv_inner.split(X_train_outer_fold, y_train_outer):
                        X_train_inner, X_valid_inner = X_train_outer_fold.iloc[inner_train_idx], X_train_outer_fold.iloc[inner_valid_idx]
                        y_train_inner, y_valid_inner = y_train_outer[inner_train_idx], y_train_outer[inner_valid_idx]
                        try:
                            pipeline.fit(X_train_inner, y_train_inner)
                            y_pred_inner = pipeline.predict(X_valid_inner)
                            f1 = f1_score(y_valid_inner, y_pred_inner, average='weighted')
                            f1_scores.append(f1)
                        except (ValueError, ArpackError, NotImplementedError):
                            # If feature selection or Logistic Regression fails
                            f1_scores.append(0.0)
                    return np.mean(f1_scores)

            # Create an Optuna study for the inner fold
            study_inner = optuna.create_study(direction='maximize', sampler=TPESampler(seed=1234))
            study_inner.optimize(objective_inner, n_trials=50, show_progress_bar=False)

            # Best hyperparameters from the inner fold
            best_params_inner = study_inner.best_params

            # Initialize the best model with the best hyperparameters
            steps = []

            # Add feature selection based on the method
            if feature_selection_method != 'none':
                if feature_selection_method == 'pca':
                    best_pca_n_components = best_params_inner.get('pca_n_components', 2)
                    steps.append(('feature_selection', PCA(n_components=best_pca_n_components, random_state=1234)))
                elif feature_selection_method == 'kpca':
                    best_kpca_kernel = best_params_inner.get('kpca_kernel', 'rbf')
                    best_kpca_n_components = best_params_inner.get('kpca_n_components', 2)
                    kpca_params = {
                        'n_components': best_kpca_n_components,
                        'kernel': best_kpca_kernel,
                        'random_state': 1234,
                        'eigen_solver': 'arpack',
                        'max_iter': 5000
                    }
                    if best_kpca_kernel in ['poly', 'rbf', 'sigmoid']:
                        kpca_params['gamma'] = best_params_inner.get('kpca_gamma', 1.0)
                    if best_kpca_kernel in ['poly', 'sigmoid']:
                        kpca_params['coef0'] = best_params_inner.get('kpca_coef0', 0.0)
                    if best_kpca_kernel == 'poly':
                        kpca_params['degree'] = best_params_inner.get('kpca_degree', 3)
                    steps.append(('feature_selection', KernelPCA(**kpca_params)))
                elif feature_selection_method == 'umap':
                    best_umap_n_components = best_params_inner.get('umap_n_components', 2)
                    best_umap_n_neighbors = best_params_inner.get('umap_n_neighbors', 15)
                    best_umap_min_dist = best_params_inner.get('umap_min_dist', 0.1)
                    steps.append(('feature_selection', umap.UMAP(
                        n_components=best_umap_n_components,
                        n_neighbors=best_umap_n_neighbors,
                        min_dist=best_umap_min_dist,
                        random_state=1234
                    )))
                elif feature_selection_method == 'pls':
                    best_pls_n_components = best_params_inner.get('pls_n_components', 2)
                    steps.append(('feature_selection', PLSFeatureSelector(
                        n_components=best_pls_n_components,
                        max_iter=1000,
                        tol=1e-06
                    )))
                elif feature_selection_method == 'elasticnet':
                    best_elasticnet_alpha = best_params_inner.get('elasticnet_alpha', 1.0)
                    best_l1_ratio = best_params_inner.get('elasticnet_l1_ratio', 0.5)
                    steps.append(('feature_selection', ElasticNetFeatureSelector(
                        alpha=best_elasticnet_alpha,
                        l1_ratio=best_l1_ratio,
                        max_iter=10000,
                        tol=1e-4
                    )))

            # Add Logistic Regression to the pipeline
            best_C = best_params_inner.get('C', 1.0)
            steps.append(('logreg', LogisticRegression(C=best_C, random_state=1234, max_iter=1000)))

            best_model_inner = Pipeline(steps)

            if tsne_selected:
                # For t-SNE, the data has already been transformed outside the pipeline
                # Thus, we only add Logistic Regression
                best_model_inner = Pipeline([
                    ('logreg', LogisticRegression(C=best_C, random_state=1234, max_iter=1000))
                ])
                X_train_outer_fold_final = X_transformed_final.iloc[train_idx].reset_index(drop=True)
                X_test_outer_fold_final = X_transformed_final.iloc[test_idx].reset_index(drop=True)
            else:
                # For other feature selection methods, include the entire pipeline
                X_train_outer_fold_final = X_train_outer_fold
                X_test_outer_fold_final = X_test_outer_fold

            # Fit the model on the outer training set
            with SuppressOutput():
                try:
                    best_model_inner.fit(X_train_outer_fold_final, y_train_outer)
                except (ValueError, ArpackError, NotImplementedError) as e:
                    print(f"Error fitting the model in outer fold {fold_idx}: {e}")
                    outer_f1_scores.append(0)
                    outer_auc_scores.append(0)
                    fold_idx += 1
                    continue

            # Predict probabilities and classes on the outer test set
            try:
                y_pred_prob_outer = best_model_inner.predict_proba(X_test_outer_fold_final)
                y_pred_class_outer = best_model_inner.predict(X_test_outer_fold_final)
            except (ValueError, ArpackError, NotImplementedError) as e:
                print(f"Error predicting in outer fold {fold_idx}: {e}")
                outer_f1_scores.append(0)
                outer_auc_scores.append(0)
                fold_idx += 1
                continue

            # Compute F1 score
            f1_outer = f1_score(y_test_outer, y_pred_class_outer, average='weighted')
            outer_f1_scores.append(f1_outer)

            # Compute AUC
            if num_classes == 2:
                try:
                    fpr, tpr, _ = roc_curve(y_test_outer, y_pred_prob_outer[:, 1])
                    auc_outer = auc(fpr, tpr)
                except ValueError:
                    auc_outer = 0.0
            else:
                try:
                    y_binarized_test = label_binarize(y_test_outer, classes=np.unique(y_encoded))
                    if y_binarized_test.shape[1] == 1:
                        y_binarized_test = np.hstack([1 - y_binarized_test, y_binarized_test])
                    fpr, tpr, _ = roc_curve(y_binarized_test.ravel(), y_pred_prob_outer.ravel())
                    auc_outer = auc(fpr, tpr)
                except ValueError:
                    auc_outer = 0.0
            outer_auc_scores.append(auc_outer)

            print(f"Fold {fold_idx} - F1 Score: {f1_outer:.4f}, AUC: {auc_outer:.4f}")
            fold_idx += 1

    # After cross-validation, proceed based on whether t-SNE was selected
    if tsne_selected:
        # Handle t-SNE path
        print("Completed cross-validation with t-SNE transformed data.")

        # Hyperparameter tuning for Logistic Regression on the entire transformed dataset
        print("Starting hyperparameter tuning for Logistic Regression on the entire t-SNE transformed dataset...")

        def objective_full_tsne(trial):
            C = trial.suggest_loguniform('C', 1e-2, 100)
            model = LogisticRegression(C=C, random_state=1234, max_iter=1000)
            with SuppressOutput():
                f1_scores = []
                for train_idx_full, valid_idx_full in cv_outer.split(X_transformed_final, y_encoded):
                    X_train_full, X_valid_full = X_transformed_final.iloc[train_idx_full], X_transformed_final.iloc[valid_idx_full]
                    y_train_full, y_valid_full = y_encoded[train_idx_full], y_encoded[valid_idx_full]
                    try:
                        model.fit(X_train_full, y_train_full)
                        y_pred_full = model.predict(X_valid_full)
                        f1 = f1_score(y_valid_full, y_pred_full, average='weighted')
                        f1_scores.append(f1)
                    except (ValueError, ArpackError, NotImplementedError):
                        f1_scores.append(0.0)
                return np.mean(f1_scores)

        # Create an Optuna study for the entire dataset with t-SNE
        study_full_tsne = optuna.create_study(direction='maximize', sampler=TPESampler(seed=1234))
        study_full_tsne.optimize(objective_full_tsne, n_trials=50, show_progress_bar=True)

        # Best hyperparameters from the entire dataset
        best_params_full_tsne = study_full_tsne.best_params
        print(f"Best parameters for Logistic Regression with t-SNE: {best_params_full_tsne}")

        # Initialize the best model with the best hyperparameters
        best_C_full_tsne = best_params_full_tsne.get('C', 1.0)
        best_model = LogisticRegression(C=best_C_full_tsne, random_state=1234, max_iter=1000)

        # Fit the model on the entire t-SNE transformed dataset
        with SuppressOutput():
            try:
                best_model.fit(X_transformed_final, y_encoded)
            except (ValueError, ArpackError, NotImplementedError) as e:
                print(f"Error fitting the final model with t-SNE: {e}")
                sys.exit(1)

        # Save the best model and transformed data
        joblib.dump(best_model, f"{prefix}_logistic_regression_model.pkl")
        joblib.dump((X_transformed_final, y_encoded, le), f"{prefix}_logistic_regression_data.pkl")

        # Output the best parameters
        print(f"Best parameters for Logistic Regression with t-SNE: {best_params_full_tsne}")

        # Save the transformed data
        X_transformed_df = pd.DataFrame(X_transformed_final, columns=[f"TSNE_Component_{i+1}" for i in range(X_transformed_final.shape[1])])
        X_transformed_df.insert(0, 'SampleID', sample_ids)
        X_transformed_df['Label'] = y
        transformed_csv_path = f"{prefix}_logistic_regression_transformed_X_tsne.csv"
        X_transformed_df.to_csv(transformed_csv_path, index=False)
        print(f"t-SNE transformed data saved to {transformed_csv_path}")

        # No variance information available for t-SNE
        variance_csv_path = f"{prefix}_logistic_regression_variance.csv"
        with open(variance_csv_path, 'w') as f:
            f.write("t-SNE does not provide explained variance information.\n")
        print(f"No variance information available for t-SNE. File created at {variance_csv_path}")

        # Prediction and Evaluation
        try:
            y_pred_prob = best_model.predict_proba(X_transformed_final)
            y_pred_class = best_model.predict(X_transformed_final)
        except (ValueError, ArpackError, NotImplementedError) as e:
            print(f"Error during prediction with t-SNE transformed data: {e}")
            y_pred_class = np.zeros_like(y_encoded)
            y_pred_prob = np.zeros((len(y_encoded), num_classes))

        # Calculate performance metrics
        acc = accuracy_score(y_encoded, y_pred_class)
        f1 = f1_score(y_encoded, y_pred_class, average='weighted')
        cm = confusion_matrix(y_encoded, y_pred_class)

        # Calculate sensitivity and specificity
        if num_classes == 2:
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            else:
                # Handle cases where one class might be missing in predictions
                sensitivity = 0
                specificity = 0
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                sensitivity = np.mean(np.divide(np.diag(cm), np.sum(cm, axis=1),
                                                out=np.zeros_like(np.diag(cm), dtype=float),
                                                where=np.sum(cm, axis=1)!=0))
                specificity = np.mean(np.divide(np.diag(cm), np.sum(cm, axis=0),
                                                out=np.zeros_like(np.diag(cm), dtype=float),
                                                where=np.sum(cm, axis=0)!=0))

        # Plot confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix for Logistic Regression with t-SNE')
        plt.savefig(f"{prefix}_logistic_regression_confusion_matrix.png", dpi=300)
        plt.close()

        # ROC and AUC
        fpr_dict = {}
        tpr_dict = {}
        roc_auc_dict = {}

        # Different handling for binary and multi-class cases
        if num_classes == 2:
            try:
                fpr_dict[0], tpr_dict[0], _ = roc_curve(y_binarized[:, 0], y_pred_prob[:, 1])
                roc_auc_dict[0] = auc(fpr_dict[0], tpr_dict[0])
            except ValueError:
                roc_auc_dict[0] = 0.0
        else:
            for i in range(y_binarized.shape[1]):
                try:
                    fpr_dict[i], tpr_dict[i], _ = roc_curve(y_binarized[:, i], y_pred_prob[:, i])
                    roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
                except ValueError:
                    fpr_dict[i], tpr_dict[i], roc_auc_dict[i] = np.array([0, 1]), np.array([0, 1]), 0.0

            # Compute overall ROC AUC for multi-class
            try:
                fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(y_binarized.ravel(), y_pred_prob.ravel())
                roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])
            except ValueError:
                fpr_dict["micro"], tpr_dict["micro"], roc_auc_dict["micro"] = np.array([0, 1]), np.array([0, 1]), 0.0

        # Save ROC data
        roc_data = {
            'fpr': fpr_dict,
            'tpr': tpr_dict,
            'roc_auc': roc_auc_dict
        }
        np.save(f"{prefix}_logistic_regression_roc_data.npy", roc_data)

        # Plot and save ROC curve
        plt.figure(figsize=(10, 8))

        if num_classes == 2:
            plt.plot(fpr_dict[0], tpr_dict[0], label=f'AUC = {roc_auc_dict[0]:.2f}')
        else:
            for i in range(len(le.classes_)):
                if i in roc_auc_dict and roc_auc_dict[i] > 0.0:
                    plt.plot(fpr_dict[i], tpr_dict[i], label=f'{le.inverse_transform([i])[0]} (AUC = {roc_auc_dict[i]:.2f})')
            if "micro" in roc_auc_dict and roc_auc_dict["micro"] > 0.0:
                plt.plot(fpr_dict["micro"], tpr_dict["micro"], label=f'Overall (AUC = {roc_auc_dict["micro"]:.2f})', linestyle='--')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Logistic Regression with t-SNE')
        plt.legend(loc="lower right")
        plt.savefig(f'{prefix}_logistic_regression_roc_curve.png', dpi=300)
        plt.close()

        # Output performance metrics as a bar chart
        metrics = {'Accuracy': acc, 'F1 Score': f1, 'Sensitivity': sensitivity, 'Specificity': specificity}
        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        ax = metrics_df.plot(kind='bar', x='Metric', y='Value', legend=False)
        plt.title('Performance Metrics for Logistic Regression with t-SNE')
        plt.ylabel('Value')
        plt.ylim(0, 1)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{prefix}_logistic_regression_metrics.png', dpi=300)
        plt.close()

        # Create a DataFrame for predictions
        predictions_df = pd.DataFrame({
            'SampleID': sample_ids,
            'Original Label': y,
            'Predicted Label': le.inverse_transform(y_pred_class)
        })

        # Save predictions to CSV
        predictions_df.to_csv(f"{prefix}_logistic_regression_predictions.csv", index=False)

        print(f"Predictions saved to {prefix}_logistic_regression_predictions.csv")

    else:
        # Handle non-t-SNE feature selection methods
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
        plt.savefig(f"{prefix}_logistic_regression_nested_cv_f1_auc.png", dpi=300)
        plt.close()

        print("Nested cross-validation completed.")
        print(f"Average F1 Score: {np.mean(outer_f1_scores):.4f} ± {np.std(outer_f1_scores):.4f}")
        print(f"Average AUC: {np.mean(outer_auc_scores):.4f} ± {np.std(outer_auc_scores):.4f}")

        # After nested CV, perform hyperparameter tuning on the entire dataset
        print("Starting hyperparameter tuning on the entire dataset...")

        # Define the objective function for Optuna on the entire dataset
        def objective_full(trial):
            steps = []

            # Add feature selection based on the method
            if feature_selection_method != 'none':
                if feature_selection_method == 'pca':
                    pca_n_components = trial.suggest_int('pca_n_components', 1, min(X.shape[1], X.shape[0]-1))
                    steps.append(('feature_selection', PCA(n_components=pca_n_components, random_state=1234)))
                elif feature_selection_method == 'kpca':
                    kpca_kernel = trial.suggest_categorical('kpca_kernel', ['linear', 'poly', 'rbf', 'sigmoid', 'cosine'])
                    kpca_n_components = trial.suggest_int('kpca_n_components', 1, min(X.shape[1], X.shape[0]-1))
                    kpca_params = {
                        'n_components': kpca_n_components,
                        'kernel': kpca_kernel,
                        'random_state': 1234,
                        'eigen_solver': 'arpack',
                        'max_iter': 5000
                    }
                    if kpca_kernel in ['poly', 'rbf', 'sigmoid']:
                        kpca_gamma = trial.suggest_loguniform('kpca_gamma', 1e-4, 1e1)
                        kpca_params['gamma'] = kpca_gamma
                    if kpca_kernel in ['poly', 'sigmoid']:
                        kpca_coef0 = trial.suggest_uniform('kpca_coef0', 0.0, 1.0)
                        kpca_params['coef0'] = kpca_coef0
                    if kpca_kernel == 'poly':
                        kpca_degree = trial.suggest_int('kpca_degree', 2, 5)
                        kpca_params['degree'] = kpca_degree
                    steps.append(('feature_selection', KernelPCA(**kpca_params)))
                elif feature_selection_method == 'umap':
                    umap_n_components = trial.suggest_int('umap_n_components', 2, min(100, X.shape[1]))
                    umap_n_neighbors = trial.suggest_int('umap_n_neighbors', 5, min(50, X.shape[0]-1))
                    umap_min_dist = trial.suggest_uniform('umap_min_dist', 0.0, 0.99)
                    steps.append(('feature_selection', umap.UMAP(
                        n_components=umap_n_components,
                        n_neighbors=umap_n_neighbors,
                        min_dist=umap_min_dist,
                        random_state=1234
                    )))
                elif feature_selection_method == 'pls':
                    pls_n_components = trial.suggest_int('pls_n_components', 1, min(X.shape[1], X.shape[0]-1))
                    steps.append(('feature_selection', PLSFeatureSelector(
                        n_components=pls_n_components,
                        max_iter=1000,
                        tol=1e-06
                    )))
                elif feature_selection_method == 'elasticnet':
                    elasticnet_alpha = trial.suggest_loguniform('elasticnet_alpha', 1e-4, 1e1)
                    l1_ratio = trial.suggest_uniform('elasticnet_l1_ratio', 0.0, 1.0)
                    steps.append(('feature_selection', ElasticNetFeatureSelector(
                        alpha=elasticnet_alpha,
                        l1_ratio=l1_ratio,
                        max_iter=10000,
                        tol=1e-4
                    )))

            # Add Logistic Regression to the pipeline
            C = trial.suggest_loguniform('C', 1e-2, 100)
            steps.append(('logreg', LogisticRegression(C=C, random_state=1234, max_iter=1000)))

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
                    except (ValueError, ArpackError, NotImplementedError):
                        # If feature selection or Logistic Regression fails
                        f1_scores.append(0.0)
                return np.mean(f1_scores)

        # Create an Optuna study for the entire dataset
        study_full = optuna.create_study(direction='maximize', sampler=TPESampler(seed=1234))
        study_full.optimize(objective_full, n_trials=50, show_progress_bar=True)

        # Best hyperparameters from the entire dataset
        best_params_full = study_full.best_params
        print(f"Best parameters for Logistic Regression: {best_params_full}")

        # Initialize the best model with the best hyperparameters
        steps = []

        # Add feature selection based on the method
        if feature_selection_method != 'none':
            if feature_selection_method == 'pca':
                best_pca_n_components_full = best_params_full.get('pca_n_components', 2)
                steps.append(('feature_selection', PCA(n_components=best_pca_n_components_full, random_state=1234)))
            elif feature_selection_method == 'kpca':
                best_kpca_kernel_full = best_params_full.get('kpca_kernel', 'rbf')
                best_kpca_n_components_full = best_params_full.get('kpca_n_components', 2)
                kpca_params = {
                    'n_components': best_kpca_n_components_full,
                    'kernel': best_kpca_kernel_full,
                    'random_state': 1234,
                    'eigen_solver': 'arpack',
                    'max_iter': 5000
                }
                if best_kpca_kernel_full in ['poly', 'rbf', 'sigmoid']:
                    kpca_params['gamma'] = best_params_full.get('kpca_gamma', 1.0)
                if best_kpca_kernel_full in ['poly', 'sigmoid']:
                    kpca_params['coef0'] = best_params_full.get('kpca_coef0', 0.0)
                if best_kpca_kernel_full == 'poly':
                    kpca_params['degree'] = best_params_full.get('kpca_degree', 3)
                steps.append(('feature_selection', KernelPCA(**kpca_params)))
            elif feature_selection_method == 'umap':
                best_umap_n_components_full = best_params_full.get('umap_n_components', 2)
                best_umap_n_neighbors_full = best_params_full.get('umap_n_neighbors', 15)
                best_umap_min_dist_full = best_params_full.get('umap_min_dist', 0.1)
                steps.append(('feature_selection', umap.UMAP(
                    n_components=best_umap_n_components_full,
                    n_neighbors=best_umap_n_neighbors_full,
                    min_dist=best_umap_min_dist_full,
                    random_state=1234
                )))
            elif feature_selection_method == 'pls':
                best_pls_n_components_full = best_params_full.get('pls_n_components', 2)
                steps.append(('feature_selection', PLSFeatureSelector(
                    n_components=best_pls_n_components_full,
                    max_iter=1000,
                    tol=1e-06
                )))
            elif feature_selection_method == 'elasticnet':
                best_elasticnet_alpha_full = best_params_full.get('elasticnet_alpha', 1.0)
                best_l1_ratio_full = best_params_full.get('elasticnet_l1_ratio', 0.5)
                steps.append(('feature_selection', ElasticNetFeatureSelector(
                    alpha=best_elasticnet_alpha_full,
                    l1_ratio=best_l1_ratio_full,
                    max_iter=10000,
                    tol=1e-4
                )))

        # Add Logistic Regression to the pipeline
        if not tsne_selected:
            best_C_full = best_params_full.get('C', 1.0)
            steps.append(('logreg', LogisticRegression(C=best_C_full, random_state=1234, max_iter=1000)))
            best_model = Pipeline(steps)
        else:
            # For t-SNE, the data has already been transformed outside the pipeline
            # Thus, we only add Logistic Regression
            best_C_full_tsne = best_params_full_tsne.get('C', 1.0)
            best_model = Pipeline([
                ('logreg', LogisticRegression(C=best_C_full_tsne, random_state=1234, max_iter=1000))
            ])

        # Fit the model on the entire dataset
        with SuppressOutput():
            try:
                if tsne_selected:
                    best_model.fit(X_transformed_final, y_encoded)
                else:
                    best_model.fit(X, y_encoded)
            except (ValueError, ArpackError, NotImplementedError) as e:
                print(f"Error fitting the final model: {e}")
                sys.exit(1)

        # Save the best model and data
        if tsne_selected:
            joblib.dump(best_model, f"{prefix}_logistic_regression_model.pkl")
            joblib.dump((X_transformed_final, y_encoded, le), f"{prefix}_logistic_regression_data.pkl")
        else:
            joblib.dump(best_model, f"{prefix}_logistic_regression_model.pkl")
            joblib.dump((X, y_encoded, le), f"{prefix}_logistic_regression_data.pkl")

        # Output the best parameters
        if tsne_selected:
            print(f"Best parameters for Logistic Regression with t-SNE: {best_params_full_tsne}")
        else:
            print(f"Best parameters for Logistic Regression: {best_params_full}")

        # If feature selection is used and not t-SNE, save the transformed data and variance information
        if feature_selection_method != 'none' and not tsne_selected:
            if 'feature_selection' in best_model.named_steps:
                try:
                    X_transformed = best_model.named_steps['feature_selection'].transform(X)
                except (NotImplementedError, ArpackError, ValueError):
                    X_transformed = None
            else:
                X_transformed = None

            if X_transformed is not None:
                if feature_selection_method in ['pca', 'kpca', 'umap', 'pls']:
                    n_components = X_transformed.shape[1]
                    transformed_columns = [f"{feature_selection_method.upper()}_Component_{i+1}" for i in range(n_components)]
                    X_transformed_df = pd.DataFrame(X_transformed, columns=transformed_columns)
                elif feature_selection_method == 'elasticnet':
                    selected_features_mask = best_model.named_steps['feature_selection'].selector.get_support()
                    selected_features = X.columns[selected_features_mask]
                    X_transformed_df = pd.DataFrame(X[selected_features].values, columns=selected_features)
                else:
                    X_transformed_df = pd.DataFrame(X_transformed)

                X_transformed_df.insert(0, 'SampleID', sample_ids)
                X_transformed_df['Label'] = y
                transformed_csv_path = f"{prefix}_logistic_regression_transformed_X.csv"
                X_transformed_df.to_csv(transformed_csv_path, index=False)
                print(f"Transformed data saved to {transformed_csv_path}")

                variance_csv_path = f"{prefix}_logistic_regression_variance.csv"
                if feature_selection_method == 'pca':
                    # Fit a full PCA to get all components' variance
                    full_pca = PCA(n_components=X.shape[1], random_state=1234)
                    with SuppressOutput():
                        full_pca.fit(X)
                    explained_variance = full_pca.explained_variance_ratio_
                    explained_variance_df = pd.DataFrame({
                        'Component': range(1, len(explained_variance)+1),
                        'Explained Variance Ratio': explained_variance
                    })
                    explained_variance_df.to_csv(variance_csv_path, index=False)
                    print(f"PCA explained variance ratios saved to {variance_csv_path}")
                elif feature_selection_method == 'pls':
                    # Fit a full PLS model to get all components' variance
                    y_full = label_binarize(y_encoded, classes=np.unique(y_encoded))
                    if y_full.ndim == 1:
                        y_full = np.vstack([1 - y_full, y_full]).T
                    pls_n_components_full = min(X.shape[1], X.shape[0]-1)
                    full_pls = PLSRegression(n_components=pls_n_components_full, max_iter=1000, tol=1e-06)
                    with SuppressOutput():
                        full_pls.fit(X, y_full)
                    x_scores = full_pls.x_scores_
                    explained_variance = np.var(x_scores, axis=0) / np.var(X, axis=0).sum()
                    explained_variance_df = pd.DataFrame({
                        'Component': range(1, len(explained_variance)+1),
                        'Explained Variance Ratio': explained_variance
                    })
                    explained_variance_df.to_csv(variance_csv_path, index=False)
                    print(f"PLS explained variance ratios saved to {variance_csv_path}")
                elif feature_selection_method in ['kpca']:
                    # KernelPCA does not provide explained variance directly
                    with open(variance_csv_path, 'w') as f:
                        f.write("KernelPCA does not provide explained variance information.\n")
                    print(f"No variance information available for KernelPCA. File created at {variance_csv_path}")
                elif feature_selection_method == 'umap':
                    with open(variance_csv_path, 'w') as f:
                        f.write("UMAP does not provide explained variance information.\n")
                    print(f"No variance information available for UMAP. File created at {variance_csv_path}")
                elif feature_selection_method == 'elasticnet':
                    with open(variance_csv_path, 'w') as f:
                        f.write("ElasticNet does not provide explained variance information.\n")
                    print(f"No variance information available for ElasticNet. File created at {variance_csv_path}")
                else:
                    with open(variance_csv_path, 'w') as f:
                        f.write(f"{feature_selection_method.upper()} does not provide explained variance information.\n")
                    print(f"No variance information available for {feature_selection_method.upper()}. File created at {variance_csv_path}")
            else:
                print("Feature selection failed or no feature selection method selected. Skipping transformed data and variance information saving.")

        # Prediction using cross_val_predict if not using t-SNE
        if not tsne_selected:
            y_pred_prob = None
            try:
                y_pred_prob = cross_val_predict(best_model, X, y_encoded, cv=cv_outer, method='predict_proba', n_jobs=-1)
                y_pred_class = np.argmax(y_pred_prob, axis=1)
            except (ValueError, ArpackError, NotImplementedError) as e:
                print(f"Error during cross_val_predict: {e}")
                y_pred_class = np.zeros_like(y_encoded)
                y_pred_prob = np.zeros((len(y_encoded), num_classes))

            # Calculate performance metrics
            acc = accuracy_score(y_encoded, y_pred_class)
            f1 = f1_score(y_encoded, y_pred_class, average='weighted')
            cm = confusion_matrix(y_encoded, y_pred_class)

            # Calculate sensitivity and specificity
            if num_classes == 2:
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                else:
                    # Handle cases where one class might be missing in predictions
                    sensitivity = 0
                    specificity = 0
            else:
                with np.errstate(divide='ignore', invalid='ignore'):
                    sensitivity = np.mean(np.divide(np.diag(cm), np.sum(cm, axis=1),
                                                    out=np.zeros_like(np.diag(cm), dtype=float),
                                                    where=np.sum(cm, axis=1)!=0))
                    specificity = np.mean(np.divide(np.diag(cm), np.sum(cm, axis=0),
                                                    out=np.zeros_like(np.diag(cm), dtype=float),
                                                    where=np.sum(cm, axis=0)!=0))

            # Plot confusion matrix
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
            disp.plot(cmap=plt.cm.Blues)
            plt.title('Confusion Matrix for Logistic Regression')
            plt.savefig(f"{prefix}_logistic_regression_confusion_matrix.png", dpi=300)
            plt.close()

            # ROC and AUC
            fpr_dict = {}
            tpr_dict = {}
            roc_auc_dict = {}

            # Different handling for binary and multi-class cases
            if num_classes == 2:
                try:
                    fpr_dict[0], tpr_dict[0], _ = roc_curve(y_binarized[:, 0], y_pred_prob[:, 1])
                    roc_auc_dict[0] = auc(fpr_dict[0], tpr_dict[0])
                except ValueError:
                    roc_auc_dict[0] = 0.0
            else:
                for i in range(y_binarized.shape[1]):
                    try:
                        fpr_dict[i], tpr_dict[i], _ = roc_curve(y_binarized[:, i], y_pred_prob[:, i])
                        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
                    except ValueError:
                        fpr_dict[i], tpr_dict[i], roc_auc_dict[i] = np.array([0, 1]), np.array([0, 1]), 0.0

                # Compute overall ROC AUC for multi-class
                try:
                    fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(y_binarized.ravel(), y_pred_prob.ravel())
                    roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])
                except ValueError:
                    fpr_dict["micro"], tpr_dict["micro"], roc_auc_dict["micro"] = np.array([0, 1]), np.array([0, 1]), 0.0

            # Save ROC data
            roc_data = {
                'fpr': fpr_dict,
                'tpr': tpr_dict,
                'roc_auc': roc_auc_dict
            }
            np.save(f"{prefix}_logistic_regression_roc_data.npy", roc_data)

            # Plot and save ROC curve
            plt.figure(figsize=(10, 8))

            if num_classes == 2:
                plt.plot(fpr_dict[0], tpr_dict[0], label=f'AUC = {roc_auc_dict[0]:.2f}')
            else:
                for i in range(len(le.classes_)):
                    if i in roc_auc_dict and roc_auc_dict[i] > 0.0:
                        plt.plot(fpr_dict[i], tpr_dict[i], label=f'{le.inverse_transform([i])[0]} (AUC = {roc_auc_dict[i]:.2f})')
                if "micro" in roc_auc_dict and roc_auc_dict["micro"] > 0.0:
                    plt.plot(fpr_dict["micro"], tpr_dict["micro"], label=f'Overall (AUC = {roc_auc_dict["micro"]:.2f})', linestyle='--')

            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves for Logistic Regression')
            plt.legend(loc="lower right")
            plt.savefig(f'{prefix}_logistic_regression_roc_curve.png', dpi=300)
            plt.close()

            # Output performance metrics as a bar chart
            metrics = {'Accuracy': acc, 'F1 Score': f1, 'Sensitivity': sensitivity, 'Specificity': specificity}
            metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
            ax = metrics_df.plot(kind='bar', x='Metric', y='Value', legend=False)
            plt.title('Performance Metrics for Logistic Regression')
            plt.ylabel('Value')
            plt.ylim(0, 1)
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f'{prefix}_logistic_regression_metrics.png', dpi=300)
            plt.close()

            # Create a DataFrame for predictions
            predictions_df = pd.DataFrame({
                'SampleID': sample_ids,
                'Original Label': y,
                'Predicted Label': le.inverse_transform(y_pred_class)
            })

            # Save predictions to CSV
            predictions_df.to_csv(f"{prefix}_logistic_regression_predictions.csv", index=False)

            print(f"Predictions saved to {prefix}_logistic_regression_predictions.csv")

def main():
    parser = argparse.ArgumentParser(description='Run Logistic Regression with Nested Cross-Validation, Feature Selection (PCA, KPCA, UMAP, t-SNE, PLS, ElasticNet), and Optuna hyperparameter optimization.')
    parser.add_argument('-i', '--csv', type=str, help='Input file in CSV format', required=True)
    parser.add_argument('-p', '--prefix', type=str, help='Prefix for output files', required=True)
    parser.add_argument('-f', '--feature_selection', type=str, choices=['none', 'elasticnet', 'pca', 'kpca', 'umap', 'tsne', 'pls'], default='none', help='Feature selection method to use. Options: none, elasticnet, pca, kpca, umap, tsne, pls.')

    args = parser.parse_args()

    # Inform the user if t-SNE is selected
    if args.feature_selection == 'tsne':
        print("Warning: t-SNE does not support transforming new data. The entire dataset will be transformed before cross-validation, which may lead to data leakage.")

    # Run the logistic regression nested cross-validation function
    logistic_regression_nested_cv(args.csv, args.prefix, args.feature_selection)

if __name__ == '__main__':
    main()