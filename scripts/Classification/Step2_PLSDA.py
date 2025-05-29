import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, label_binarize, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.cross_decomposition import PLSRegression
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
from optuna.exceptions import TrialPruned
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.linear_model import ElasticNet
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
import umap
from scipy.sparse.linalg import ArpackError

# Suppress specific warnings except ConvergenceWarning
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

# Custom PLSDA classifier
class PLSDAClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=2, max_iter=1000):
        self.n_components = n_components
        self.max_iter = max_iter
        self.pls = None
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        y_onehot = label_binarize(y, classes=self.classes_)
        if y_onehot.ndim == 1:
            # Binary case
            y_onehot = np.vstack([1 - y_onehot, y_onehot]).T
        self.pls = PLSRegression(n_components=self.n_components, max_iter=self.max_iter, tol=1e-06)
        self.pls.fit(X, y_onehot)
        return self

    def predict(self, X):
        y_pred_continuous = self.pls.predict(X)
        if y_pred_continuous.shape[1] > 1:
            y_pred = self.classes_[np.argmax(y_pred_continuous, axis=1)]
        else:
            # Binary case
            y_bin = (y_pred_continuous >= 0.5).astype(int).ravel()
            y_pred = self.classes_[y_bin]
        return y_pred

    def predict_proba(self, X):
        y_pred_continuous = self.pls.predict(X)
        y_pred_proba = np.maximum(y_pred_continuous, 0)
        if y_pred_proba.shape[1] > 1:
            sum_probs = y_pred_proba.sum(axis=1, keepdims=True)
            y_pred_proba = y_pred_proba / sum_probs
        else:
            # Binary case
            prob_class1 = y_pred_proba.ravel()
            prob_class0 = 1 - prob_class1
            y_pred_proba = np.vstack([prob_class0, prob_class1]).T
        return y_pred_proba

# ElasticNet feature selector
class ElasticNetFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=10000, tol=1e-4, random_state=1234):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.selector = SelectFromModel(
            ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio,
                      max_iter=self.max_iter, tol=self.tol,
                      random_state=self.random_state)
        )

    def fit(self, X, y):
        self.selector.fit(X, y)
        return self

    def transform(self, X):
        return self.selector.transform(X)

# UMAP transformer (no additional transformer class needed since UMAP is used directly in the pipeline)

def safe_umap(n_components, n_neighbors, min_dist, X, random_state=1234):
    n_samples = X.shape[0]
    n_components = min(n_components, max(1, n_samples - 1))
    n_neighbors = min(n_neighbors, max(2, n_samples - 2))
    return umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        init='random'
    )

def plsda_nested_cv(inp, prefix, feature_selection_method):
    # Read data
    data = pd.read_csv(inp)

    # Ensure 'SampleID' and 'Label' columns exist
    if 'SampleID' not in data.columns or 'Label' not in data.columns:
        raise ValueError("Input data must contain 'SampleID' and 'Label' columns.")

    sample_ids = data['SampleID']
    X = data.drop(columns=['SampleID', 'Label']).copy()
    y = data['Label']

    # Standardization
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    # X = pd.DataFrame(X_scaled, columns=X.columns)

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

    # Iterate over each outer fold
    fold_idx = 1
    for train_idx, test_idx in cv_outer.split(X, y_encoded):
        print(f"Processing outer fold {fold_idx}...")
        X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
        y_train_outer, y_test_outer = y_encoded[train_idx], y_encoded[test_idx]

        # Define inner cross-validation strategy
        cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

        # Define the objective function for Optuna within the outer fold
        def objective_inner(trial):
            steps = [('scaler', StandardScaler())]

            # Add feature selection based on the method
            if feature_selection_method == 'elasticnet':
                # Suggest hyperparameters for ElasticNet
                elasticnet_alpha = trial.suggest_loguniform('elasticnet_alpha', 1e-4, 1e1)
                l1_ratio = trial.suggest_uniform('elasticnet_l1_ratio', 0.0, 1.0)
                steps.append(('feature_selection', ElasticNetFeatureSelector(
                    alpha=elasticnet_alpha,
                    l1_ratio=l1_ratio,
                    max_iter=10000,
                    tol=1e-4,
                    random_state=1234
                )))
                max_n_components = min(10, X_train_outer.shape[1], X_train_outer.shape[0])
            elif feature_selection_method == 'umap':
                # Suggest hyperparameters for UMAP
                umap_n_components = trial.suggest_int('umap_n_components', 2, min(100, X_train_outer.shape[1]))
                umap_n_neighbors = trial.suggest_int('umap_n_neighbors', 5, min(50, X_train_outer.shape[0]-1))
                umap_min_dist = trial.suggest_uniform('umap_min_dist', 0.0, 0.99)
                steps.append(('feature_selection', safe_umap(
                    n_components=umap_n_components,
                    n_neighbors=umap_n_neighbors,
                    min_dist=umap_min_dist,
                    X=X_train_outer
                )))

                max_n_components = umap_n_components
            else:
                # No feature selection
                max_n_components = min(100, X_train_outer.shape[1], X_train_outer.shape[0])
                if max_n_components < 1:
                    max_n_components = 1

            # Suggest hyperparameters for PLSDA
            plsda_n_components = trial.suggest_int('plsda_n_components', 1, max_n_components)
            steps.append(('plsda', PLSDAClassifier(n_components=plsda_n_components, max_iter=1000)))

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
                    except (ValueError, ArpackError):
                        # If feature selection or PLSDA fails
                        f1_scores.append(0.0)
                return np.mean(f1_scores)

        # Create an Optuna study for the inner fold
        study_inner = optuna.create_study(direction='maximize', sampler=TPESampler(seed=1234))
        study_inner.optimize(objective_inner, n_trials=50, show_progress_bar=False)

        # Best hyperparameters from the inner fold
        best_params_inner = study_inner.best_params

        # Initialize the best model with the best hyperparameters
        steps = [('scaler', StandardScaler())]

        if feature_selection_method == 'elasticnet':
            best_elasticnet_alpha = best_params_inner.get('elasticnet_alpha', 1.0)
            best_l1_ratio = best_params_inner.get('elasticnet_l1_ratio', 0.5)
            steps.append(('feature_selection', ElasticNetFeatureSelector(
                alpha=best_elasticnet_alpha,
                l1_ratio=best_l1_ratio,
                max_iter=10000,
                tol=1e-4,
                random_state=1234
            )))
            max_n_components = min(10, X_train_outer.shape[1], X_train_outer.shape[0])
        elif feature_selection_method == 'umap':
            best_umap_n_components = best_params_inner.get('umap_n_components', 2)
            best_umap_n_neighbors = best_params_inner.get('umap_n_neighbors', 15)
            best_umap_min_dist = best_params_inner.get('umap_min_dist', 0.1)
            steps.append(('feature_selection', safe_umap(
                n_components=best_umap_n_components,
                n_neighbors=best_umap_n_neighbors,
                min_dist=best_umap_min_dist,
                X=X_train_outer
            )))

            max_n_components = best_umap_n_components
        else:
            max_n_components = min(100, X_train_outer.shape[1], X_train_outer.shape[0])
            if max_n_components < 1:
                max_n_components = 1

        # Get the best PLSDA n_components
        best_plsda_n_components = best_params_inner.get('plsda_n_components', 2)
        steps.append(('plsda', PLSDAClassifier(n_components=best_plsda_n_components, max_iter=1000)))

        # Create the best model pipeline
        best_model_inner = Pipeline(steps)

        # Fit the model on the outer training set
        with SuppressOutput():
            try:
                best_model_inner.fit(X_train_outer, y_train_outer)
            except (ValueError, ArpackError) as e:
                print(f"Error fitting the model in outer fold {fold_idx}: {e}")
                outer_f1_scores.append(0)
                outer_auc_scores.append(0)
                fold_idx += 1
                continue

        # Predict probabilities and classes on the outer test set
        try:
            y_pred_prob_outer = best_model_inner.predict_proba(X_test_outer)
            y_pred_class_outer = best_model_inner.predict(X_test_outer)
        except (ValueError, ArpackError) as e:
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
    plt.savefig(f"{prefix}_plsda_nested_cv_f1_auc.png", dpi=300)
    plt.close()

    print("Nested cross-validation completed.")
    print(f"Average F1 Score: {np.mean(outer_f1_scores):.4f} ± {np.std(outer_f1_scores):.4f}")
    print(f"Average AUC: {np.mean(outer_auc_scores):.4f} ± {np.std(outer_auc_scores):.4f}")

    # After nested CV, perform hyperparameter tuning on the entire dataset
    print("Starting hyperparameter tuning on the entire dataset...")

    # Define the objective function for Optuna on the entire dataset
    def objective_full(trial):
        steps = [('scaler', StandardScaler())]

        # Add feature selection based on the method
        if feature_selection_method == 'elasticnet':
            # Suggest hyperparameters for ElasticNet
            elasticnet_alpha = trial.suggest_loguniform('elasticnet_alpha', 1e-4, 1e1)
            l1_ratio = trial.suggest_uniform('elasticnet_l1_ratio', 0.0, 1.0)
            steps.append(('feature_selection', ElasticNetFeatureSelector(
                alpha=elasticnet_alpha,
                l1_ratio=l1_ratio,
                max_iter=10000,
                tol=1e-4,
                random_state=1234
            )))
            max_n_components_full = min(10, X.shape[1], X.shape[0])
        elif feature_selection_method == 'umap':
            # Suggest hyperparameters for UMAP
            umap_n_components = trial.suggest_int('umap_n_components', 2, min(100, X.shape[1]))
            umap_n_neighbors = trial.suggest_int('umap_n_neighbors', 5, min(50, X.shape[0]-1))
            umap_min_dist = trial.suggest_uniform('umap_min_dist', 0.0, 0.99)
            steps.append(('feature_selection', safe_umap(
                n_components=umap_n_components,
                n_neighbors=umap_n_neighbors,
                min_dist=umap_min_dist,
                X=X
            )))

            max_n_components_full = umap_n_components
        else:
            # No feature selection
            max_n_components_full = min(100, X.shape[1], X.shape[0])
            if max_n_components_full < 1:
                max_n_components_full = 1

        # Suggest hyperparameters for PLSDA
        plsda_n_components = trial.suggest_int('plsda_n_components', 1, max_n_components_full)
        steps.append(('plsda', PLSDAClassifier(n_components=plsda_n_components, max_iter=1000)))

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
                except (ValueError, ArpackError):
                    # If feature selection or PLSDA fails
                    f1_scores.append(0.0)
            return np.mean(f1_scores)

    # Create an Optuna study for the entire dataset
    study_full = optuna.create_study(direction='maximize', sampler=TPESampler(seed=1234))
    study_full.optimize(objective_full, n_trials=50, show_progress_bar=True)

    # Best hyperparameters from the entire dataset
    best_params_full = study_full.best_params
    print(f"Best parameters for PLSDA: {best_params_full}")

    # Initialize the best model with the best hyperparameters
    steps = [('scaler', StandardScaler())]

    if feature_selection_method == 'elasticnet':
        best_elasticnet_alpha_full = best_params_full.get('elasticnet_alpha', 1.0)
        best_l1_ratio_full = best_params_full.get('elasticnet_l1_ratio', 0.5)
        steps.append(('feature_selection', ElasticNetFeatureSelector(
            alpha=best_elasticnet_alpha_full,
            l1_ratio=best_l1_ratio_full,
            max_iter=10000,
            tol=1e-4,
            random_state=1234
        )))
        max_n_components_full = min(10, X.shape[1], X.shape[0])
    elif feature_selection_method == 'umap':
        best_umap_n_components_full = best_params_full.get('umap_n_components', 2)
        best_umap_n_neighbors_full = best_params_full.get('umap_n_neighbors', 15)
        best_umap_min_dist_full = best_params_full.get('umap_min_dist', 0.1)
        steps.append(('feature_selection', safe_umap(
            n_components=best_umap_n_components_full,
            n_neighbors=best_umap_n_neighbors_full,
            min_dist=best_umap_min_dist_full,
            X=X
        )))

        max_n_components_full = best_umap_n_components_full
    else:
        max_n_components_full = min(100, X.shape[1], X.shape[0])
        if max_n_components_full < 1:
            max_n_components_full = 1

    # Get the best PLSDA n_components
    best_plsda_n_components_full = best_params_full.get('plsda_n_components', 2)
    steps.append(('plsda', PLSDAClassifier(n_components=best_plsda_n_components_full, max_iter=1000)))

    # Create the best model pipeline
    best_model = Pipeline(steps)

    # Fit the model on the entire dataset
    with SuppressOutput():
        try:
            best_model.fit(X, y_encoded)
        except (ValueError, ArpackError) as e:
            print(f"Error fitting the final model: {e}")
            sys.exit(1)

    # Save the best model and data
    joblib.dump(best_model, f"{prefix}_plsda_model.pkl")
    joblib.dump((X, y_encoded, le), f"{prefix}_plsda_data.pkl")

    # Output the best parameters
    print(f"Best parameters for PLSDA: {best_params_full}")

    # If feature selection is used, save the transformed data and variance information
    if feature_selection_method != 'none':
        if 'feature_selection' in best_model.named_steps:
            try:
                X_transformed = best_model.named_steps['feature_selection'].transform(X)
            except (NotImplementedError, ArpackError, ValueError):
                X_transformed = None
        else:
            X_transformed = None

        if X_transformed is not None:
            if feature_selection_method == 'umap':
                n_components = X_transformed.shape[1]
                transformed_columns = [f"UMAP_Component_{i+1}" for i in range(n_components)]
                X_transformed_df = pd.DataFrame(X_transformed, columns=transformed_columns)
            elif feature_selection_method == 'elasticnet':
                selected_features = X.columns[best_model.named_steps['feature_selection'].selector.get_support()]
                X_transformed_df = X[selected_features].copy()
            else:
                X_transformed_df = pd.DataFrame(X_transformed)

            X_transformed_df.insert(0, 'SampleID', sample_ids)
            X_transformed_df['Label'] = y
            transformed_csv_path = f"{prefix}_plsda_transformed_X.csv"
            X_transformed_df.to_csv(transformed_csv_path, index=False)
            print(f"Transformed data saved to {transformed_csv_path}")

            variance_csv_path = f"{prefix}_plsda_variance.csv"
            if feature_selection_method == 'elasticnet':
                with open(variance_csv_path, 'w') as f:
                    f.write("ElasticNet does not provide variance information.\n")
                print(f"No variance information available for ElasticNet. File created at {variance_csv_path}")
            elif feature_selection_method == 'umap':
                with open(variance_csv_path, 'w') as f:
                    f.write("UMAP does not provide variance information.\n")
                print(f"No variance information available for UMAP. File created at {variance_csv_path}")
            else:
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
    except (ValueError, ArpackError) as e:
        print(f"Error during cross_val_predict: {e}")
        y_pred_class = np.zeros_like(y_encoded)

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
    plt.title('Confusion Matrix for PLSDA')
    plt.savefig(f"{prefix}_plsda_confusion_matrix.png", dpi=300)
    plt.close()

    # ROC and AUC
    fpr = {}
    tpr = {}
    roc_auc = {}

    if num_classes == 2:
        try:
            fpr[0], tpr[0], _ = roc_curve(y_binarized[:, 0], y_pred_prob[:, 1])
            roc_auc[0] = auc(fpr[0], tpr[0])
        except ValueError:
            roc_auc[0] = 0.0
    else:
        for i in range(y_binarized.shape[1]):
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
    roc_data = {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}
    np.save(f"{prefix}_plsda_roc_data.npy", roc_data)

    # Plot and save ROC curve
    plt.figure(figsize=(10, 8))

    if num_classes == 2:
        plt.plot(fpr[0], tpr[0], label=f'AUC = {roc_auc[0]:.2f}')
    else:
        for i in range(len(le.classes_)):
            if i in roc_auc and roc_auc[i] > 0.0:
                plt.plot(fpr[i], tpr[i], label=f'{le.inverse_transform([i])[0]} (AUC = {roc_auc[i]:.2f})')
        if "micro" in roc_auc and roc_auc["micro"] > 0.0:
            plt.plot(fpr["micro"], tpr["micro"], label=f'Overall (AUC = {roc_auc["micro"]:.2f})', linestyle='--')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for PLSDA')
    plt.legend(loc="lower right")
    plt.savefig(f'{prefix}_plsda_roc_curve.png', dpi=300)
    plt.close()

    # Output performance metrics as a bar chart
    metrics = {'Accuracy': acc, 'F1 Score': f1, 'Sensitivity': sensitivity, 'Specificity': specificity}
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    ax = metrics_df.plot(kind='bar', x='Metric', y='Value', legend=False)
    plt.title('Performance Metrics for PLSDA')
    plt.ylabel('Value')
    plt.ylim(0, 1)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{prefix}_plsda_metrics.png', dpi=300)
    plt.close()

    # Create a DataFrame for predictions
    predictions_df = pd.DataFrame({
        'SampleID': sample_ids,
        'Original Label': y,
        'Predicted Label': le.inverse_transform(y_pred_class)
    })

    # Save predictions to CSV
    predictions_df.to_csv(f"{prefix}_plsda_predictions.csv", index=False)
    print(f"Predictions saved to {prefix}_plsda_predictions.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run PLSDA with Nested Cross-Validation, Feature Selection (UMAP or ElasticNet), and Optuna hyperparameter optimization.')
    parser.add_argument('-i', type=str, required=True, help='Input file in CSV format')
    parser.add_argument('-p', type=str, required=True, help='Output prefix')
    parser.add_argument('-f', type=str, choices=['none', 'elasticnet', 'umap'], default='none', help='Feature selection method to use. Options: none, elasticnet, umap.')
    args = parser.parse_args()

    plsda_nested_cv(args.i, args.p, args.f)
