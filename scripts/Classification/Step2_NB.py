import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, label_binarize, StandardScaler, LabelBinarizer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
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
from optuna.exceptions import TrialPruned
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
import umap
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import ConvergenceWarning
from scipy.sparse.linalg import ArpackError
from sklearn.utils.class_weight import compute_class_weight

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# Font helper to unify style with other models
def enlarge_fonts(ax, label=22, ticks=18, title=24, legend=20, legend_title=22):
    ax.xaxis.label.set_size(label)
    ax.yaxis.label.set_size(label)
    if hasattr(ax, "title"):
        ax.title.set_size(title)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontsize(ticks)
    leg = ax.get_legend()
    if leg is not None:
        for txt in leg.get_texts():
            txt.set_fontsize(legend)
        if leg.get_title() is not None:
            leg.get_title().set_fontsize(legend_title)

# Suppress stdout/stderr in blocks
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

# Multiclass specificity helper
def multiclass_specificity(cm: np.ndarray) -> float:
    cm = cm.astype(float)
    K = cm.shape[0]
    specs = []
    for i in range(K):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return np.mean(specs) if len(specs) > 0 else 0.0

# PLS feature selector with label binarization and component safety
class PLSFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, max_iter=1000, tol=1e-06):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.pls = None

    def fit(self, X, y):
        self.feature_names_ = X.columns if hasattr(X, "columns") else None
        X_array = X.values if hasattr(X, "values") else X
        lb = LabelBinarizer()
        Y = lb.fit_transform(y)
        if Y.ndim == 1:
            Y = np.vstack([1 - Y, Y]).T
        n_comp = max(1, min(self.n_components, X_array.shape[1], X_array.shape[0] - 1))
        self.pls = PLSRegression(n_components=n_comp, max_iter=self.max_iter, tol=self.tol)
        self.pls.fit(X_array, Y)
        return self

    def transform(self, X):
        X_array = X.values if hasattr(X, "values") else X
        X_pls = self.pls.transform(X_array)
        return pd.DataFrame(
            X_pls,
            index=X.index if hasattr(X, "index") else None,
            columns=[f"PLS_Component_{i+1}" for i in range(X_pls.shape[1])]
        )

# TSNE transformer is kept for completeness but not used inside CV pipelines
class TSNETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, perplexity=30, learning_rate=200, max_iter=1000):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tsne = TSNE(
            n_components=self.n_components,
            perplexity=self.perplexity,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            random_state=1234
        )
        self.X_transformed_ = None

    def fit(self, X, y=None):
        self.X_transformed_ = self.tsne.fit_transform(X)
        return self

    def transform(self, X):
        if self.X_transformed_ is not None and X.shape[0] == self.X_transformed_.shape[0]:
            return self.X_transformed_
        else:
            return X

# ElasticNet-like feature selector using LogisticRegression with elastic-net penalty
class ElasticNetFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, C=1.0, l1_ratio=0.5, max_iter=10000, tol=1e-4):
        self.C = C
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.selector = SelectFromModel(
            LogisticRegression(
                penalty='elasticnet',
                solver='saga',
                l1_ratio=self.l1_ratio,
                C=self.C,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=1234,
                class_weight='balanced',
                n_jobs=-1
            )
        )

    def fit(self, X, y):
        self.selector.fit(X, y)
        return self

    def transform(self, X):
        return self.selector.transform(X)

# Safer UMAP builder
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

def gaussian_nb_nested_cv(inp, prefix, feature_selection_method):
    # Read data
    data = pd.read_csv(inp)

    # Ensure required columns
    if 'SampleID' not in data.columns or 'Label' not in data.columns:
        raise ValueError("Input data must contain 'SampleID' and 'Label' columns.")

    sample_ids = data['SampleID']
    X = data.drop(columns=['SampleID', 'Label']).copy()
    y = data['Label']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_binarized = label_binarize(y_encoded, classes=np.unique(y_encoded))
    num_classes = len(np.unique(y_encoded))

    tsne_selected = (feature_selection_method == 'tsne')

    # If t-SNE is selected, tune it on the full dataset and transform once (with leakage warning)
    if tsne_selected:
        print("Warning: t-SNE does not support transforming new data. The entire dataset will be transformed before cross-validation, which may lead to data leakage.")
        print("Applying t-SNE transformation to the entire dataset before cross-validation...")

        def tsne_objective(trial):
            n_components = trial.suggest_int('tsne_n_components', 2, 3)
            perplexity = trial.suggest_int('tsne_perplexity', 5, min(50, X.shape[0]-1))
            learning_rate = trial.suggest_loguniform('tsne_learning_rate', 10, 1000)
            max_iter = trial.suggest_int('tsne_max_iter', 250, 2000)
            tsne = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                learning_rate=learning_rate,
                max_iter=max_iter,
                random_state=1234
            )
            with SuppressOutput():
                X_tsne = tsne.fit_transform(X)
                variance = np.var(X_tsne)
            return variance

        study_tsne = optuna.create_study(direction='maximize', sampler=TPESampler(seed=1234))
        study_tsne.optimize(tsne_objective, n_trials=20, show_progress_bar=False)
        best_tsne_params = study_tsne.best_params
        print(f"Best t-SNE parameters: {best_tsne_params}")

        tsne_final = TSNE(
            n_components=best_tsne_params['tsne_n_components'],
            perplexity=best_tsne_params['tsne_perplexity'],
            learning_rate=best_tsne_params['tsne_learning_rate'],
            max_iter=best_tsne_params['tsne_max_iter'],
            random_state=1234
        )
        with SuppressOutput():
            X_tsne_final = tsne_final.fit_transform(X)
        X_transformed = pd.DataFrame(
            X_tsne_final,
            columns=[f"TSNE_Component_{i+1}" for i in range(X_tsne_final.shape[1])]
        )
        X_transformed_final = X_transformed.reset_index(drop=True)

    # Outer CV
    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
    outer_f1_scores = []
    outer_auc_scores = []

    fold_idx = 1
    for train_idx, test_idx in cv_outer.split(X, y_encoded):
        print(f"Processing outer fold {fold_idx}...")
        if tsne_selected:
            X_train_outer, X_test_outer = X_transformed_final.iloc[train_idx], X_transformed_final.iloc[test_idx]
        else:
            X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
        y_train_outer, y_test_outer = y_encoded[train_idx], y_encoded[test_idx]

        cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

        if not tsne_selected:
            def objective_inner(trial):
                steps = [('scaler', StandardScaler())]

                # Feature selection options
                if feature_selection_method == 'elasticnet':
                    lr_C = trial.suggest_loguniform('lr_C', 1e-3, 1e3)
                    lr_l1_ratio = trial.suggest_uniform('lr_l1_ratio', 0.0, 1.0)
                    steps.append(('feature_selection', ElasticNetFeatureSelector(
                        C=lr_C,
                        l1_ratio=lr_l1_ratio,
                        max_iter=10000,
                        tol=1e-4
                    )))
                elif feature_selection_method == 'pca':
                    pca_n_components = trial.suggest_int('pca_n_components', 1, min(X_train_outer.shape[1], X_train_outer.shape[0]-1))
                    steps.append(('feature_selection', PCA(n_components=pca_n_components, random_state=1234)))
                elif feature_selection_method == 'kpca':
                    kpca_kernel = trial.suggest_categorical('kpca_kernel', ['linear', 'poly', 'rbf', 'sigmoid', 'cosine'])
                    kpca_n_components = trial.suggest_int('kpca_n_components', 1, min(X_train_outer.shape[1], X_train_outer.shape[0]-1))
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
                    umap_n_components = trial.suggest_int('umap_n_components', 2, min(100, X_train_outer.shape[1]))
                    umap_n_neighbors = trial.suggest_int('umap_n_neighbors', 5, min(50, X_train_outer.shape[0]-1))
                    umap_min_dist = trial.suggest_uniform('umap_min_dist', 0.0, 0.99)
                    steps.append(('feature_selection', safe_umap(
                        n_components=umap_n_components,
                        n_neighbors=umap_n_neighbors,
                        min_dist=umap_min_dist,
                        X=X_train_outer
                    )))
                elif feature_selection_method == 'pls':
                    pls_max_components = min(X_train_outer.shape[0] - 1, X_train_outer.shape[1])
                    pls_n_components = trial.suggest_int('pls_n_components', 1, max(1, pls_max_components))
                    steps.append(('feature_selection', PLSFeatureSelector(
                        n_components=pls_n_components,
                        max_iter=1000,
                        tol=1e-06
                    )))
                else:
                    pass

                steps.append(('gnb', GaussianNB()))
                pipeline = Pipeline(steps)

                with SuppressOutput():
                    f1_scores = []
                    for inner_train_idx, inner_valid_idx in cv_inner.split(X_train_outer, y_train_outer):
                        X_train_inner = X_train_outer.iloc[inner_train_idx]
                        X_valid_inner = X_train_outer.iloc[inner_valid_idx]
                        y_train_inner = y_train_outer[inner_train_idx]
                        y_valid_inner = y_train_outer[inner_train_idx.shape[0]:][:0]  # placeholder to avoid linter

                        y_valid_inner = y_train_outer[inner_valid_idx]
                        classes = np.unique(y_train_inner)
                        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_inner)
                        cw_dict = dict(zip(classes, class_weights))
                        sample_weight_inner = np.array([cw_dict[label] for label in y_train_inner])

                        try:
                            pipeline.fit(X_train_inner, y_train_inner, gnb__sample_weight=sample_weight_inner)
                            y_pred_inner = pipeline.predict(X_valid_inner)
                            f1 = f1_score(y_valid_inner, y_pred_inner, average='weighted')
                            f1_scores.append(f1)
                        except (ValueError, ArpackError):
                            f1_scores.append(0.0)
                    return np.mean(f1_scores)

            study_inner = optuna.create_study(direction='maximize', sampler=TPESampler(seed=1234))
            study_inner.optimize(objective_inner, n_trials=50, show_progress_bar=False)
            best_params_inner = study_inner.best_params

            steps = [('scaler', StandardScaler())]
            if feature_selection_method == 'elasticnet':
                best_lr_C = best_params_inner.get('lr_C', 1.0)
                best_lr_l1_ratio = best_params_inner.get('lr_l1_ratio', 0.5)
                steps.append(('feature_selection', ElasticNetFeatureSelector(
                    C=best_lr_C,
                    l1_ratio=best_lr_l1_ratio,
                    max_iter=10000,
                    tol=1e-4
                )))
            elif feature_selection_method == 'pca':
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
                steps.append(('feature_selection', safe_umap(
                    n_components=best_umap_n_components,
                    n_neighbors=best_umap_n_neighbors,
                    min_dist=best_umap_min_dist,
                    X=X_train_outer
                )))
            elif feature_selection_method == 'pls':
                best_pls_n_components = min(best_params_inner.get('pls_n_components', 2), X_train_outer.shape[0] - 1, X_train_outer.shape[1])
                steps.append(('feature_selection', PLSFeatureSelector(
                    n_components=best_pls_n_components,
                    max_iter=1000,
                    tol=1e-06
                )))
            else:
                pass

            steps.append(('gnb', GaussianNB()))
            best_model_inner = Pipeline(steps)

            with SuppressOutput():
                try:
                    classes = np.unique(y_train_outer)
                    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_outer)
                    cw_dict = dict(zip(classes, class_weights))
                    sample_weight_outer = np.array([cw_dict[label] for label in y_train_outer])

                    best_model_inner.fit(X_train_outer, y_train_outer, gnb__sample_weight=sample_weight_outer)
                except (ValueError, ArpackError) as e:
                    print(f"Error fitting the model in outer fold {fold_idx}: {e}")
                    outer_f1_scores.append(0)
                    outer_auc_scores.append(0)
                    fold_idx += 1
                    continue
        else:
            best_model_inner = Pipeline([('gnb', GaussianNB())])
            with SuppressOutput():
                try:
                    classes = np.unique(y_train_outer)
                    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_outer)
                    cw_dict = dict(zip(classes, class_weights))
                    sample_weight_outer = np.array([cw_dict[label] for label in y_train_outer])

                    best_model_inner.fit(X_train_outer, y_train_outer, gnb__sample_weight=sample_weight_outer)
                except (ValueError, ArpackError) as e:
                    print(f"Error fitting the model in outer fold {fold_idx}: {e}")
                    outer_f1_scores.append(0)
                    outer_auc_scores.append(0)
                    fold_idx += 1
                    continue

        # Predict on outer test set
        try:
            y_pred_prob_outer = best_model_inner.predict_proba(X_test_outer)
            y_pred_class_outer = best_model_inner.predict(X_test_outer)
        except (ValueError, ArpackError) as e:
            print(f"Error predicting in outer fold {fold_idx}: {e}")
            outer_f1_scores.append(0)
            outer_auc_scores.append(0)
            fold_idx += 1
            continue

        f1_outer = f1_score(y_test_outer, y_pred_class_outer, average='weighted')
        outer_f1_scores.append(f1_outer)

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

    # Plot per-fold F1 / AUC
    plt.figure(figsize=(10, 6))
    folds = range(1, cv_outer.get_n_splits() + 1)
    plt.plot(folds, outer_f1_scores, marker='o', linestyle='-', label='F1 Score')
    plt.plot(folds, outer_auc_scores, marker='s', linestyle='-', label='AUC Score')
    plt.title('F1 and AUC Scores per Outer Fold', fontsize=26, fontweight='bold', pad=14)
    plt.xlabel('Outer Fold Number', fontsize=22, labelpad=12)
    plt.ylabel('Score (F1 / AUC)', fontsize=22, labelpad=12)
    plt.xticks(folds, fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18, title_fontsize=20)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    ax_cv = plt.gca()
    enlarge_fonts(ax_cv)
    plt.tight_layout()
    plt.savefig(f"{prefix}_gaussiannb_nested_cv_f1_auc.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Nested cross-validation completed.")
    print(f"Average F1 Score: {np.mean(outer_f1_scores):.4f} ± {np.std(outer_f1_scores):.4f}")
    print(f"Average AUC: {np.mean(outer_auc_scores):.4f} ± {np.std(outer_auc_scores):.4f}")

    # Hyperparameter tuning on the entire dataset
    print("Starting hyperparameter tuning on the entire dataset...")

    if tsne_selected:
        def objective_full_tsne(trial):
            steps = [('scaler', StandardScaler()), ('gnb', GaussianNB())]
            model = Pipeline(steps)
            with SuppressOutput():
                f1_scores_tsne = []
                for train_idx_full, valid_idx_full in cv_outer.split(X_transformed_final, y_encoded):
                    X_train_full, X_valid_full = X_transformed_final.iloc[train_idx_full], X_transformed_final.iloc[valid_idx_full]
                    y_train_full, y_valid_full = y_encoded[train_idx_full], y_encoded[valid_idx_full]
                    try:
                        model.fit(X_train_full, y_train_full)
                        y_pred_full = model.predict(X_valid_full)
                        f1 = f1_score(y_valid_full, y_pred_full, average='weighted')
                        f1_scores_tsne.append(f1)
                    except (ValueError, ArpackError):
                        f1_scores_tsne.append(0.0)
                return np.mean(f1_scores_tsne)

        study_full_tsne = optuna.create_study(direction='maximize', sampler=TPESampler(seed=1234))
        study_full_tsne.optimize(objective_full_tsne, n_trials=50, show_progress_bar=True)
        best_params_full_tsne = study_full_tsne.best_params
        print(f"Best parameters for GaussianNB with t-SNE: {best_params_full_tsne}")

        # Final model on full t-SNE features
        best_model = Pipeline([('scaler', StandardScaler()), ('gnb', GaussianNB())])
        with SuppressOutput():
            try:
                classes_full = np.unique(y_encoded)
                class_weights_full = compute_class_weight(class_weight='balanced', classes=classes_full, y=y_encoded)
                cw_dict_full = dict(zip(classes_full, class_weights_full))
                sample_weight_full = np.array([cw_dict_full[label] for label in y_encoded])
                best_model.fit(X_transformed_final, y_encoded, gnb__sample_weight=sample_weight_full)
            except (ValueError, ArpackError) as e:
                print(f"Error fitting the final model with t-SNE: {e}")
                sys.exit(1)

        joblib.dump(best_model, f"{prefix}_gaussiannb_model.pkl")
        joblib.dump((X_transformed_final, y_encoded, le), f"{prefix}_gaussiannb_data.pkl")

        # Save transformed data CSV for t-SNE
        X_transformed_df = pd.DataFrame(
            X_transformed_final,
            columns=[f"TSNE_Component_{i+1}" for i in range(X_transformed_final.shape[1])]
        )
        X_transformed_df.insert(0, 'SampleID', sample_ids)
        X_transformed_df['Label'] = y
        transformed_csv_path = f"{prefix}_gaussiannb_transformed_X_tsne.csv"
        X_transformed_df.to_csv(transformed_csv_path, index=False)
        print(f"t-SNE transformed data saved to {transformed_csv_path}")

        variance_csv_path = f"{prefix}_gaussiannb_variance.csv"
        with open(variance_csv_path, 'w') as f:
            f.write("t-SNE does not provide explained variance information.\n")
        print(f"No variance information available for t-SNE. File created at {variance_csv_path}")

        # Predictions on full t-SNE features
        try:
            y_pred_prob = best_model.predict_proba(X_transformed_final)
            y_pred_class = best_model.predict(X_transformed_final)
        except (ValueError, ArpackError) as e:
            print(f"Error during prediction with t-SNE transformed data: {e}")
            y_pred_class = np.zeros_like(y_encoded)
            y_pred_prob = np.zeros((len(y_encoded), num_classes))

        acc = accuracy_score(y_encoded, y_pred_class)
        f1 = f1_score(y_encoded, y_pred_class, average='weighted')
        cm = confusion_matrix(y_encoded, y_pred_class)

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
                sensitivity = np.mean(
                    np.divide(
                        np.diag(cm),
                        np.sum(cm, axis=1),
                        out=np.zeros_like(np.diag(cm), dtype=float),
                        where=np.sum(cm, axis=1)!=0
                    )
                )
            specificity = multiclass_specificity(cm)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix for GaussianNB with t-SNE', fontsize=16, fontweight='bold', pad=12)
        enlarge_fonts(disp.ax_)
        plt.savefig(f"{prefix}_gaussiannb_confusion_matrix.png", dpi=300, bbox_inches="tight")
        plt.close()

        fpr = {}
        tpr = {}
        roc_auc = {}

        if num_classes == 2:
            try:
                fpr[0], tpr[0], _ = roc_curve(y_encoded, y_pred_prob[:, 1])
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

            try:
                fpr["micro"], tpr["micro"], _ = roc_curve(y_binarized.ravel(), y_pred_prob.ravel())
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            except ValueError:
                fpr["micro"], tpr["micro"], roc_auc["micro"] = np.array([0, 1]), np.array([0, 1]), 0.0

            try:
                class_keys = [k for k in fpr.keys() if isinstance(k, (int, np.integer))]
                valid_keys = [k for k in class_keys if len(fpr[k]) > 1 and len(tpr[k]) > 1]
                if len(valid_keys) > 0:
                    all_fpr = np.unique(np.concatenate([fpr[k] for k in valid_keys]))
                    mean_tpr = np.zeros_like(all_fpr)
                    for k in valid_keys:
                        mean_tpr += np.interp(all_fpr, fpr[k], tpr[k])
                    mean_tpr /= len(valid_keys)
                    fpr["macro"] = all_fpr
                    tpr["macro"] = mean_tpr
                    roc_auc["macro"] = auc(all_fpr, mean_tpr)
                else:
                    fpr["macro"], tpr["macro"], roc_auc["macro"] = np.array([0, 1]), np.array([0, 1]), 0.0
            except Exception:
                fpr["macro"], tpr["macro"], roc_auc["macro"] = np.array([0, 1]), np.array([0, 1]), 0.0

        roc_data = {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}
        np.save(f"{prefix}_gaussiannb_roc_data.npy", roc_data, allow_pickle=True)

        plt.figure(figsize=(10, 8))
        if num_classes == 2:
            plt.plot(fpr[0], tpr[0], label=f'AUC = {roc_auc[0]:.2f}')
        else:
            for i in range(len(le.classes_)):
                if i in roc_auc and roc_auc[i] > 0.0:
                    plt.plot(fpr[i], tpr[i], label=f'{le.inverse_transform([i])[0]} (AUC = {roc_auc[i]:.2f})')
            if "micro" in roc_auc and roc_auc["micro"] > 0.0:
                plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})', linestyle='--')
            if "macro" in roc_auc and roc_auc["macro"] > 0.0:
                plt.plot(fpr["macro"], tpr["macro"], label=f'Macro-average (AUC = {roc_auc["macro"]:.2f})', linestyle=':')

        plt.plot([0, 1], [0, 1], 'k--', lw=1.2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=22, labelpad=12)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=22, labelpad=12)
        plt.title('ROC Curves for GaussianNB with t-SNE', fontsize=26, fontweight='bold', pad=14)
        plt.legend(loc="lower right", fontsize=18, title_fontsize=20)
        ax_roc_tsne = plt.gca()
        enlarge_fonts(ax_roc_tsne)
        plt.tight_layout()
        plt.savefig(f'{prefix}_gaussiannb_roc_curve.png', dpi=300, bbox_inches="tight")
        plt.close()

        metrics = {'Accuracy': acc, 'F1 Score': f1, 'Sensitivity': sensitivity, 'Specificity': specificity}
        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        ax = metrics_df.plot(kind='bar', x='Metric', y='Value', legend=False)
        plt.title('Performance Metrics for GaussianNB with t-SNE', fontsize=20, fontweight='bold', pad=10)
        plt.ylabel('Value', fontsize=22, labelpad=12)
        plt.ylim(0, 1.1)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=5)
        plt.xticks(rotation=45, ha='right', fontsize=18)
        ax.tick_params(axis='y', labelsize=18)
        enlarge_fonts(ax)
        plt.tight_layout()
        plt.savefig(f'{prefix}_gaussiannb_metrics.png', dpi=300, bbox_inches="tight")
        plt.close()

        predictions_df = pd.DataFrame({
            'SampleID': sample_ids,
            'Original Label': y,
            'Predicted Label': le.inverse_transform(y_pred_class)
        })
        predictions_df.to_csv(f"{prefix}_gaussiannb_predictions.csv", index=False)
        print(f"Predictions saved to {prefix}_gaussiannb_predictions.csv")

    else:
        def objective_full(trial):
            steps = [('scaler', StandardScaler())]

            if feature_selection_method == 'elasticnet':
                lr_C = trial.suggest_loguniform('lr_C', 1e-3, 1e3)
                lr_l1_ratio = trial.suggest_uniform('lr_l1_ratio', 0.0, 1.0)
                steps.append(('feature_selection', ElasticNetFeatureSelector(
                    C=lr_C,
                    l1_ratio=lr_l1_ratio,
                    max_iter=10000,
                    tol=1e-4
                )))
            elif feature_selection_method == 'pca':
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
                steps.append(('feature_selection', safe_umap(
                    n_components=umap_n_components,
                    n_neighbors=umap_n_neighbors,
                    min_dist=umap_min_dist,
                    X=X
                )))
            elif feature_selection_method == 'pls':
                pls_max_components = min(X.shape[0] - 1, X.shape[1])
                pls_n_components = trial.suggest_int('pls_n_components', 1, max(1, pls_max_components))
                steps.append(('feature_selection', PLSFeatureSelector(
                    n_components=pls_n_components,
                    max_iter=1000,
                    tol=1e-06
                )))
            else:
                pass

            steps.append(('gnb', GaussianNB()))
            pipeline = Pipeline(steps)

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
                        f1_scores.append(0.0)
                return np.mean(f1_scores)

        study_full = optuna.create_study(direction='maximize', sampler=TPESampler(seed=1234))
        study_full.optimize(objective_full, n_trials=50, show_progress_bar=True)

        best_params_full = study_full.best_params
        print(f"Best parameters for GaussianNB: {best_params_full}")

        steps = [('scaler', StandardScaler())]
        if feature_selection_method == 'elasticnet':
            best_lr_C_full = best_params_full.get('lr_C', 1.0)
            best_lr_l1_ratio_full = best_params_full.get('lr_l1_ratio', 0.5)
            steps.append(('feature_selection', ElasticNetFeatureSelector(
                C=best_lr_C_full,
                l1_ratio=best_lr_l1_ratio_full,
                max_iter=10000,
                tol=1e-4
            )))
        elif feature_selection_method == 'pca':
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
            steps.append(('feature_selection', safe_umap(
                n_components=best_umap_n_components_full,
                n_neighbors=best_umap_n_neighbors_full,
                min_dist=best_umap_min_dist_full,
                X=X
            )))
        elif feature_selection_method == 'pls':
            best_pls_n_components_full = min(best_params_full.get('pls_n_components', 2), X.shape[0] - 1, X.shape[1])
            steps.append(('feature_selection', PLSFeatureSelector(
                n_components=best_pls_n_components_full,
                max_iter=1000,
                tol=1e-06
            )))
        else:
            pass

        steps.append(('gnb', GaussianNB()))
        best_model = Pipeline(steps)

        with SuppressOutput():
            try:
                classes_full = np.unique(y_encoded)
                class_weights_full = compute_class_weight(class_weight='balanced', classes=classes_full, y=y_encoded)
                cw_dict_full = dict(zip(classes_full, class_weights_full))
                sample_weight_full = np.array([cw_dict_full[label] for label in y_encoded])
                best_model.fit(X, y_encoded, gnb__sample_weight=sample_weight_full)
            except (ValueError, ArpackError) as e:
                print(f"Error fitting the final model: {e}")
                sys.exit(1)

        joblib.dump(best_model, f"{prefix}_gaussiannb_model.pkl")
        joblib.dump((X, y_encoded, le), f"{prefix}_gaussiannb_data.pkl")

        # Save transformed data and variance info if feature selection used
        if feature_selection_method != 'none':
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
                    selected_features = X.columns[best_model.named_steps['feature_selection'].selector.get_support()]
                    X_transformed_df = X[selected_features].copy()
                else:
                    X_transformed_df = pd.DataFrame(X_transformed)

                X_transformed_df.insert(0, 'SampleID', sample_ids)
                X_transformed_df['Label'] = y
                transformed_csv_path = f"{prefix}_gaussiannb_transformed_X.csv"
                X_transformed_df.to_csv(transformed_csv_path, index=False)
                print(f"Transformed data saved to {transformed_csv_path}")

                variance_csv_path = f"{prefix}_gaussiannb_variance.csv"
                if feature_selection_method == 'pca':
                    n_components_full_pca = min(X.shape[0], X.shape[1])
                    full_pca = PCA(n_components=n_components_full_pca, random_state=1234)
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
                    y_full = label_binarize(y_encoded, classes=np.unique(y_encoded))
                    if y_full.ndim == 1:
                        y_full = np.vstack([1 - y_full, y_full]).T
                    pls_n_components_full = max(1, min(X.shape[0] - 1, X.shape[1]))
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
                print("No valid transformed data available. Skipping transformed data and variance information saving.")

        # cross_val_predict for final metrics on original features
        try:
            y_pred_prob = cross_val_predict(best_model, X, y_encoded, cv=cv_outer, method='predict_proba', n_jobs=-1)
            y_pred_class = np.argmax(y_pred_prob, axis=1)
        except (ValueError, ArpackError) as e:
            print(f"Error during cross_val_predict: {e}")
            y_pred_class = np.zeros_like(y_encoded)
            y_pred_prob = np.zeros((len(y_encoded), num_classes))

        acc = accuracy_score(y_encoded, y_pred_class)
        f1 = f1_score(y_encoded, y_pred_class, average='weighted')
        cm = confusion_matrix(y_encoded, y_pred_class)

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
                sensitivity = np.mean(
                    np.divide(
                        np.diag(cm),
                        np.sum(cm, axis=1),
                        out=np.zeros_like(np.diag(cm), dtype=float),
                        where=np.sum(cm, axis=1)!=0
                    )
                )
            specificity = multiclass_specificity(cm)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix for GaussianNB', fontsize=16, fontweight='bold', pad=12)
        enlarge_fonts(disp.ax_)
        plt.savefig(f"{prefix}_gaussiannb_confusion_matrix.png", dpi=300, bbox_inches="tight")
        plt.close()

        fpr = {}
        tpr = {}
        roc_auc = {}

        if num_classes == 2:
            try:
                fpr[0], tpr[0], _ = roc_curve(y_encoded, y_pred_prob[:, 1])
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

            try:
                fpr["micro"], tpr["micro"], _ = roc_curve(y_binarized.ravel(), y_pred_prob.ravel())
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            except ValueError:
                fpr["micro"], tpr["micro"], roc_auc["micro"] = np.array([0, 1]), np.array([0, 1]), 0.0

            try:
                class_keys = [k for k in fpr.keys() if isinstance(k, (int, np.integer))]
                valid_keys = [k for k in class_keys if len(fpr[k]) > 1 and len(tpr[k]) > 1]
                if len(valid_keys) > 0:
                    all_fpr = np.unique(np.concatenate([fpr[k] for k in valid_keys]))
                    mean_tpr = np.zeros_like(all_fpr)
                    for k in valid_keys:
                        mean_tpr += np.interp(all_fpr, fpr[k], tpr[k])
                    mean_tpr /= len(valid_keys)
                    fpr["macro"] = all_fpr
                    tpr["macro"] = mean_tpr
                    roc_auc["macro"] = auc(all_fpr, mean_tpr)
                else:
                    fpr["macro"], tpr["macro"], roc_auc["macro"] = np.array([0, 1]), np.array([0, 1]), 0.0
            except Exception:
                fpr["macro"], tpr["macro"], roc_auc["macro"] = np.array([0, 1]), np.array([0, 1]), 0.0

        roc_data = {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}
        np.save(f"{prefix}_gaussiannb_roc_data.npy", roc_data, allow_pickle=True)

        plt.figure(figsize=(10, 8))
        if num_classes == 2:
            plt.plot(fpr[0], tpr[0], label=f'AUC = {roc_auc[0]:.2f}')
        else:
            for i in range(len(le.classes_)):
                if i in roc_auc and roc_auc[i] > 0.0:
                    plt.plot(fpr[i], tpr[i], label=f'{le.inverse_transform([i])[0]} (AUC = {roc_auc[i]:.2f})')
            if "micro" in roc_auc and roc_auc["micro"] > 0.0:
                plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})', linestyle='--')
            if "macro" in roc_auc and roc_auc["macro"] > 0.0:
                plt.plot(fpr["macro"], tpr["macro"], label=f'Macro-average (AUC = {roc_auc["macro"]:.2f})', linestyle=':')

        plt.plot([0, 1], [0, 1], 'k--', lw=1.2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=22, labelpad=12)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=22, labelpad=12)
        plt.title('ROC Curves for GaussianNB', fontsize=26, fontweight='bold', pad=14)
        plt.legend(loc="lower right", fontsize=18, title_fontsize=20)
        ax_roc = plt.gca()
        enlarge_fonts(ax_roc)
        plt.tight_layout()
        plt.savefig(f'{prefix}_gaussiannb_roc_curve.png', dpi=300, bbox_inches="tight")
        plt.close()

        metrics = {'Accuracy': acc, 'F1 Score': f1, 'Sensitivity': sensitivity, 'Specificity': specificity}
        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        ax = metrics_df.plot(kind='bar', x='Metric', y='Value', legend=False)
        plt.title('Performance Metrics for GaussianNB', fontsize=20, fontweight='bold', pad=10)
        plt.ylabel('Value', fontsize=22, labelpad=12)
        plt.ylim(0, 1.1)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=5)
        plt.xticks(rotation=45, ha='right', fontsize=18)
        ax.tick_params(axis='y', labelsize=18)
        enlarge_fonts(ax)
        plt.tight_layout()
        plt.savefig(f'{prefix}_gaussiannb_metrics.png', dpi=300, bbox_inches="tight")
        plt.close()

        predictions_df = pd.DataFrame({
            'SampleID': sample_ids,
            'Original Label': y,
            'Predicted Label': le.inverse_transform(y_pred_class)
        })
        predictions_df.to_csv(f"{prefix}_gaussiannb_predictions.csv", index=False)
        print(f"Predictions saved to {prefix}_gaussiannb_predictions.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run GaussianNB with Nested Cross-Validation, Feature Selection (PCA, KPCA, UMAP, TSNE, PLS, ElasticNet), and Optuna hyperparameter optimization.'
    )
    parser.add_argument('-i', type=str, required=True, help='Path to the input CSV file.')
    parser.add_argument('-p', type=str, required=True, help='Prefix for output files.')
    parser.add_argument('-f', type=str,
                        choices=['none', 'elasticnet', 'pca', 'kpca', 'umap', 'tsne', 'pls'],
                        default='none',
                        help='Feature selection method to use. Options: none, elasticnet, pca, kpca, umap, tsne, pls.')
    args = parser.parse_args()

    if args.f == 'tsne':
        print("Warning: t-SNE does not support transforming new data. The entire dataset will be transformed before cross-validation, which may lead to data leakage.")

    gaussian_nb_nested_cv(args.i, args.p, args.f)
