import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
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
    ConfusionMatrixDisplay
)
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import warnings
import sys
import contextlib
import joblib
import optuna
from optuna.samplers import TPESampler
import umap
from scipy.sparse.linalg import ArpackError

# warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)


# ----------------------------------
# Utility: suppress stdout/stderr
# ----------------------------------
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


# ------------------------------------------------------
# Feature selector: PLS with proper y handling (one-hot)
# ------------------------------------------------------
class PLSFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, max_iter=1000, tol=1e-06):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.pls = None
        self.classes_ = None
        self.y_onehot_dim_ = None

    def _prepare_y(self, y):
        classes = np.unique(y)
        self.classes_ = classes
        y_onehot = label_binarize(y, classes=classes)
        if y_onehot.ndim == 1:
            y_onehot = np.vstack([1 - y_onehot, y_onehot]).T
        self.y_onehot_dim_ = y_onehot.shape[1]
        return y_onehot

    def fit(self, X, y):
        y_onehot = self._prepare_y(y)
        max_allowed = max(1, min(X.shape[0] - 1, X.shape[1], self.y_onehot_dim_))
        n_comp = min(self.n_components, max_allowed)
        self.n_components = n_comp
        self.pls = PLSRegression(n_components=n_comp, max_iter=self.max_iter, tol=self.tol)
        self.pls.fit(X, y_onehot)
        return self

    def transform(self, X):
        return self.pls.transform(X)


# ----------------------------------
# Safe UMAP constructor
# ----------------------------------
def safe_umap(n_components, n_neighbors, min_dist, X, random_state=1234):
    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_components = min(max(2, n_components), max(2, min(n_features, n_samples - 1)))
    n_neighbors = min(max(2, n_neighbors), max(2, n_samples - 2))
    return umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        init='random'
    )


# ----------------------------------
# Metrics helpers
# ----------------------------------
def multiclass_specificity(cm):
    # macro-average specificity: TN/(TN+FP) per class in one-vs-rest
    K = cm.shape[0]
    total = np.sum(cm)
    specs = []
    for i in range(K):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        TN = total - TP - FN - FP
        denom = TN + FP
        spec_i = TN / denom if denom > 0 else 0.0
        specs.append(spec_i)
    return float(np.mean(specs))


def compute_roc_multi(y_true, y_prob, classes):
    # per-class, micro-average, macro-average ROC
    y_bin = label_binarize(y_true, classes=classes)
    if y_bin.ndim == 1:
        y_bin = np.vstack([1 - y_bin, y_bin]).T

    fpr = {}
    tpr = {}
    roc_auc = {}

    # per-class
    for i in range(y_bin.shape[1]):
        if np.sum(y_bin[:, i]) == 0:
            fpr[i], tpr[i] = np.array([0, 1]), np.array([0, 1])
            roc_auc[i] = 0.0
        else:
            try:
                fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_prob[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            except ValueError:
                fpr[i], tpr[i], roc_auc[i] = np.array([0, 1]), np.array([0, 1]), 0.0

    # micro
    try:
        fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_prob.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    except ValueError:
        fpr["micro"], tpr["micro"], roc_auc["micro"] = np.array([0, 1]), np.array([0, 1]), 0.0

    # macro (interpolate to a common grid)
    valid = [i for i in range(y_bin.shape[1]) if i in fpr and fpr[i].ndim > 0 and tpr[i].ndim > 0]
    if len(valid) > 0:
        all_fpr = np.unique(np.concatenate([fpr[i] for i in valid]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in valid:
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= len(valid)
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    else:
        fpr["macro"], tpr["macro"], roc_auc["macro"] = np.array([0, 1]), np.array([0, 1]), 0.0

    return fpr, tpr, roc_auc


# ----------------------------------
# Main
# ----------------------------------
def svm_nested_cv(inp, prefix, feature_selection_method):
    # read data
    data = pd.read_csv(inp)

    if 'SampleID' not in data.columns or 'Label' not in data.columns:
        raise ValueError("Input data must contain 'SampleID' and 'Label' columns.")

    sample_ids = data['SampleID']
    X = data.drop(columns=['SampleID', 'Label']).copy()
    y = data['Label']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    classes_enc = np.unique(y_encoded)
    num_classes = len(classes_enc)

    # pre-export all components when requested
    if feature_selection_method == 'pca':
        max_pca_components = min(X.shape[0], X.shape[1])
        full_pca = PCA(n_components=max_pca_components, random_state=1234)
        with SuppressOutput():
            X_pca_full = full_pca.fit_transform(X)
        explained_variance = full_pca.explained_variance_ratio_
        explained_variance_df = pd.DataFrame({
            'Component': range(1, len(explained_variance) + 1),
            'Explained Variance Ratio': explained_variance
        })
        explained_variance_df.to_csv(f"{prefix}_svm_pca_explained_variance_full.csv", index=False)
        X_pca_full_df = pd.DataFrame(X_pca_full, columns=[f"PCA_Component_{i+1}" for i in range(X_pca_full.shape[1])])
        X_pca_full_df.insert(0, 'SampleID', sample_ids)
        X_pca_full_df['Label'] = y
        X_pca_full_df.to_csv(f"{prefix}_svm_pca_all_components.csv", index=False)

    elif feature_selection_method == 'pls':
        pls_n_full = max(1, min(X.shape[0] - 1, X.shape[1], num_classes))
        pls = PLSRegression(n_components=pls_n_full, max_iter=1000, tol=1e-06)
        with SuppressOutput():
            y_onehot_full = label_binarize(y_encoded, classes=classes_enc)
            if y_onehot_full.ndim == 1:
                y_onehot_full = np.vstack([1 - y_onehot_full, y_onehot_full]).T
            X_pls_full = pls.fit_transform(X, y_onehot_full)[0]
        explained_variance = np.var(X_pls_full, axis=0) / np.var(X, axis=0).sum()
        explained_variance_df = pd.DataFrame({
            'Component': range(1, len(explained_variance) + 1),
            'Explained Variance Ratio': explained_variance
        })
        explained_variance_df.to_csv(f"{prefix}_svm_pls_explained_variance_full.csv", index=False)
        X_pls_full_df = pd.DataFrame(X_pls_full, columns=[f"PLS_Component_{i+1}" for i in range(X_pls_full.shape[1])])
        X_pls_full_df.insert(0, 'SampleID', sample_ids)
        X_pls_full_df['Label'] = y
        X_pls_full_df.to_csv(f"{prefix}_svm_pls_all_components.csv", index=False)

    elif feature_selection_method == 'kpca':
        n_samples = X.shape[0]
        max_kpca_components = max(1, min(X.shape[1], n_samples - 1))
        kpca = KernelPCA(
            n_components=max_kpca_components,
            kernel='rbf',
            gamma=1.0,
            random_state=1234,
            eigen_solver='arpack',
            max_iter=5000
        )
        with SuppressOutput():
            try:
                X_kpca_full = kpca.fit_transform(X)
            except ArpackError as e:
                print(f"KernelPCA fitting failed: {e}")
                X_kpca_full = np.zeros((X.shape[0], max_kpca_components))
        X_kpca_full_df = pd.DataFrame(X_kpca_full, columns=[f"KPCA_Component_{i+1}" for i in range(X_kpca_full.shape[1])])
        X_kpca_full_df.insert(0, 'SampleID', sample_ids)
        X_kpca_full_df['Label'] = y
        X_kpca_full_df.to_csv(f"{prefix}_svm_kpca_all_components.csv", index=False)

    elif feature_selection_method == 'umap':
        n_samples = X.shape[0]
        n_features = X.shape[1]
        max_umap_components = max(2, min(n_features, n_samples - 1))
        safe_neighbors = min(15, max(2, n_samples - 2))
        umap_full = umap.UMAP(
            n_components=max_umap_components,
            n_neighbors=safe_neighbors,
            min_dist=0.1,
            random_state=1234,
            init='random'
        )
        with SuppressOutput():
            X_umap_full = umap_full.fit_transform(X)
        X_umap_full_df = pd.DataFrame(X_umap_full, columns=[f"UMAP_Component_{i+1}" for i in range(X_umap_full.shape[1])])
        X_umap_full_df.insert(0, 'SampleID', sample_ids)
        X_umap_full_df['Label'] = y
        X_umap_full_df.to_csv(f"{prefix}_svm_umap_all_components.csv", index=False)

    elif feature_selection_method == 'tsne':
        print("Warning: t-SNE is precomputed on the full dataset. This may cause information leakage.")
        tsne_full = TSNE(
            n_components=2,
            perplexity=min(30, max(5, X.shape[0] - 1)),
            learning_rate=200,
            n_iter=1000,
            random_state=1234
        )
        with SuppressOutput():
            X_tsne_full = tsne_full.fit_transform(X)
        X_tsne_full_df = pd.DataFrame(X_tsne_full, columns=[f"TSNE_Component_{i+1}" for i in range(X_tsne_full.shape[1])])
        X_tsne_full_df.insert(0, 'SampleID', sample_ids)
        X_tsne_full_df['Label'] = y
        X_tsne_full_df.to_csv(f"{prefix}_svm_tsne_all_components.csv", index=False)

    # choose modeling matrix
    if feature_selection_method == 'tsne':
        X_use = pd.DataFrame(X_tsne_full, columns=[f"TSNE_Component_{i+1}" for i in range(X_tsne_full.shape[1])])
    else:
        X_use = X

    # CV setup
    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
    outer_f1_scores = []
    outer_auc_scores = []

    # fonts
    TITLE_FONTSIZE = 22
    LABEL_FONTSIZE = 22
    TICK_FONTSIZE = 18
    LEGEND_FONTSIZE = 18

    # outer loop
    fold_idx = 1
    for train_idx, test_idx in cv_outer.split(X_use, y_encoded):
        print(f"Processing outer fold {fold_idx}...")
        X_train_outer, X_test_outer = X_use.iloc[train_idx], X_use.iloc[test_idx]
        y_train_outer, y_test_outer = y_encoded[train_idx], y_encoded[test_idx]

        cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

        def objective_inner(trial):
            C = trial.suggest_loguniform('C', 1e-2, 1e4)
            gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])

            steps = [('scaler', StandardScaler())]

            # feature selection (exclude t-SNE; already embedded in X_use)
            if feature_selection_method == 'elasticnet':
                l1_ratio = trial.suggest_uniform('l1_ratio', 0.0, 1.0)
                C_lr = trial.suggest_loguniform('C_lr', 1e-2, 1e2)
                steps.append((
                    'feature_selection',
                    SelectFromModel(
                        LogisticRegression(
                            penalty='elasticnet',
                            solver='saga',
                            l1_ratio=l1_ratio,
                            C=C_lr,
                            max_iter=200000,
                            class_weight='balanced',
                            random_state=1234
                        )
                    )
                ))
            elif feature_selection_method == 'pca':
                max_pca_components_inner = max(1, min(X_train_outer.shape[1], X_train_outer.shape[0]))
                n_components = trial.suggest_int('n_components', 1, max_pca_components_inner)
                steps.append(('feature_selection', PCA(n_components=n_components, random_state=1234)))
            elif feature_selection_method == 'kpca':
                n_train = X_train_outer.shape[0]
                max_kpca_components_inner = max(1, min(X_train_outer.shape[1], n_train - 1))
                n_components = trial.suggest_int('n_components', 1, max_kpca_components_inner)
                kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid', 'cosine'])
                kpca_params = {
                    'n_components': n_components,
                    'kernel': kernel,
                    'random_state': 1234,
                    'eigen_solver': 'arpack',
                    'max_iter': 5000
                }
                if kernel in ['poly', 'rbf', 'sigmoid']:
                    kpca_params['gamma'] = trial.suggest_loguniform('kpca_gamma', 1e-4, 1e1)
                if kernel in ['poly', 'sigmoid']:
                    kpca_params['coef0'] = trial.suggest_uniform('kpca_coef0', 0.0, 1.0)
                if kernel == 'poly':
                    kpca_params['degree'] = trial.suggest_int('kpca_degree', 2, 5)
                steps.append(('feature_selection', KernelPCA(**kpca_params)))
            elif feature_selection_method == 'umap':
                n_train = X_train_outer.shape[0]
                max_umap_components = max(2, min(X_train_outer.shape[1], n_train - 1))
                n_components_umap = trial.suggest_int('n_components', 2, max_umap_components)
                n_neighbors_umap = trial.suggest_int('n_neighbors', 5, min(50, n_train - 1))
                n_neighbors_umap = min(n_neighbors_umap, max(2, n_train - 2))
                min_dist_umap = trial.suggest_uniform('min_dist', 0.0, 0.99)
                steps.append(('feature_selection', safe_umap(
                    n_components=n_components_umap,
                    n_neighbors=n_neighbors_umap,
                    min_dist=min_dist_umap,
                    X=X_train_outer
                )))
            elif feature_selection_method == 'pls':
                n_train = X_train_outer.shape[0]
                max_pls_components_inner = max(1, min(n_train - 1, X_train_outer.shape[1], num_classes))
                n_components_pls = trial.suggest_int('n_components', 1, max_pls_components_inner)
                steps.append(('feature_selection', PLSFeatureSelector(n_components=n_components_pls, max_iter=1000, tol=1e-06)))
            elif feature_selection_method == 'tsne':
                pass
            else:
                pass

            steps.append(('svm', SVC(kernel='rbf', C=C, gamma=gamma, probability=True, random_state=1234, class_weight='balanced')))
            pipeline = Pipeline(steps)

            with SuppressOutput():
                f1_scores = []
                for inner_train_idx, inner_valid_idx in cv_inner.split(X_train_outer, y_train_outer):
                    X_train_inner, X_valid_inner = X_train_outer.iloc[inner_train_idx], X_train_outer.iloc[inner_valid_idx]
                    y_train_inner, y_valid_inner = y_train_outer[inner_train_idx], y_train_outer[inner_valid_idx]
                    try:
                        pipeline.fit(X_train_inner, y_train_inner)
                        y_pred_inner = pipeline.predict(X_valid_inner)
                        f1_val = f1_score(y_valid_inner, y_pred_inner, average='weighted')
                        f1_scores.append(f1_val)
                    except (ArpackError, NotImplementedError, ValueError):
                        return 0.0
                return float(np.mean(f1_scores))

        study_inner = optuna.create_study(direction='maximize', sampler=TPESampler(seed=1234))
        study_inner.optimize(objective_inner, n_trials=50, show_progress_bar=False)
        best_params_inner = study_inner.best_params

        steps = [('scaler', StandardScaler())]

        if feature_selection_method == 'elasticnet':
            best_l1_ratio = best_params_inner.get('l1_ratio', 0.5)
            best_C_lr = best_params_inner.get('C_lr', 1.0)
            steps.append((
                'feature_selection',
                SelectFromModel(
                    LogisticRegression(
                        penalty='elasticnet',
                        solver='saga',
                        l1_ratio=best_l1_ratio,
                        C=best_C_lr,
                        max_iter=200000,
                        class_weight='balanced',
                        random_state=1234
                    )
                )
            ))
        elif feature_selection_method == 'pca':
            best_n_components = best_params_inner.get('n_components', max(1, min(X_train_outer.shape[1], X_train_outer.shape[0])))
            steps.append(('feature_selection', PCA(n_components=best_n_components, random_state=1234)))
        elif feature_selection_method == 'kpca':
            best_n_components = best_params_inner.get('n_components', max(1, min(X_train_outer.shape[1], X_train_outer.shape[0] - 1)))
            best_kernel = best_params_inner.get('kernel', 'rbf')
            kpca_params = {
                'n_components': best_n_components,
                'kernel': best_kernel,
                'random_state': 1234,
                'eigen_solver': 'arpack',
                'max_iter': 5000
            }
            if best_kernel in ['poly', 'rbf', 'sigmoid']:
                kpca_params['gamma'] = best_params_inner.get('kpca_gamma', 1.0)
            if best_kernel in ['poly', 'sigmoid']:
                kpca_params['coef0'] = best_params_inner.get('kpca_coef0', 0.0)
            if best_kernel == 'poly':
                kpca_params['degree'] = best_params_inner.get('kpca_degree', 3)
            steps.append(('feature_selection', KernelPCA(**kpca_params)))
        elif feature_selection_method == 'umap':
            best_n_components = best_params_inner.get('n_components', max(2, min(X_train_outer.shape[1], X_train_outer.shape[0] - 1)))
            best_n_neighbors = best_params_inner.get('n_neighbors', min(15, max(2, X_train_outer.shape[0] - 2)))
            best_min_dist = best_params_inner.get('min_dist', 0.1)
            steps.append(('feature_selection', safe_umap(
                n_components=best_n_components,
                n_neighbors=best_n_neighbors,
                min_dist=best_min_dist,
                X=X_train_outer
            )))
        elif feature_selection_method == 'pls':
            best_n_components = best_params_inner.get('n_components', max(1, min(X_train_outer.shape[0] - 1, X_train_outer.shape[1], num_classes)))
            steps.append(('feature_selection', PLSFeatureSelector(n_components=best_n_components, max_iter=1000, tol=1e-06)))
        elif feature_selection_method == 'tsne':
            pass
        else:
            pass

        best_C = best_params_inner.get('C', 1.0)
        best_gamma = best_params_inner.get('gamma', 'scale')
        steps.append(('svm', SVC(kernel='rbf', C=best_C, gamma=best_gamma, probability=True, random_state=1234, class_weight='balanced')))
        best_model_inner = Pipeline(steps)

        with SuppressOutput():
            try:
                best_model_inner.fit(X_train_outer, y_train_outer)
            except (NotImplementedError, ArpackError, ValueError) as e:
                outer_f1_scores.append(0.0)
                outer_auc_scores.append(0.0)
                print(f"Fold {fold_idx} - F1 Score: 0.0000, AUC: 0.0000 ({str(e)})")
                fold_idx += 1
                continue

        try:
            y_pred_prob_outer = best_model_inner.predict_proba(X_test_outer)
            y_pred_class_outer = best_model_inner.predict(X_test_outer)
        except (NotImplementedError, ArpackError, ValueError) as e:
            y_pred_prob_outer = np.zeros((X_test_outer.shape[0], num_classes))
            y_pred_class_outer = np.zeros(X_test_outer.shape[0])
            print(f"Prediction failed for fold {fold_idx} due to: {str(e)}")

        f1_outer = f1_score(y_test_outer, y_pred_class_outer, average='weighted')
        outer_f1_scores.append(f1_outer)

        if num_classes == 2 and y_pred_prob_outer.shape[1] == 2:
            try:
                fpr_val, tpr_val, _ = roc_curve(y_test_outer, y_pred_prob_outer[:, 1])
                auc_outer = auc(fpr_val, tpr_val)
            except ValueError:
                auc_outer = 0.0
        else:
            try:
                y_bin_test = label_binarize(y_test_outer, classes=classes_enc)
                if y_bin_test.ndim == 1:
                    y_bin_test = np.vstack([1 - y_bin_test, y_bin_test]).T
                fpr_val, tpr_val, _ = roc_curve(y_bin_test.ravel(), y_pred_prob_outer.ravel())
                auc_outer = auc(fpr_val, tpr_val)
            except ValueError:
                auc_outer = 0.0
        outer_auc_scores.append(auc_outer)

        print(f"Fold {fold_idx} - F1 Score: {f1_outer:.4f}, AUC: {auc_outer:.4f}")
        fold_idx += 1

    # plot outer fold metrics
    plt.figure(figsize=(10, 6))
    folds = range(1, cv_outer.get_n_splits() + 1)
    plt.plot(folds, outer_f1_scores, marker='o', linestyle='-', label='F1 Score')
    plt.plot(folds, outer_auc_scores, marker='s', linestyle='-', label='AUC Score')
    plt.title('F1 and AUC Scores per Outer Fold', fontsize=TITLE_FONTSIZE, fontweight='bold', pad=15)
    plt.xlabel('Outer Fold Number', fontsize=LABEL_FONTSIZE, labelpad=10)
    plt.ylabel('Score (F1 / AUC)', fontsize=LABEL_FONTSIZE, labelpad=10)
    plt.xticks(folds, fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{prefix}_svm_nested_cv_f1_auc.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Nested cross-validation completed.")
    print(f"Average F1 Score: {np.mean(outer_f1_scores):.4f} ± {np.std(outer_f1_scores):.4f}")
    print(f"Average AUC: {np.mean(outer_auc_scores):.4f} ± {np.std(outer_auc_scores):.4f}")

    # full-data tuning
    print("Starting hyperparameter tuning on the entire dataset...")

    def objective_full(trial):
        C = trial.suggest_loguniform('C', 1e-2, 1e4)
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])

        steps = [('scaler', StandardScaler())]

        if feature_selection_method == 'elasticnet':
            l1_ratio = trial.suggest_uniform('l1_ratio', 0.0, 1.0)
            C_lr = trial.suggest_loguniform('C_lr', 1e-2, 1e2)
            steps.append((
                'feature_selection',
                SelectFromModel(
                    LogisticRegression(
                        penalty='elasticnet',
                        solver='saga',
                        l1_ratio=l1_ratio,
                        C=C_lr,
                        max_iter=200000,
                        class_weight='balanced',
                        random_state=1234
                    )
                )
            ))
        elif feature_selection_method == 'pca':
            max_pca_components_full = max(1, min(X_use.shape[1], X_use.shape[0]))
            n_components = trial.suggest_int('n_components', 1, max_pca_components_full)
            steps.append(('feature_selection', PCA(n_components=n_components, random_state=1234)))
        elif feature_selection_method == 'kpca':
            n_all = X_use.shape[0]
            max_kpca_components_full = max(1, min(X_use.shape[1], n_all - 1))
            n_components = trial.suggest_int('n_components', 1, max_kpca_components_full)
            kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid', 'cosine'])
            kpca_params = {
                'n_components': n_components,
                'kernel': kernel,
                'random_state': 1234,
                'eigen_solver': 'arpack',
                'max_iter': 5000
            }
            if kernel in ['poly', 'rbf', 'sigmoid']:
                kpca_params['gamma'] = trial.suggest_loguniform('kpca_gamma', 1e-4, 1e1)
            if kernel in ['poly', 'sigmoid']:
                kpca_params['coef0'] = trial.suggest_uniform('kpca_coef0', 0.0, 1.0)
            if kernel == 'poly':
                kpca_params['degree'] = trial.suggest_int('kpca_degree', 2, 5)
            steps.append(('feature_selection', KernelPCA(**kpca_params)))
        elif feature_selection_method == 'umap':
            n_all = X_use.shape[0]
            max_umap_components_full = max(2, min(X_use.shape[1], n_all - 1))
            n_components_umap = trial.suggest_int('n_components', 2, max_umap_components_full)
            n_neighbors_umap = trial.suggest_int('n_neighbors', 5, min(50, n_all - 1))
            n_neighbors_umap = min(n_neighbors_umap, max(2, n_all - 2))
            min_dist_umap = trial.suggest_uniform('min_dist', 0.0, 0.99)
            steps.append(('feature_selection', safe_umap(
                n_components=n_components_umap,
                n_neighbors=n_neighbors_umap,
                min_dist=min_dist_umap,
                X=X_use
            )))
        elif feature_selection_method == 'pls':
            n_all = X_use.shape[0]
            max_pls_components_full = max(1, min(n_all - 1, X_use.shape[1], num_classes))
            n_components_pls = trial.suggest_int('n_components', 1, max_pls_components_full)
            steps.append(('feature_selection', PLSFeatureSelector(n_components=n_components_pls, max_iter=1000, tol=1e-06)))
        elif feature_selection_method == 'tsne':
            pass
        else:
            pass

        steps.append(('svm', SVC(kernel='rbf', C=C, gamma=gamma, probability=True, random_state=1234, class_weight='balanced')))
        pipeline = Pipeline(steps)

        with SuppressOutput():
            f1_scores = []
            for train_idx_full, valid_idx_full in cv_outer.split(X_use, y_encoded):
                X_train_full, X_valid_full = X_use.iloc[train_idx_full], X_use.iloc[valid_idx_full]
                y_train_full, y_valid_full = y_encoded[train_idx_full], y_encoded[valid_idx_full]
                try:
                    pipeline.fit(X_train_full, y_train_full)
                    y_pred_full = pipeline.predict(X_valid_full)
                    f1_val = f1_score(y_valid_full, y_pred_full, average='weighted')
                    f1_scores.append(f1_val)
                except (ArpackError, NotImplementedError, ValueError):
                    f1_scores.append(0.0)
            return float(np.mean(f1_scores))

    study_full = optuna.create_study(direction='maximize', sampler=TPESampler(seed=1234))
    study_full.optimize(objective_full, n_trials=50, show_progress_bar=True)

    best_params_full = study_full.best_params
    print(f"Best parameters: {best_params_full}")

    steps = [('scaler', StandardScaler())]

    if feature_selection_method == 'elasticnet':
        best_l1_ratio_full = best_params_full.get('l1_ratio', 0.5)
        best_C_lr_full = best_params_full.get('C_lr', 1.0)
        steps.append((
            'feature_selection',
            SelectFromModel(
                LogisticRegression(
                    penalty='elasticnet',
                    solver='saga',
                    l1_ratio=best_l1_ratio_full,
                    C=best_C_lr_full,
                    max_iter=200000,
                    class_weight='balanced',
                    random_state=1234
                )
            )
        ))
    elif feature_selection_method == 'pca':
        best_n_components_full = best_params_full.get('n_components', max(1, min(X_use.shape[1], X_use.shape[0])))
        steps.append(('feature_selection', PCA(n_components=best_n_components_full, random_state=1234)))
    elif feature_selection_method == 'kpca':
        best_n_components_full = best_params_full.get('n_components', max(1, min(X_use.shape[1], X_use.shape[0] - 1)))
        best_kernel_full = best_params_full.get('kernel', 'rbf')
        kpca_params = {
            'n_components': best_n_components_full,
            'kernel': best_kernel_full,
            'random_state': 1234,
            'eigen_solver': 'arpack',
            'max_iter': 5000
        }
        if best_kernel_full in ['poly', 'rbf', 'sigmoid']:
            kpca_params['gamma'] = best_params_full.get('kpca_gamma', 1.0)
        if best_kernel_full in ['poly', 'sigmoid']:
            kpca_params['coef0'] = best_params_full.get('kpca_coef0', 0.0)
        if best_kernel_full == 'poly':
            kpca_params['degree'] = best_params_full.get('kpca_degree', 3)
        steps.append(('feature_selection', KernelPCA(**kpca_params)))
    elif feature_selection_method == 'umap':
        best_n_components_full = best_params_full.get('n_components', max(2, min(X_use.shape[1], X_use.shape[0] - 1)))
        best_n_neighbors_full = best_params_full.get('n_neighbors', min(15, max(2, X_use.shape[0] - 2)))
        best_min_dist_full = best_params_full.get('min_dist', 0.1)
        steps.append(('feature_selection', safe_umap(
            n_components=best_n_components_full,
            n_neighbors=best_n_neighbors_full,
            min_dist=best_min_dist_full,
            X=X_use
        )))
    elif feature_selection_method == 'pls':
        best_n_components_full = best_params_full.get('n_components', max(1, min(X_use.shape[0] - 1, X_use.shape[1], num_classes)))
        steps.append(('feature_selection', PLSFeatureSelector(n_components=best_n_components_full, max_iter=1000, tol=1e-06)))
    elif feature_selection_method == 'tsne':
        pass
    else:
        pass

    best_C_full = best_params_full.get('C', 1.0)
    best_gamma_full = best_params_full.get('gamma', 'scale')
    steps.append(('svm', SVC(kernel='rbf', C=best_C_full, gamma=best_gamma_full, probability=True, random_state=1234, class_weight='balanced')))
    best_model = Pipeline(steps)

    with SuppressOutput():
        try:
            best_model.fit(X_use, y_encoded)
        except (NotImplementedError, ArpackError, ValueError):
            print("Feature selection failed on the entire dataset. Training SVM with scaler only.")
            steps = [('scaler', StandardScaler()),
                     ('svm', SVC(kernel='rbf', C=best_C_full, gamma=best_gamma_full, probability=True, random_state=1234, class_weight='balanced'))]
            best_model = Pipeline(steps)
            best_model.fit(X_use, y_encoded)

    # save
    joblib.dump(best_model, f"{prefix}_svm_model.pkl")
    joblib.dump((X_use, y_encoded, le), f"{prefix}_svm_data.pkl")

    print(f"Best parameters: {best_params_full}")

    # transformed export and variance info
    if feature_selection_method != 'none':
        if feature_selection_method == 'tsne':
            X_transformed = X_use.values
            transformed_columns = [f"TSNE_Component_{i+1}" for i in range(X_transformed.shape[1])]
            X_transformed_df = pd.DataFrame(X_transformed, columns=transformed_columns)
        elif 'feature_selection' in best_model.named_steps:
            try:
                X_transformed = best_model.named_steps['feature_selection'].transform(X_use)
            except (NotImplementedError, ArpackError, ValueError):
                X_transformed = None
            if X_transformed is not None:
                if feature_selection_method in ['pca', 'kpca', 'umap', 'pls']:
                    n_components = X_transformed.shape[1]
                    transformed_columns = [f"{feature_selection_method.upper()}_Component_{i+1}" for i in range(n_components)]
                    X_transformed_df = pd.DataFrame(X_transformed, columns=transformed_columns)
                elif feature_selection_method == 'elasticnet':
                    mask = best_model.named_steps['feature_selection'].get_support() if hasattr(best_model.named_steps['feature_selection'], 'get_support') else None
                    if mask is None and hasattr(best_model.named_steps['feature_selection'], 'selector') and hasattr(best_model.named_steps['feature_selection'].selector, 'get_support'):
                        mask = best_model.named_steps['feature_selection'].selector.get_support()
                    if mask is None:
                        X_transformed_df = pd.DataFrame(X_transformed)
                    else:
                        selected_features = X_use.columns[mask]
                        X_transformed_df = X_use[selected_features].copy()
                else:
                    X_transformed_df = pd.DataFrame(X_transformed)
            else:
                X_transformed_df = None
        else:
            X_transformed_df = None

        if X_transformed_df is not None:
            X_transformed_df.insert(0, 'SampleID', sample_ids)
            X_transformed_df['Label'] = y
            transformed_csv_path = f"{prefix}_svm_transformed_X.csv"
            X_transformed_df.to_csv(transformed_csv_path, index=False)
            print(f"Transformed data saved to {transformed_csv_path}")

            variance_csv_path = f"{prefix}_svm_variance.csv"
            if feature_selection_method == 'pca':
                n_comp_full = min(X_use.shape[0], X_use.shape[1])
                full_pca = PCA(n_components=n_comp_full, random_state=1234)
                with SuppressOutput():
                    full_pca.fit(X_use)
                explained_variance = full_pca.explained_variance_ratio_
                pd.DataFrame({
                    'Component': range(1, len(explained_variance) + 1),
                    'Explained Variance Ratio': explained_variance
                }).to_csv(variance_csv_path, index=False)
                print(f"PCA explained variance ratios saved to {variance_csv_path}")
            elif feature_selection_method == 'pls':
                y_full = label_binarize(y_encoded, classes=classes_enc)
                if y_full.ndim == 1:
                    y_full = np.vstack([1 - y_full, y_full]).T
                pls_n_comp_full = max(1, min(X_use.shape[0] - 1, X_use.shape[1], num_classes))
                full_pls = PLSRegression(n_components=pls_n_comp_full, max_iter=1000, tol=1e-06)
                with SuppressOutput():
                    full_pls.fit(X_use, y_full)
                x_scores = full_pls.x_scores_
                explained_variance = np.var(x_scores, axis=0) / np.var(X_use, axis=0).sum()
                pd.DataFrame({
                    'Component': range(1, len(explained_variance) + 1),
                    'Explained Variance Ratio': explained_variance
                }).to_csv(variance_csv_path, index=False)
                print(f"PLS explained variance ratios saved to {variance_csv_path}")
            elif feature_selection_method == 'kpca':
                with open(variance_csv_path, 'w') as f:
                    f.write("KernelPCA does not provide variance information.\n")
                print(f"No variance information available for KernelPCA. File created at {variance_csv_path}")
            elif feature_selection_method == 'umap':
                with open(variance_csv_path, 'w') as f:
                    f.write("UMAP does not provide variance information.\n")
                print(f"No variance information available for UMAP. File created at {variance_csv_path}")
            elif feature_selection_method == 'tsne':
                with open(variance_csv_path, 'w') as f:
                    f.write("t-SNE does not provide variance information.\n")
                print(f"No variance information available for t-SNE. File created at {variance_csv_path}")
            elif feature_selection_method == 'elasticnet':
                with open(variance_csv_path, 'w') as f:
                    f.write("ElasticNet does not provide variance information.\n")
                print(f"No variance information available for ElasticNet. File created at {variance_csv_path}")
            else:
                with open(variance_csv_path, 'w') as f:
                    f.write(f"{feature_selection_method.upper()} does not provide variance information.\n")
                print(f"No variance information available for {feature_selection_method.upper()}. File created at {variance_csv_path}")
    else:
        print("No feature selection method selected. Skipping transformed data and variance information saving.")

    # predictions via cross_val_predict on the exact final pipeline and X_use
    try:
        y_pred_prob = cross_val_predict(best_model, X_use, y_encoded, cv=cv_outer, method='predict_proba', n_jobs=-1)
        y_pred_class = np.argmax(y_pred_prob, axis=1)
    except (NotImplementedError, ArpackError, ValueError):
        y_pred_prob = np.zeros((X_use.shape[0], num_classes))
        y_pred_class = np.zeros(X_use.shape[0])

    acc = accuracy_score(y_encoded, y_pred_class)
    f1 = f1_score(y_encoded, y_pred_class, average='weighted')
    cm = confusion_matrix(y_encoded, y_pred_class)

    if num_classes == 2:
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            sensitivity = 0.0
            specificity = 0.0
            warnings.warn("Confusion matrix is not 2x2. Sensitivity and Specificity set to 0.")
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            sensitivity = float(np.mean(
                np.divide(
                    np.diag(cm),
                    np.sum(cm, axis=1),
                    out=np.zeros_like(np.diag(cm), dtype=float),
                    where=np.sum(cm, axis=1) != 0
                )
            ))
        specificity = multiclass_specificity(cm)

    # confusion matrix plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for SVM', fontsize=TITLE_FONTSIZE, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=LABEL_FONTSIZE, labelpad=10)
    plt.ylabel('True Label', fontsize=LABEL_FONTSIZE, labelpad=10)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.savefig(f"{prefix}_svm_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ROC data and plots
    fpr = {}
    tpr = {}
    roc_auc = {}

    if num_classes == 2 and y_pred_prob.shape[1] == 2:
        try:
            fpr[0], tpr[0], _ = roc_curve(y_encoded, y_pred_prob[:, 1])
            roc_auc[0] = auc(fpr[0], tpr[0])
        except ValueError:
            roc_auc[0] = 0.0
    else:
        fpr, tpr, roc_auc = compute_roc_multi(y_encoded, y_pred_prob, classes_enc)

    roc_data = {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}
    np.save(f"{prefix}_svm_roc_data.npy", roc_data, allow_pickle=True)

    plt.figure(figsize=(10, 8))
    if num_classes == 2 and y_pred_prob.shape[1] == 2:
        plt.plot(fpr[0], tpr[0], label=f'AUC = {roc_auc[0]:.2f}')
    else:
        for i in range(num_classes):
            if i in roc_auc and roc_auc[i] > 0.0:
                plt.plot(fpr[i], tpr[i], label=f'{le.inverse_transform([i])[0]} (AUC = {roc_auc[i]:.2f})')
        if "micro" in roc_auc and roc_auc["micro"] > 0.0:
            plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})', linestyle='--')
        if "macro" in roc_auc and roc_auc["macro"] > 0.0:
            plt.plot(fpr["macro"], tpr["macro"], label=f'Macro-average (AUC = {roc_auc["macro"]:.2f})', linestyle=':')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=LABEL_FONTSIZE, labelpad=10)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=LABEL_FONTSIZE, labelpad=10)
    plt.title('ROC Curves for SVM', fontsize=TITLE_FONTSIZE, fontweight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=LEGEND_FONTSIZE)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.tight_layout()
    plt.savefig(f"{prefix}_svm_roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    # metrics bar
    metrics = {'Accuracy': acc, 'F1 Score': f1, 'Sensitivity': sensitivity, 'Specificity': specificity}
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    ax = metrics_df.plot(kind='bar', x='Metric', y='Value', legend=False)
    plt.title('Performance Metrics for SVM', fontsize=TITLE_FONTSIZE, fontweight='bold')
    plt.xlabel('Metric', fontsize=LABEL_FONTSIZE, labelpad=10)
    plt.ylabel('Value', fontsize=LABEL_FONTSIZE)
    plt.ylim(0, 1.1)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=5)
    plt.xticks(rotation=45, ha='right', fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.tight_layout()
    plt.savefig(f"{prefix}_svm_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()

    # predictions export
    predictions_df = pd.DataFrame({
        'SampleID': sample_ids,
        'Original Label': y,
        'Predicted Label': le.inverse_transform(y_pred_class.astype(int))
    })
    predictions_df.to_csv(f"{prefix}_svm_predictions.csv", index=False)
    print(f"Predictions saved to {prefix}_svm_predictions.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SVM with Nested Cross-Validation, Optional Feature Selection, and Optuna hyperparameter optimization.')
    parser.add_argument('-i', '--csv', type=str, help='Input file in CSV format', required=True)
    parser.add_argument('-p', '--prefix', type=str, help='Prefix for output files', required=True)
    parser.add_argument('-f', '--feature_selection', type=str, choices=['none', 'elasticnet', 'pca', 'kpca', 'umap', 'pls', 'tsne'], default='none', help='Feature selection method to use.')
    args = parser.parse_args()

    svm_nested_cv(args.csv, args.prefix, args.feature_selection)
