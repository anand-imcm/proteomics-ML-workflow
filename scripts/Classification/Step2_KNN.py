import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, LabelBinarizer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import ElasticNet, LogisticRegression
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

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)

def enlarge_fonts(ax, label=22, ticks=18, title=24, legend=20, legend_title=22):
    # axis labels and title
    ax.xaxis.label.set_size(label)
    ax.yaxis.label.set_size(label)
    if hasattr(ax, "title"):
        ax.title.set_size(title)

    # tick labels
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontsize(ticks)

    # legend text and legend title
    leg = ax.get_legend()
    if leg is not None:
        for txt in leg.get_texts():
            txt.set_fontsize(legend)
        if leg.get_title() is not None:
            leg.get_title().set_fontsize(legend_title)

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

class PLSFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.pls = None
        self._lb = LabelBinarizer()

    def fit(self, X, y):
        Y = self._lb.fit_transform(y)
        n_comp = max(1, min(self.n_components, X.shape[1], X.shape[0] - 1))
        self.pls = PLSRegression(n_components=n_comp)
        self.pls.fit(X, Y)
        return self

    def transform(self, X):
        return self.pls.transform(X)

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
        if self.X_transformed_ is not None and X.shape[0] == self.X_transformed_.shape[0]:
            return self.X_transformed_
        else:
            raise NotImplementedError("TSNETransformer does not support transforming new data.")

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

def multiclass_specificity(cm):
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

def knn_nested_cv(inp, prefix, feature_selection_method):
    data = pd.read_csv(inp)

    if 'SampleID' not in data.columns or 'Label' not in data.columns:
        raise ValueError("Input data must contain 'SampleID' and 'Label' columns.")

    sample_ids = data['SampleID']
    X = data.drop(columns=['SampleID', 'Label']).copy()
    y = data['Label']

    scaler = StandardScaler()

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_binarized = pd.get_dummies(y_encoded).values
    num_classes = len(np.unique(y_encoded))

    if feature_selection_method == 'pca':
        max_pca_components = min(X.shape[0], X.shape[1]) - 1
        if max_pca_components < 1:
            max_pca_components = 1
        full_pca = PCA(n_components=max_pca_components, random_state=1234)
        X_pca_full = full_pca.fit_transform(X)
        explained_variance = full_pca.explained_variance_ratio_
        explained_variance_df = pd.DataFrame({
            'Component': range(1, len(explained_variance) + 1),
            'Explained Variance Ratio': explained_variance
        })
        explained_variance_df.to_csv(f"{prefix}_knn_pca_explained_variance_full.csv", index=False)
        X_pca_full_df = pd.DataFrame(X_pca_full, columns=[f"PCA_Component_{i+1}" for i in range(X_pca_full.shape[1])])
        X_pca_full_df.insert(0, 'SampleID', sample_ids)
        X_pca_full_df['Label'] = y
        X_pca_full_df.to_csv(f"{prefix}_knn_pca_all_components.csv", index=False)

    elif feature_selection_method == 'pls':
        max_pls_components = min(X.shape[0] - 1, X.shape[1])
        max_pls_components = max(1, max_pls_components)
        pls = PLSRegression(n_components=max_pls_components)
        with SuppressOutput():
            lb_tmp = LabelBinarizer()
            Y_full = lb_tmp.fit_transform(y_encoded)
            X_pls_full = pls.fit_transform(X, Y_full)[0]
        explained_variance = np.var(X_pls_full, axis=0) / np.var(X, axis=0).sum()
        explained_variance_df = pd.DataFrame({
            'Component': range(1, len(explained_variance) + 1),
            'Explained Variance Ratio': explained_variance
        })
        explained_variance_df.to_csv(f"{prefix}_knn_pls_explained_variance_full.csv", index=False)
        X_pls_full_df = pd.DataFrame(X_pls_full, columns=[f"PLS_Component_{i+1}" for i in range(X_pls_full.shape[1])])
        X_pls_full_df.insert(0, 'SampleID', sample_ids)
        X_pls_full_df['Label'] = y
        X_pls_full_df.to_csv(f"{prefix}_knn_pls_all_components.csv", index=False)

    elif feature_selection_method == 'kpca':
        n_samples = X.shape[0]
        n_features = X.shape[1]
        max_kpca_components = min(n_features, n_samples - 1)
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
        X_kpca_full_df.to_csv(f"{prefix}_knn_kpca_all_components.csv", index=False)

    elif feature_selection_method == 'umap':
        umap_full = safe_umap(
            n_components=min(X.shape[1], 100),
            n_neighbors=15,
            min_dist=0.1,
            X=X
        )
        with SuppressOutput():
            X_umap_full = umap_full.fit_transform(X)
        X_umap_full_df = pd.DataFrame(
            X_umap_full,
            columns=[f"UMAP_Component_{i+1}" for i in range(X_umap_full.shape[1])]
        )
        X_umap_full_df.insert(0, 'SampleID', sample_ids)
        X_umap_full_df['Label'] = y
        X_umap_full_df.to_csv(f"{prefix}_knn_umap_all_components.csv", index=False)

    elif feature_selection_method == 'tsne':
        tsne_full = TSNETransformer(
            n_components=2,
            perplexity=30,
            learning_rate=200,
            max_iter=1000,
            random_state=1234
        )
        with SuppressOutput():
            X_tsne_full = tsne_full.fit_transform(X)
        X_tsne_full_df = pd.DataFrame(
            X_tsne_full,
            columns=[f"TSNE_Component_{i+1}" for i in range(X_tsne_full.shape[1])]
        )
        X_tsne_full_df.insert(0, 'SampleID', sample_ids)
        X_tsne_full_df['Label'] = y
        X_tsne_full_df.to_csv(f"{prefix}_knn_tsne_all_components.csv", index=False)

    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
    outer_f1_scores = []
    outer_auc_scores = []

    fold_idx = 1
    for train_idx, test_idx in cv_outer.split(X, y_encoded):
        print(f"Processing outer fold {fold_idx}...")
        X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
        y_train_outer, y_test_outer = y_encoded[train_idx], y_encoded[test_idx]

        cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

        def objective_inner(trial):
            steps_inner = [('scaler', StandardScaler())]

            if feature_selection_method == 'elasticnet':
                lr_C = trial.suggest_loguniform('lr_C', 1e-3, 1e3)
                lr_l1_ratio = trial.suggest_uniform('lr_l1_ratio', 0.0, 1.0)
                steps_inner.append((
                    'feature_selection',
                    SelectFromModel(
                        LogisticRegression(
                            penalty='elasticnet',
                            solver='saga',
                            l1_ratio=lr_l1_ratio,
                            C=lr_C,
                            max_iter=10000,
                            random_state=1234
                        )
                    )
                ))
            elif feature_selection_method == 'pca':
                max_pca_components = min(X_train_outer.shape[1], X_train_outer.shape[0] - 1)
                n_components = trial.suggest_int('pca_n_components', 1, max_pca_components)
                steps_inner.append((
                    'feature_selection',
                    PCA(n_components=n_components, random_state=1234)
                ))
            elif feature_selection_method == 'kpca':
                n_samples_train = X_train_outer.shape[0]
                max_kpca_components = min(X_train_outer.shape[1], n_samples_train - 1)
                max_kpca_components = max(1, max_kpca_components)
                n_components = trial.suggest_int('kpca_n_components', 1, max_kpca_components)
                kernel = trial.suggest_categorical(
                    'kpca_kernel',
                    ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']
                )
                kpca_params = {
                    'n_components': n_components,
                    'kernel': kernel,
                    'random_state': 1234,
                    'eigen_solver': 'arpack',
                    'max_iter': 5000
                }
                if kernel in ['poly', 'rbf', 'sigmoid']:
                    kpca_gamma = trial.suggest_loguniform('kpca_gamma', 1e-4, 1e1)
                    kpca_params['gamma'] = kpca_gamma
                if kernel in ['poly', 'sigmoid']:
                    kpca_coef0 = trial.suggest_uniform('kpca_coef0', 0.0, 1.0)
                    kpca_params['coef0'] = kpca_coef0
                if kernel == 'poly':
                    kpca_degree = trial.suggest_int('kpca_degree', 2, 5)
                    kpca_params['degree'] = kpca_degree
                steps_inner.append(('feature_selection', KernelPCA(**kpca_params)))
            elif feature_selection_method == 'umap':
                max_umap_components = min(X_train_outer.shape[1], 100)
                n_components = trial.suggest_int('umap_n_components', 2, max_umap_components)
                umap_n_neighbors = trial.suggest_int(
                    'umap_n_neighbors',
                    5,
                    min(50, X_train_outer.shape[0] - 1)
                )
                min_dist = trial.suggest_uniform('umap_min_dist', 0.0, 0.99)
                steps_inner.append((
                    'feature_selection',
                    safe_umap(
                        n_components=n_components,
                        n_neighbors=umap_n_neighbors,
                        min_dist=min_dist,
                        X=X_train_outer
                    )
                ))

            elif feature_selection_method == 'pls':
                max_pls_components = min(X_train_outer.shape[1], X_train_outer.shape[0] - 1)
                max_pls_components = max(1, max_pls_components)
                n_components = trial.suggest_int('pls_n_components', 1, max_pls_components)
                steps_inner.append(('feature_selection', PLSFeatureSelector(n_components=n_components)))
            elif feature_selection_method == 'tsne':
                n_components_tsne = trial.suggest_int('tsne_n_components', 2, 3)
                perplexity_tsne = trial.suggest_int(
                    'tsne_perplexity',
                    5,
                    min(50, X_train_outer.shape[0] - 1)
                )
                learning_rate_tsne = trial.suggest_loguniform('tsne_learning_rate', 10, 1000)
                max_iter_tsne = trial.suggest_int('tsne_max_iter', 250, 2000)
                steps_inner.append((
                    'feature_selection',
                    TSNETransformer(
                        n_components=n_components_tsne,
                        perplexity=perplexity_tsne,
                        learning_rate=learning_rate_tsne,
                        max_iter=max_iter_tsne,
                        random_state=1234
                    )
                ))

            knn_n_neighbors = trial.suggest_int(
                'knn_n_neighbors',
                1,
                min(30, len(X_train_outer) - 1)
            )
            steps_inner.append(('knn', KNeighborsClassifier(n_neighbors=knn_n_neighbors, n_jobs=-1)))

            pipeline_inner = Pipeline(steps_inner)

            with SuppressOutput():
                f1_scores_inner = []
                for inner_train_idx, inner_valid_idx in cv_inner.split(
                    X_train_outer,
                    y_train_outer
                ):
                    X_train_inner = X_train_outer.iloc[inner_train_idx]
                    X_valid_inner = X_train_outer.iloc[inner_valid_idx]
                    y_train_inner = y_train_outer[inner_train_idx]
                    y_valid_inner = y_train_outer[inner_valid_idx]
                    try:
                        pipeline_inner.fit(X_train_inner, y_train_inner)
                        y_pred_inner = pipeline_inner.predict(X_valid_inner)
                        f1_val = f1_score(y_valid_inner, y_pred_inner, average='weighted')
                        f1_scores_inner.append(f1_val)
                    except (NotImplementedError, ArpackError, ValueError):
                        return 0.0
                return np.mean(f1_scores_inner)

        study_inner = optuna.create_study(direction='maximize', sampler=TPESampler(seed=1234))
        study_inner.optimize(objective_inner, n_trials=50, show_progress_bar=False)

        best_params_inner = study_inner.best_params

        steps_outer = [('scaler', StandardScaler())]

        if feature_selection_method == 'elasticnet':
            best_lr_C = best_params_inner['lr_C']
            best_lr_l1_ratio = best_params_inner['lr_l1_ratio']
            steps_outer.append((
                'feature_selection',
                SelectFromModel(
                    LogisticRegression(
                        penalty='elasticnet',
                        solver='saga',
                        l1_ratio=best_lr_l1_ratio,
                        C=best_lr_C,
                        max_iter=10000,
                        random_state=1234
                    )
                )
            ))
        elif feature_selection_method == 'pca':
            best_pca_n_components = best_params_inner['pca_n_components']
            steps_outer.append((
                'feature_selection',
                PCA(n_components=best_pca_n_components, random_state=1234)
            ))
        elif feature_selection_method == 'kpca':
            best_kpca_n_components = best_params_inner['kpca_n_components']
            best_kpca_kernel = best_params_inner['kpca_kernel']
            kpca_params_best = {
                'n_components': best_kpca_n_components,
                'kernel': best_kpca_kernel,
                'random_state': 1234,
                'eigen_solver': 'arpack',
                'max_iter': 5000
            }
            if best_kpca_kernel in ['poly', 'rbf', 'sigmoid']:
                kpca_params_best['gamma'] = best_params_inner['kpca_gamma']
            if best_kpca_kernel in ['poly', 'sigmoid']:
                kpca_params_best['coef0'] = best_params_inner['kpca_coef0']
            if best_kpca_kernel == 'poly':
                kpca_params_best['degree'] = best_params_inner['kpca_degree']
            steps_outer.append(('feature_selection', KernelPCA(**kpca_params_best)))
        elif feature_selection_method == 'umap':
            best_umap_n_components = best_params_inner['umap_n_components']
            best_umap_n_neighbors = best_params_inner['umap_n_neighbors']
            best_umap_min_dist = best_params_inner['umap_min_dist']
            steps_outer.append((
                'feature_selection',
                safe_umap(
                    n_components=best_umap_n_components,
                    n_neighbors=best_umap_n_neighbors,
                    min_dist=best_umap_min_dist,
                    X=X_train_outer
                )
            ))

        elif feature_selection_method == 'pls':
            best_pls_n_components = best_params_inner['pls_n_components']
            steps_outer.append(('feature_selection', PLSFeatureSelector(n_components=best_pls_n_components)))
        elif feature_selection_method == 'tsne':
            best_tsne_n_components = best_params_inner['tsne_n_components']
            best_tsne_perplexity = best_params_inner['tsne_perplexity']
            best_tsne_learning_rate = best_params_inner['tsne_learning_rate']
            best_tsne_max_iter = best_params_inner['tsne_max_iter']
            steps_outer.append((
                'feature_selection',
                TSNETransformer(
                    n_components=best_tsne_n_components,
                    perplexity=best_tsne_perplexity,
                    learning_rate=best_tsne_learning_rate,
                    max_iter=best_tsne_max_iter,
                    random_state=1234
                )
            ))

        best_knn_n_neighbors = best_params_inner['knn_n_neighbors']
        steps_outer.append(('knn', KNeighborsClassifier(n_neighbors=best_knn_n_neighbors, n_jobs=-1)))

        best_model_inner = Pipeline(steps_outer)
        y_pred_prob_outer = None
        y_pred_class_outer = None

        with SuppressOutput():
            try:
                best_model_inner.fit(X_train_outer, y_train_outer)
                y_pred_prob_outer = best_model_inner.predict_proba(X_test_outer)
                y_pred_class_outer = best_model_inner.predict(X_test_outer)
            except (NotImplementedError, ArpackError, ValueError) as e:
                print(f"Fold {fold_idx} encountered an error: {str(e)}")
                y_pred_prob_outer = np.zeros((X_test_outer.shape[0], num_classes))
                y_pred_class_outer = np.zeros(X_test_outer.shape[0])

        f1_outer = f1_score(y_test_outer, y_pred_class_outer, average='weighted')
        outer_f1_scores.append(f1_outer)

        if num_classes == 2 and y_pred_prob_outer.shape[1] == 2:
            try:
                fpr_fold, tpr_fold, _ = roc_curve(y_test_outer, y_pred_prob_outer[:, 1])
                auc_fold = auc(fpr_fold, tpr_fold)
            except ValueError:
                auc_fold = 0.0
        else:
            try:
                y_binarized_test = y_binarized[test_idx]
                fpr_fold, tpr_fold, _ = roc_curve(y_binarized_test.ravel(), y_pred_prob_outer.ravel())
                auc_fold = auc(fpr_fold, tpr_fold)
            except ValueError:
                auc_fold = 0.0

        outer_auc_scores.append(auc_fold)
        print(f"Fold {fold_idx} - F1 Score: {f1_outer:.4f}, AUC: {auc_fold:.4f}")
        fold_idx += 1

    plt.figure(figsize=(10, 6))
    folds = range(1, cv_outer.get_n_splits() + 1)
    plt.plot(folds, outer_f1_scores, marker='o', label='F1 Score')
    plt.plot(folds, outer_auc_scores, marker='s', label='AUC')
    plt.xlabel('Outer Fold Number', fontsize=18, labelpad=10)
    plt.ylabel('Score (F1 / AUC)', fontsize=18, labelpad=10)
    plt.title('F1 and AUC Scores per Outer Fold', fontsize=16, fontweight='bold', pad=15)
    plt.xticks(folds, fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14, title_fontsize=16)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    ax_cv = plt.gca()             
    enlarge_fonts(ax_cv)     
    plt.savefig(f"{prefix}_knn_nested_cv_f1_auc.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Nested cross-validation completed.")
    print(f"Average F1 Score: {np.mean(outer_f1_scores):.4f} ± {np.std(outer_f1_scores):.4f}")
    print(f"Average AUC: {np.mean(outer_auc_scores):.4f} ± {np.std(outer_auc_scores):.4f}")

    print("Starting hyperparameter tuning on the entire dataset...")

    def objective_full(trial):
        steps_full = [('scaler', StandardScaler())]

        if feature_selection_method == 'elasticnet':
            lr_C_all = trial.suggest_loguniform('lr_C', 1e-3, 1e3)
            lr_l1_ratio_all = trial.suggest_uniform('lr_l1_ratio', 0.0, 1.0)
            steps_full.append((
                'feature_selection',
                SelectFromModel(
                    LogisticRegression(
                        penalty='elasticnet',
                        solver='saga',
                        l1_ratio=lr_l1_ratio_all,
                        C=lr_C_all,
                        max_iter=10000,
                        random_state=1234
                    )
                )
            ))
        elif feature_selection_method == 'pca':
            max_pca_components = min(X.shape[1], X.shape[0] - 1)
            max_pca_components = max(1, max_pca_components)
            n_components = trial.suggest_int('pca_n_components', 1, max_pca_components)
            steps_full.append((
                'feature_selection',
                PCA(n_components=n_components, random_state=1234)
            ))
        elif feature_selection_method == 'kpca':
            n_samples_all = X.shape[0]
            n_features_all = X.shape[1]
            max_kpca_components_all = min(n_features_all, n_samples_all - 1)
            max_kpca_components_all = max(1, max_kpca_components_all)
            n_components_all = trial.suggest_int('kpca_n_components', 1, max_kpca_components_all)
            kernel_all = trial.suggest_categorical(
                'kpca_kernel',
                ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']
            )
            kpca_params_all = {
                'n_components': n_components_all,
                'kernel': kernel_all,
                'random_state': 1234,
                'eigen_solver': 'arpack',
                'max_iter': 5000
            }
            if kernel_all in ['poly', 'rbf', 'sigmoid']:
                kpca_gamma_all = trial.suggest_loguniform('kpca_gamma', 1e-4, 1e1)
                kpca_params_all['gamma'] = kpca_gamma_all
            if kernel_all in ['poly', 'sigmoid']:
                kpca_coef0_all = trial.suggest_uniform('kpca_coef0', 0.0, 1.0)
                kpca_params_all['coef0'] = kpca_coef0_all
            if kernel_all == 'poly':
                kpca_degree_all = trial.suggest_int('kpca_degree', 2, 5)
                kpca_params_all['degree'] = kpca_degree_all
            steps_full.append(('feature_selection', KernelPCA(**kpca_params_all)))
        elif feature_selection_method == 'umap':
            max_umap_components_all = min(X.shape[1], 100)
            n_components_all = trial.suggest_int('umap_n_components', 2, max_umap_components_all)
            umap_n_neighbors_all = trial.suggest_int('umap_n_neighbors', 5, min(50, X.shape[0] - 1))
            min_dist_all = trial.suggest_uniform('umap_min_dist', 0.0, 0.99)
            steps_full.append((
                'feature_selection',
                safe_umap(
                    n_components=n_components_all,
                    n_neighbors=umap_n_neighbors_all,
                    min_dist=min_dist_all,
                    X=X
                )
            ))
        elif feature_selection_method == 'pls':
            max_pls_components_all = min(X.shape[1], X.shape[0] - 1)
            max_pls_components_all = max(1, max_pls_components_all)
            n_components_pls_all = trial.suggest_int('pls_n_components', 1, max_pls_components_all)
            steps_full.append(('feature_selection', PLSFeatureSelector(n_components=n_components_pls_all)))
        elif feature_selection_method == 'tsne':
            n_components_tsne_all = trial.suggest_int('tsne_n_components', 2, 3)
            perplexity_tsne_all = trial.suggest_int('tsne_perplexity', 5, min(50, X.shape[0] - 1))
            learning_rate_tsne_all = trial.suggest_loguniform('tsne_learning_rate', 10, 1000)
            max_iter_tsne_all = trial.suggest_int('tsne_max_iter', 250, 2000)
            steps_full.append((
                'feature_selection',
                TSNETransformer(
                    n_components=n_components_tsne_all,
                    perplexity=perplexity_tsne_all,
                    learning_rate=learning_rate_tsne_all,
                    max_iter=max_iter_tsne_all,
                    random_state=1234
                )
            ))
        else:
            pass

        knn_n_neighbors_all = trial.suggest_int('knn_n_neighbors', 1, min(30, len(X) - 1))
        steps_full.append(('knn', KNeighborsClassifier(n_neighbors=knn_n_neighbors_all, n_jobs=-1)))

        pipeline_full = Pipeline(steps_full)

        with SuppressOutput():
            f1_scores_full = []
            for train_idx_full, valid_idx_full in cv_outer.split(X, y_encoded):
                X_train_full = X.iloc[train_idx_full]
                X_valid_full = X.iloc[valid_idx_full]
                y_train_full = y_encoded[train_idx_full]
                y_valid_full = y_encoded[valid_idx_full]
                try:
                    pipeline_full.fit(X_train_full, y_train_full)
                    y_pred_full = pipeline_full.predict(X_valid_full)
                    f1_val_full = f1_score(y_valid_full, y_pred_full, average='weighted')
                    f1_scores_full.append(f1_val_full)
                except (NotImplementedError, ArpackError, ValueError):
                    f1_scores_full.append(0.0)
            return np.mean(f1_scores_full)

    study_full = optuna.create_study(direction='maximize', sampler=TPESampler(seed=1234))
    study_full.optimize(objective_full, n_trials=50, show_progress_bar=True)
    best_params_full = study_full.best_params
    print(f"Best parameters for KNN: {best_params_full}")

    steps_final = [('scaler', StandardScaler())]

    if feature_selection_method == 'elasticnet':
        lr_C_best = best_params_full['lr_C']
        lr_l1_ratio_best = best_params_full['lr_l1_ratio']
        steps_final.append((
            'feature_selection',
            SelectFromModel(
                LogisticRegression(
                    penalty='elasticnet',
                    solver='saga',
                    l1_ratio=lr_l1_ratio_best,
                    C=lr_C_best,
                    max_iter=10000,
                    random_state=1234
                )
            )
        ))
    elif feature_selection_method == 'pca':
        best_pca_n_components_full = best_params_full['pca_n_components']
        steps_final.append((
            'feature_selection',
            PCA(n_components=best_pca_n_components_full, random_state=1234)
        ))
    elif feature_selection_method == 'kpca':
        best_kpca_n_components_full = best_params_full['kpca_n_components']
        best_kpca_kernel_full = best_params_full['kpca_kernel']
        kpca_params_full = {
            'n_components': best_kpca_n_components_full,
            'kernel': best_kpca_kernel_full,
            'random_state': 1234,
            'eigen_solver': 'arpack',
            'max_iter': 5000
        }
        if best_kpca_kernel_full in ['poly', 'rbf', 'sigmoid']:
            kpca_params_full['gamma'] = best_params_full['kpca_gamma']
        if best_kpca_kernel_full in ['poly', 'sigmoid']:
            kpca_params_full['coef0'] = best_params_full['kpca_coef0']
        if best_kpca_kernel_full == 'poly':
            kpca_params_full['degree'] = best_params_full['kpca_degree']
        steps_final.append(('feature_selection', KernelPCA(**kpca_params_full)))
    elif feature_selection_method == 'umap':
        best_umap_n_components_full = best_params_full['umap_n_components']
        best_umap_n_neighbors_full = best_params_full['umap_n_neighbors']
        best_umap_min_dist_full = best_params_full['umap_min_dist']
        steps_final.append((
            'feature_selection',
            safe_umap(
                n_components=best_umap_n_components_full,
                n_neighbors=best_umap_n_neighbors_full,
                min_dist=best_umap_min_dist_full,
                X=X
            )
        ))

    elif feature_selection_method == 'pls':
        best_pls_n_components_full = best_params_full['pls_n_components']
        steps_final.append(('feature_selection', PLSFeatureSelector(n_components=best_pls_n_components_full)))
    elif feature_selection_method == 'tsne':
        tsne_n_components_full = best_params_full['tsne_n_components']
        tsne_perplexity_full = best_params_full['tsne_perplexity']
        tsne_learning_rate_full = best_params_full['tsne_learning_rate']
        tsne_max_iter_full = best_params_full['tsne_max_iter']
        steps_final.append((
            'feature_selection',
            TSNETransformer(
                n_components=tsne_n_components_full,
                perplexity=tsne_perplexity_full,
                learning_rate=tsne_learning_rate_full,
                max_iter=tsne_max_iter_full,
                random_state=1234
            )
        ))
    else:
        pass

    best_knn_n_neighbors_full = best_params_full['knn_n_neighbors']
    steps_final.append(('knn', KNeighborsClassifier(n_neighbors=best_knn_n_neighbors_full, n_jobs=-1)))

    best_model = Pipeline(steps_final)
    with SuppressOutput():
        try:
            best_model.fit(X, y_encoded)
        except (NotImplementedError, ArpackError, ValueError):
            print("Feature selection method failed on the entire dataset. Skipping feature selection.")
            steps_final = [('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=best_knn_n_neighbors_full, n_jobs=-1))]
            best_model = Pipeline(steps_final)
            best_model.fit(X, y_encoded)

    joblib.dump(best_model, f"{prefix}_knn_model.pkl")
    feature_names = X.columns.tolist()
    joblib.dump((X, y_encoded, le, feature_names), f"{prefix}_knn_data.pkl")

    print(f"Best parameters for KNN: {best_params_full}")

    if feature_selection_method != 'none':
        if 'feature_selection' in best_model.named_steps:
            try:
                X_transformed_final = best_model.named_steps['feature_selection'].transform(X)
            except (NotImplementedError, ArpackError, ValueError):
                X_transformed_final = None
        else:
            X_transformed_final = None

        if X_transformed_final is not None:
            if feature_selection_method in ['pca', 'kpca', 'umap', 'pls']:
                n_comps = X_transformed_final.shape[1]
                transformed_columns = [
                    f"{feature_selection_method.upper()}_Component_{i+1}"
                    for i in range(n_comps)
                ]
                X_transformed_df = pd.DataFrame(
                    X_transformed_final,
                    columns=transformed_columns
                )
            elif feature_selection_method == 'tsne':
                transformed_columns = [
                    f"{feature_selection_method.upper()}_Component_{i+1}"
                    for i in range(X_transformed_final.shape[1])
                ]
                X_transformed_df = pd.DataFrame(
                    X_transformed_final,
                    columns=transformed_columns
                )
            elif feature_selection_method == 'elasticnet':
                selected_mask = best_model.named_steps['feature_selection'].estimator_.coef_
                if selected_mask.ndim == 2:
                    importance = np.abs(selected_mask).mean(axis=0)
                else:
                    importance = np.abs(selected_mask)
                threshold = np.median(importance)
                selected_features = X.columns[importance >= threshold]
                X_transformed_df = X[selected_features].copy()
            else:
                X_transformed_df = pd.DataFrame(X_transformed_final)

            X_transformed_df.insert(0, 'SampleID', sample_ids)
            X_transformed_df['Label'] = y
            transformed_csv_path = f"{prefix}_knn_transformed_X.csv"
            X_transformed_df.to_csv(transformed_csv_path, index=False)

            variance_csv_path = f"{prefix}_knn_variance.csv"
            if feature_selection_method == 'pca':
                pca_step = best_model.named_steps['feature_selection']
                pca_var = pca_step.explained_variance_ratio_
                pca_var_df = pd.DataFrame({
                    'Component': range(1, len(pca_var) + 1),
                    'Explained Variance Ratio': pca_var
                })
                pca_var_df.to_csv(variance_csv_path, index=False)
                print(f"PCA explained variance ratios saved to {variance_csv_path}")
            elif feature_selection_method == 'pls':
                pls_step = best_model.named_steps['feature_selection'].pls
                x_scores_pls = pls_step.x_scores_
                pls_explained_var = np.var(x_scores_pls, axis=0) / np.var(X, axis=0).sum()
                pls_var_df = pd.DataFrame({
                    'Component': range(1, len(pls_explained_var) + 1),
                    'Explained Variance Ratio': pls_explained_var
                })
                pls_var_df.to_csv(variance_csv_path, index=False)
                print(f"PLS explained variance ratios saved to {variance_csv_path}")
            elif feature_selection_method == 'tsne':
                with open(variance_csv_path, 'w') as f:
                    f.write("t-SNE does not provide variance information.\n")
                print(f"No variance information available for t-SNE. File created at {variance_csv_path}")
            elif feature_selection_method == 'elasticnet':
                with open(variance_csv_path, 'w') as f:
                    f.write("ElasticNet-penalized LogisticRegression does not provide variance information.\n")
                print(f"No variance information available for ElasticNet-penalized LogisticRegression. File created at {variance_csv_path}")
            else:
                with open(variance_csv_path, 'w') as f:
                    f.write(f"{feature_selection_method.upper()} does not provide variance information.\n")
                print(f"No variance information available for {feature_selection_method.upper()}. File created at {variance_csv_path}")
        else:
            print("Feature selection transformation was not available.")
    else:
        print("No feature selection method selected. Skipping transformed data.")

    y_pred_prob_final = None
    y_pred_class_final = None
    try:
        y_pred_prob_final = cross_val_predict(
            best_model,
            X,
            y_encoded,
            cv=cv_outer,
            method='predict_proba',
            n_jobs=-1
        )
        y_pred_class_final = np.argmax(y_pred_prob_final, axis=1)
    except (NotImplementedError, ArpackError, ValueError):
        y_pred_prob_final = np.zeros((X.shape[0], num_classes))
        y_pred_class_final = np.zeros(X.shape[0])

    acc_final = accuracy_score(y_encoded, y_pred_class_final)
    f1_final = f1_score(y_encoded, y_pred_class_final, average='weighted')
    cm_final = confusion_matrix(y_encoded, y_pred_class_final)

    if num_classes == 2:
        if cm_final.shape == (2, 2):
            tn, fp, fn, tp = cm_final.ravel()
            sensitivity_final = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity_final = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            sensitivity_final = 0
            specificity_final = 0
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            sensitivity_final = np.mean(
                np.divide(
                    np.diag(cm_final),
                    np.sum(cm_final, axis=1),
                    out=np.zeros_like(np.diag(cm_final), dtype=float),
                    where=np.sum(cm_final, axis=1) != 0
                )
            )
        specificity_final = multiclass_specificity(cm_final)

    disp_final = ConfusionMatrixDisplay(confusion_matrix=cm_final, display_labels=le.classes_)
    disp_final.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for KNN', fontsize=12, fontweight='bold')
    enlarge_fonts(disp_final.ax_)
    plt.savefig(f"{prefix}_knn_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    fpr_final = {}
    tpr_final = {}
    roc_auc_final = {}

    if num_classes == 2 and y_pred_prob_final.shape[1] == 2:
        try:
            fpr_final[0], tpr_final[0], _ = roc_curve(y_encoded, y_pred_prob_final[:, 1])
            roc_auc_final[0] = auc(fpr_final[0], tpr_final[0])
        except ValueError:
            roc_auc_final[0] = 0.0
    else:
        for i in range(y_binarized.shape[1]):
            if np.sum(y_binarized[:, i]) == 0:
                fpr_final[i] = np.array([0, 1])
                tpr_final[i] = np.array([0, 1])
                roc_auc_final[i] = 0.0
            else:
                try:
                    fpr_final[i], tpr_final[i], _ = roc_curve(y_binarized[:, i], y_pred_prob_final[:, i])
                    roc_auc_final[i] = auc(fpr_final[i], tpr_final[i])
                except ValueError:
                    fpr_final[i], tpr_final[i], roc_auc_final[i] = np.array([0, 1]), np.array([0, 1]), 0.0

        try:
            fpr_final["micro"], tpr_final["micro"], _ = roc_curve(y_binarized.ravel(), y_pred_prob_final.ravel())
            roc_auc_final["micro"] = auc(fpr_final["micro"], tpr_final["micro"])
        except ValueError:
            fpr_final["micro"] = np.array([0, 1])
            tpr_final["micro"] = np.array([0, 1])
            roc_auc_final["micro"] = 0.0

        try:
            all_fpr = np.unique(np.concatenate([fpr_final[i] for i in range(y_binarized.shape[1]) if isinstance(i, int) or isinstance(i, np.integer)]))
            mean_tpr = np.zeros_like(all_fpr)
            valid_classes = 0
            for i in range(y_binarized.shape[1]):
                if i in fpr_final and i in tpr_final and len(fpr_final[i]) > 1 and len(tpr_final[i]) > 1:
                    mean_tpr += np.interp(all_fpr, fpr_final[i], tpr_final[i])
                    valid_classes += 1
            if valid_classes > 0:
                mean_tpr /= valid_classes
                fpr_final["macro"] = all_fpr
                tpr_final["macro"] = mean_tpr
                roc_auc_final["macro"] = auc(all_fpr, mean_tpr)
            else:
                fpr_final["macro"] = np.array([0, 1])
                tpr_final["macro"] = np.array([0, 1])
                roc_auc_final["macro"] = 0.0
        except Exception:
            fpr_final["macro"] = np.array([0, 1])
            tpr_final["macro"] = np.array([0, 1])
            roc_auc_final["macro"] = 0.0

    roc_data_final = {
        'fpr': fpr_final,
        'tpr': tpr_final,
        'roc_auc': roc_auc_final
    }
    np.save(f"{prefix}_knn_roc_data.npy", roc_data_final, allow_pickle=True)

    plt.figure(figsize=(10, 8))
    if num_classes == 2 and y_pred_prob_final.shape[1] == 2:
        plt.plot(fpr_final[0], tpr_final[0], label=f'AUC = {roc_auc_final[0]:.2f}')
    else:
        for i in range(len(le.classes_)):
            if roc_auc_final.get(i, 0.0) > 0.0:
                class_label = le.inverse_transform([i])[0]
                plt.plot(fpr_final[i], tpr_final[i], label=f'{class_label} (AUC = {roc_auc_final[i]:.2f})')
        if "micro" in fpr_final and "micro" in tpr_final and "micro" in roc_auc_final:
            plt.plot(
                fpr_final["micro"],
                tpr_final["micro"],
                label=f'Micro-average (AUC = {roc_auc_final["micro"]:.2f})',
                linestyle='--'
            )
        if "macro" in fpr_final and "macro" in tpr_final and "macro" in roc_auc_final:
            plt.plot(
                fpr_final["macro"],
                tpr_final["macro"],
                label=f'Macro-average (AUC = {roc_auc_final["macro"]:.2f})',
                linestyle=':'
            )

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=18, labelpad=10)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=18, labelpad=10)
    plt.title('ROC Curves for KNN', fontsize=22, fontweight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=14, title_fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    ax_roc = plt.gca()            
    enlarge_fonts(ax_roc)         
    plt.savefig(f"{prefix}_knn_roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    metrics_dict_final = {
        'Accuracy': acc_final,
        'F1 Score': f1_final,
        'Sensitivity': sensitivity_final,
        'Specificity': specificity_final
    }
    metrics_df_final = pd.DataFrame(list(metrics_dict_final.items()), columns=['Metric', 'Value'])
    ax_final = metrics_df_final.plot(kind='bar', x='Metric', y='Value', legend=False)
    plt.title('Performance Metrics for KNN', fontsize=14, fontweight='bold')
    plt.ylabel('Value')
    plt.ylim(0, 1.1)
    for container in ax_final.containers:
        ax_final.bar_label(container, fmt='%.2f', label_type='edge', padding=5)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    enlarge_fonts(ax_final)
    plt.savefig(f"{prefix}_knn_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()

    predictions_df_final = pd.DataFrame({
        'SampleID': sample_ids,
        'Original Label': y,
        'Predicted Label': le.inverse_transform(y_pred_class_final.astype(int))
    })
    predictions_df_final.to_csv(f"{prefix}_knn_predictions.csv", index=False)

    print(f"Predictions saved to {prefix}_knn_predictions.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run KNN model with Nested Cross-Validation, Optional Feature Selection, and Optuna hyperparameter optimization.'
    )
    parser.add_argument('-i', '--csv', type=str, required=True, help='Input file in CSV format.')
    parser.add_argument('-p', '--prefix', type=str, required=True, help='Output prefix.')
    parser.add_argument(
        '-f',
        '--feature_selection',
        type=str,
        choices=['none', 'elasticnet', 'pca', 'kpca', 'umap', 'pls', 'tsne'],
        default='none',
        help='Feature selection method to use.'
    )
    args = parser.parse_args()

    knn_nested_cv(args.csv, args.prefix, args.feature_selection)
