import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, label_binarize, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier
from sklearn.linear_model import ElasticNet
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
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)

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
    """
    Modified PLSFeatureSelector to return a DataFrame with columns named
    PLS_Component_1, PLS_Component_2, etc. This ensures SHAP can correctly
    handle the transformed data in a pipeline.
    """
    def __init__(self, n_components=2, max_iter=1000, tol=1e-06):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.pls = None
        self.feature_names_ = None

    def fit(self, X, y):
        if hasattr(X, "columns"):
            self.feature_names_ = X.columns.tolist()
        else:
            self.feature_names_ = [f"Feature_{i}" for i in range(X.shape[1])]
        self.pls = PLSRegression(n_components=self.n_components, max_iter=self.max_iter, tol=self.tol)
        self.pls.fit(X, y)
        return self

    def transform(self, X):
        X_pls = self.pls.transform(X)
        columns = [f"PLS_Component_{i+1}" for i in range(X_pls.shape[1])]
        return pd.DataFrame(X_pls, index=range(X_pls.shape[0]), columns=columns)

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

class ElasticNetFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=100000, tol=1e-4):
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

class PCATransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components, random_state=1234)
        self.explained_variance_ratio_ = None

    def fit(self, X, y=None):
        self.pca.fit(X)
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        return self

    def transform(self, X):
        return self.pca.transform(X)

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

def xgboost_nested_cv(inp, prefix, feature_selection_method):
    data = pd.read_csv(inp)
    if 'SampleID' not in data.columns or 'Label' not in data.columns:
        raise ValueError("Input data must contain 'SampleID' and 'Label' columns.")

    sample_ids = data['SampleID']
    X = data.drop(columns=['SampleID', 'Label']).copy()
    y = data['Label']

    scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    # X = pd.DataFrame(X_scaled, columns=X.columns)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_binarized = label_binarize(y_encoded, classes=np.unique(y_encoded))
    if y_binarized.shape[1] == 1:
        y_binarized = np.hstack([1 - y_binarized, y_binarized])

    num_classes = len(np.unique(y_encoded))

    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
    outer_f1_scores = []
    outer_auc_scores = []

    tsne_selected = (feature_selection_method == 'tsne')

    if tsne_selected:
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

    fold_idx = 1
    for train_idx, test_idx in cv_outer.split(X, y_encoded):
        print(f"Processing outer fold {fold_idx}...")
        X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
        y_train_outer, y_test_outer = y_encoded[train_idx], y_encoded[test_idx]

        if tsne_selected:
            X_train_outer_fold = X_transformed_final.iloc[train_idx].reset_index(drop=True)
            X_test_outer_fold = X_transformed_final.iloc[test_idx].reset_index(drop=True)
        else:
            X_train_outer_fold = X_train_outer
            X_test_outer_fold = X_test_outer

        cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

        if not tsne_selected:
            def objective_inner(trial):
                steps = [('scaler', StandardScaler())]

                if feature_selection_method != 'none':
                    if feature_selection_method == 'pca':
                        max_components = min(X_train_outer_fold.shape[1], X_train_outer_fold.shape[0]-1)
                        pca_n_components = trial.suggest_int('pca_n_components', 1, max_components)
                        steps.append(('feature_selection', PCA(n_components=pca_n_components, random_state=1234)))
                    elif feature_selection_method == 'kpca':
                        max_components = min(X_train_outer_fold.shape[1], X_train_outer_fold.shape[0]-1)
                        kpca_kernel = trial.suggest_categorical('kpca_kernel', ['linear', 'poly', 'rbf', 'sigmoid', 'cosine'])
                        kpca_n_components = trial.suggest_int('kpca_n_components', 1, max_components)
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
                        steps.append(('feature_selection', safe_umap(
                            n_components=umap_n_components,
                            n_neighbors=umap_n_neighbors,
                            min_dist=umap_min_dist,
                            X=X_train_outer_fold
                        )))
                    elif feature_selection_method == 'pls':
                        pls_max_components = min(X_train_outer_fold.shape[0], X_train_outer_fold.shape[1]) - 1
                        pls_max_components = max(1, pls_max_components)
                        pls_n_components = trial.suggest_int('pls_n_components', 1, pls_max_components)
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
                            max_iter=100000,
                            tol=1e-4
                        )))

                n_estimators = trial.suggest_int('n_estimators', 10, 300)
                max_depth = trial.suggest_int('max_depth', 3, 15)
                learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 0.1)
                reg_alpha = trial.suggest_loguniform('reg_alpha', 1e-8, 10.0)
                reg_lambda = trial.suggest_loguniform('reg_lambda', 1e-8, 10.0)
                gamma = trial.suggest_loguniform('gamma', 1e-8, 10.0)

                steps.append(('xgb', XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                    gamma=gamma,
                    eval_metric='mlogloss',
                    use_label_encoder=False,
                    verbosity=0,
                    random_state=1234
                )))

                pipeline = Pipeline(steps)

                with SuppressOutput():
                    f1_scores = []
                    for inner_train_idx, inner_valid_idx in cv_inner.split(X_train_outer_fold, y_train_outer):
                        X_train_inner, X_valid_inner = X_train_outer_fold.iloc[inner_train_idx], X_train_outer_fold.iloc[inner_valid_idx]
                        y_train_inner, y_valid_inner = y_train_outer[inner_train_idx], y_train_outer[inner_valid_idx]
                        try:
                            classes = np.unique(y_train_inner)
                            class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_inner)
                            cw_dict = dict(zip(classes, class_weights))
                            sample_weight_inner = np.array([cw_dict[label] for label in y_train_inner])
                            
                            pipeline.fit(X_train_inner, y_train_inner, xgb__sample_weight=sample_weight_inner)
                            y_pred_inner = pipeline.predict(X_valid_inner)
                            f1_val = f1_score(y_valid_inner, y_pred_inner, average='weighted')
                            f1_scores.append(f1_val)
                        except (ValueError, ArpackError, NotImplementedError):
                            f1_scores.append(0.0)
                    return np.mean(f1_scores)

            study_inner = optuna.create_study(direction='maximize', sampler=TPESampler(seed=1234))
            study_inner.optimize(objective_inner, n_trials=50, show_progress_bar=False)

            best_params_inner = study_inner.best_params
            steps = [('scaler', StandardScaler())]

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
                    steps.append(('feature_selection', safe_umap(
                        n_components=best_umap_n_components,
                        n_neighbors=best_umap_n_neighbors,
                        min_dist=best_umap_min_dist,
                        X=X_train_outer_fold
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
                        max_iter=100000,
                        tol=1e-4
                    )))

            best_n_estimators = best_params_inner.get('n_estimators', 100)
            best_max_depth = best_params_inner.get('max_depth', 5)
            best_learning_rate = best_params_inner.get('learning_rate', 0.05)
            best_reg_alpha = best_params_inner.get('reg_alpha', 0.0)
            best_reg_lambda = best_params_inner.get('reg_lambda', 1.0)
            best_gamma = best_params_inner.get('gamma', 0.0)

            steps.append(('xgb', XGBClassifier(
                n_estimators=best_n_estimators,
                max_depth=best_max_depth,
                learning_rate=best_learning_rate,
                reg_alpha=best_reg_alpha,
                reg_lambda=best_reg_lambda,
                gamma=best_gamma,
                eval_metric='mlogloss',
                use_label_encoder=False,
                verbosity=0,
                random_state=1234
            )))

            best_model_inner = Pipeline(steps)

            if tsne_selected:
                best_model_inner = Pipeline([
                    ('xgb', XGBClassifier(
                        n_estimators=best_n_estimators,
                        max_depth=best_max_depth,
                        learning_rate=best_learning_rate,
                        reg_alpha=best_reg_alpha,
                        reg_lambda=best_reg_lambda,
                        gamma=best_gamma,
                        eval_metric='mlogloss',
                        use_label_encoder=False,
                        verbosity=0,
                        random_state=1234
                    ))
                ])
                X_train_outer_fold_final = X_transformed_final.iloc[train_idx].reset_index(drop=True)
                X_test_outer_fold_final = X_transformed_final.iloc[test_idx].reset_index(drop=True)
            else:
                X_train_outer_fold_final = X_train_outer_fold
                X_test_outer_fold_final = X_test_outer_fold

            with SuppressOutput():
                try:
                    classes = np.unique(y_train_outer)
                    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_outer)
                    cw_dict = dict(zip(classes, class_weights))
                    sample_weight_outer = np.array([cw_dict[label] for label in y_train_outer])
                    
                    best_model_inner.fit(X_train_outer_fold_final, y_train_outer, xgb__sample_weight=sample_weight_outer)

                except (ValueError, ArpackError, NotImplementedError) as e:
                    print(f"Error fitting the model in outer fold {fold_idx}: {e}")
                    outer_f1_scores.append(0)
                    outer_auc_scores.append(0)
                    fold_idx += 1
                    continue

            try:
                y_pred_prob_outer = best_model_inner.predict_proba(X_test_outer_fold_final)
                y_pred_class_outer = best_model_inner.predict(X_test_outer_fold_final)
            except (ValueError, ArpackError, NotImplementedError) as e:
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

    if tsne_selected:
        print("Completed cross-validation with t-SNE transformed data.")
        print("Starting hyperparameter tuning for XGBoost on the entire t-SNE transformed dataset...")

        def objective_full_tsne(trial):
            n_estimators = trial.suggest_int('n_estimators', 10, 300)
            max_depth = trial.suggest_int('max_depth', 3, 15)
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 0.1)
            reg_alpha = trial.suggest_loguniform('reg_alpha', 1e-8, 10.0)
            reg_lambda = trial.suggest_loguniform('reg_lambda', 1e-8, 10.0)
            gamma = trial.suggest_loguniform('gamma', 1e-8, 10.0)

            model = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                gamma=gamma,
                eval_metric='mlogloss',
                use_label_encoder=False,
                verbosity=0,
                random_state=1234
            )

            with SuppressOutput():
                f1_scores = []
                for train_idx_full, valid_idx_full in cv_outer.split(X_transformed_final, y_encoded):
                    X_train_full, X_valid_full = X_transformed_final.iloc[train_idx_full], X_transformed_final.iloc[valid_idx_full]
                    y_train_full, y_valid_full = y_encoded[train_idx_full], y_encoded[valid_idx_full]
                    try:
                        model.fit(X_train_full, y_train_full)
                        y_pred_full = model.predict(X_valid_full)
                        f1_val = f1_score(y_valid_full, y_pred_full, average='weighted')
                        f1_scores.append(f1_val)
                    except (ValueError, ArpackError, NotImplementedError):
                        f1_scores.append(0.0)
                return np.mean(f1_scores)

        study_full_tsne = optuna.create_study(direction='maximize', sampler=TPESampler(seed=1234))
        study_full_tsne.optimize(objective_full_tsne, n_trials=50, show_progress_bar=True)

        best_params_full_tsne = study_full_tsne.best_params
        print(f"Best parameters for XGBoost with t-SNE: {best_params_full_tsne}")

        best_model = XGBClassifier(
            n_estimators=best_params_full_tsne.get('n_estimators', 100),
            max_depth=best_params_full_tsne.get('max_depth', 5),
            learning_rate=best_params_full_tsne.get('learning_rate', 0.05),
            reg_alpha=best_params_full_tsne.get('reg_alpha', 0.0),
            reg_lambda=best_params_full_tsne.get('reg_lambda', 1.0),
            gamma=best_params_full_tsne.get('gamma', 0.0),
            eval_metric='mlogloss',
            use_label_encoder=False,
            verbosity=0,
            random_state=1234
        )

        with SuppressOutput():
            try:
                best_model.fit(X_transformed_final, y_encoded)
            except (ValueError, ArpackError, NotImplementedError) as e:
                print(f"Error fitting the final model with t-SNE: {e}")
                sys.exit(1)

        joblib.dump(best_model, f"{prefix}_xgboost_model.pkl")
        joblib.dump((X_transformed_final, y_encoded, le), f"{prefix}_xgboost_data.pkl")

        print(f"Best parameters for XGBoost with t-SNE: {best_params_full_tsne}")

        X_transformed_df = pd.DataFrame(X_transformed_final, columns=[f"TSNE_Component_{i+1}" for i in range(X_transformed_final.shape[1])])
        X_transformed_df.insert(0, 'SampleID', sample_ids)
        X_transformed_df['Label'] = y
        transformed_csv_path = f"{prefix}_xgboost_transformed_X_tsne.csv"
        X_transformed_df.to_csv(transformed_csv_path, index=False)
        print(f"t-SNE transformed data saved to {transformed_csv_path}")

        variance_csv_path = f"{prefix}_xgboost_variance.csv"
        with open(variance_csv_path, 'w') as f:
            f.write("t-SNE does not provide explained variance information.\n")
        print(f"No variance information available for t-SNE. File created at {variance_csv_path}")

        try:
            y_pred_prob = best_model.predict_proba(X_transformed_final)
            y_pred_class = best_model.predict(X_transformed_final)
        except (ValueError, ArpackError, NotImplementedError) as e:
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
                sensitivity = np.mean(np.divide(np.diag(cm), np.sum(cm, axis=1),
                                                out=np.zeros_like(np.diag(cm), dtype=float),
                                                where=np.sum(cm, axis=1)!=0))
                specificity = np.mean(np.divide(np.diag(cm), np.sum(cm, axis=0),
                                                out=np.zeros_like(np.diag(cm), dtype=float),
                                                where=np.sum(cm, axis=0)!=0))

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix for XGBoost with t-SNE',fontsize=12,fontweight='bold')
        plt.savefig(f"{prefix}_xgboost_confusion_matrix.png", dpi=300)
        plt.close()

        fpr_dict = {}
        tpr_dict = {}
        roc_auc_dict = {}

        if num_classes == 2:
            try:
                fpr_dict[0], tpr_dict[0], _ = roc_curve(y_binarized[:, 1], y_pred_prob[:, 1])
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

            try:
                fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(y_binarized.ravel(), y_pred_prob.ravel())
                roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])
            except ValueError:
                fpr_dict["micro"], tpr_dict["micro"], roc_auc_dict["micro"] = np.array([0, 1]), np.array([0, 1]), 0.0

        roc_data = {
            'fpr': fpr_dict,
            'tpr': tpr_dict,
            'roc_auc': roc_auc_dict
        }
        np.save(f"{prefix}_xgboost_roc_data.npy", roc_data)

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
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=18, labelpad=10)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=18, labelpad=10)
        plt.title('ROC Curves for XGBoost with t-SNE', fontsize=22, fontweight='bold', pad=15)
        plt.legend(loc="lower right", fontsize=14, title_fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{prefix}_xgboost_roc_curve.png', dpi=300)
        plt.close()

        metrics = {'Accuracy': acc, 'F1 Score': f1, 'Sensitivity': sensitivity, 'Specificity': specificity}
        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        ax = metrics_df.plot(kind='bar', x='Metric', y='Value', legend=False)
        plt.title('Performance Metrics for XGBoost with t-SNE',fontsize=14,fontweight='bold')
        plt.ylabel('Value')
        plt.ylim(0, 1.1)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=5)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{prefix}_xgboost_metrics.png', dpi=300)
        plt.close()

        predictions_df = pd.DataFrame({
            'SampleID': sample_ids,
            'Original Label': y,
            'Predicted Label': le.inverse_transform(y_pred_class)
        })
        predictions_df.to_csv(f"{prefix}_xgboost_predictions.csv", index=False)
        print(f"Predictions saved to {prefix}_xgboost_predictions.csv")

    else:
        plt.figure(figsize=(10, 6))
        folds = range(1, cv_outer.get_n_splits() + 1)
        plt.plot(folds, outer_f1_scores, marker='o', linestyle='-', label='F1 Score')
        plt.plot(folds, outer_auc_scores, marker='s', linestyle='-', label='AUC Score')
        
        plt.title('F1 and AUC Scores per Outer Fold', fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('Outer Fold Number', fontsize=18, labelpad=10)
        plt.ylabel('Score (F1 / AUC)', fontsize=18, labelpad=10)
        plt.xticks(folds, fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=14, title_fontsize=16)
        plt.ylim(0, 1.05)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{prefix}_xgboost_nested_cv_f1_auc.png", dpi=300)
        plt.close()

        print("Nested cross-validation completed.")
        print(f"Average F1 Score: {np.mean(outer_f1_scores):.4f} ± {np.std(outer_f1_scores):.4f}")
        print(f"Average AUC: {np.mean(outer_auc_scores):.4f} ± {np.std(outer_auc_scores):.4f}")

        print("Starting hyperparameter tuning on the entire dataset...")

        def objective_full(trial):
            steps = [('scaler', StandardScaler())]

            if feature_selection_method != 'none':
                if feature_selection_method == 'pca':
                    max_components = min(X.shape[1], X.shape[0]-1)
                    pca_n_components = trial.suggest_int('pca_n_components', 1, max_components)
                    steps.append(('feature_selection', PCA(n_components=pca_n_components, random_state=1234)))
                elif feature_selection_method == 'kpca':
                    max_components = min(X.shape[1], X.shape[0]-1)
                    kpca_kernel = trial.suggest_categorical('kpca_kernel', ['linear', 'poly', 'rbf', 'sigmoid', 'cosine'])
                    kpca_n_components = trial.suggest_int('kpca_n_components', 1, max_components)
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
                        X=X_train_outer_fold
                    )))
                elif feature_selection_method == 'pls':
                    pls_max_components = min(X.shape[0], X.shape[1]) - 1
                    pls_max_components = max(1, pls_max_components)
                    pls_n_components = trial.suggest_int('pls_n_components', 1, pls_max_components)
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
                        max_iter=100000,
                        tol=1e-4
                    )))

            n_estimators = trial.suggest_int('n_estimators', 10, 300)
            max_depth = trial.suggest_int('max_depth', 3, 15)
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 0.1)
            reg_alpha = trial.suggest_loguniform('reg_alpha', 1e-8, 10.0)
            reg_lambda = trial.suggest_loguniform('reg_lambda', 1e-8, 10.0)
            gamma = trial.suggest_loguniform('gamma', 1e-8, 10.0)

            steps.append(('xgb', XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                gamma=gamma,
                eval_metric='mlogloss',
                use_label_encoder=False,
                verbosity=0,
                random_state=1234
            )))

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
                    except (ValueError, ArpackError, NotImplementedError):
                        f1_scores.append(0.0)
                return np.mean(f1_scores)

        study_full = optuna.create_study(direction='maximize', sampler=TPESampler(seed=1234))
        study_full.optimize(objective_full, n_trials=50, show_progress_bar=True)

        best_params_full = study_full.best_params
        print(f"Best parameters for XGBoost: {best_params_full}")

        steps = [('scaler', StandardScaler())]
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
                steps.append(('feature_selection', safe_umap(
                    n_components=best_umap_n_components_full,
                    n_neighbors=best_umap_n_neighbors_full,
                    min_dist=best_umap_min_dist_full,
                    X=X
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
                    max_iter=100000,
                    tol=1e-4
                )))

        best_n_estimators_full = best_params_full.get('n_estimators', 100)
        best_max_depth_full = best_params_full.get('max_depth', 5)
        best_learning_rate_full = best_params_full.get('learning_rate', 0.05)
        best_reg_alpha_full = best_params_full.get('reg_alpha', 0.0)
        best_reg_lambda_full = best_params_full.get('reg_lambda', 1.0)
        best_gamma_full = best_params_full.get('gamma', 0.0)

        steps.append(('xgb', XGBClassifier(
            n_estimators=best_n_estimators_full,
            max_depth=best_max_depth_full,
            learning_rate=best_learning_rate_full,
            reg_alpha=best_reg_alpha_full,
            reg_lambda=best_reg_lambda_full,
            gamma=best_gamma_full,
            eval_metric='mlogloss',
            use_label_encoder=False,
            verbosity=0,
            random_state=1234
        )))

        if tsne_selected:
            best_model = Pipeline([
                ('xgb', XGBClassifier(
                    n_estimators=best_n_estimators_full,
                    max_depth=best_max_depth_full,
                    learning_rate=best_learning_rate_full,
                    reg_alpha=best_reg_alpha_full,
                    reg_lambda=best_reg_lambda_full,
                    gamma=best_gamma_full,
                    eval_metric='mlogloss',
                    use_label_encoder=False,
                    verbosity=0,
                    random_state=1234
                ))
            ])
        else:
            best_model = Pipeline(steps)

        with SuppressOutput():
            try:
                if tsne_selected:
                    best_model.fit(X_transformed_final, y_encoded)
                else:
                    classes = np.unique(y_encoded)
                    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_encoded)
                    cw_dict = dict(zip(classes, class_weights))
                    sample_weight_full = np.array([cw_dict[label] for label in y_encoded])
                    
                    best_model.fit(X, y_encoded, xgb__sample_weight=sample_weight_full)

            except (ValueError, ArpackError, NotImplementedError) as e:
                print(f"Error fitting the final model: {e}")
                sys.exit(1)

        if tsne_selected:
            joblib.dump(best_model, f"{prefix}_xgboost_model.pkl")
            joblib.dump((X_transformed_final, y_encoded, le), f"{prefix}_xgboost_data.pkl")
        else:
            joblib.dump(best_model, f"{prefix}_xgboost_model.pkl")
            joblib.dump((X, y_encoded, le), f"{prefix}_xgboost_data.pkl")

        if tsne_selected:
            print(f"Best parameters for XGBoost with t-SNE: {best_params_full_tsne}")
        else:
            print(f"Best parameters for XGBoost: {best_params_full}")

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
                    transformed_columns = [
                        f"{feature_selection_method.upper()}_Component_{i+1}" for i in range(n_components)
                    ]
                    X_transformed_df = pd.DataFrame(X_transformed, columns=transformed_columns)
                elif feature_selection_method == 'elasticnet':
                    selected_features_mask = best_model.named_steps['feature_selection'].selector.get_support()
                    selected_features = X.columns[selected_features_mask]
                    X_transformed_df = pd.DataFrame(X[selected_features].values, columns=selected_features)
                else:
                    X_transformed_df = pd.DataFrame(X_transformed)

                X_transformed_df.insert(0, 'SampleID', sample_ids)
                X_transformed_df['Label'] = y
                transformed_csv_path = f"{prefix}_xgboost_transformed_X.csv"
                X_transformed_df.to_csv(transformed_csv_path, index=False)
                print(f"Transformed data saved to {transformed_csv_path}")

                variance_csv_path = f"{prefix}_xgboost_variance.csv"
                if feature_selection_method == 'pca':
                    pca_safe_n_components = min(X.shape[0], X.shape[1])
                    full_pca = PCA(n_components=pca_safe_n_components, random_state=1234)
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
                elif feature_selection_method == 'kpca':
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

        if not tsne_selected:
            y_pred_prob = None
            try:
                y_pred_prob = cross_val_predict(best_model, X, y_encoded, cv=cv_outer, method='predict_proba', n_jobs=-1)
                y_pred_class = np.argmax(y_pred_prob, axis=1)
            except (ValueError, ArpackError, NotImplementedError) as e:
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
                    sensitivity = np.mean(np.divide(np.diag(cm), np.sum(cm, axis=1),
                                                    out=np.zeros_like(np.diag(cm), dtype=float),
                                                    where=np.sum(cm, axis=1)!=0))
                    specificity = np.mean(np.divide(np.diag(cm), np.sum(cm, axis=0),
                                                    out=np.zeros_like(np.diag(cm), dtype=float),
                                                    where=np.sum(cm, axis=0)!=0))

            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
            disp.plot(cmap=plt.cm.Blues)
            plt.title('Confusion Matrix for XGBoost',fontsize=12,fontweight='bold')
            plt.savefig(f"{prefix}_xgboost_confusion_matrix.png", dpi=300)
            plt.close()

            fpr_dict = {}
            tpr_dict = {}
            roc_auc_dict = {}

            if num_classes == 2:
                try:
                    if y_binarized.shape[1] < 2 or y_pred_prob.shape[1] < 2:
                        print("Warning: Only one class detected in y_binarized or y_pred_prob. Skipping ROC computation.")
                        fpr_dict[0], tpr_dict[0], roc_auc_dict[0] = np.array([0, 1]), np.array([0, 1]), 0.0
                    else:
                        fpr_dict[0], tpr_dict[0], _ = roc_curve(y_binarized[:, 1], y_pred_prob[:, 1])
                        roc_auc_dict[0] = auc(fpr_dict[0], tpr_dict[0])
                except ValueError:
                    fpr_dict[0], tpr_dict[0], roc_auc_dict[0] = np.array([0, 1]), np.array([0, 1]), 0.0

            else:
                for i in range(y_binarized.shape[1]):
                    try:
                        fpr_dict[i], tpr_dict[i], _ = roc_curve(y_binarized[:, i], y_pred_prob[:, i])
                        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
                    except ValueError:
                        fpr_dict[i], tpr_dict[i], roc_auc_dict[i] = np.array([0, 1]), np.array([0, 1]), 0.0

                try:
                    fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(y_binarized.ravel(), y_pred_prob.ravel())
                    roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])
                except ValueError:
                    fpr_dict["micro"], tpr_dict["micro"], roc_auc_dict["micro"] = (
                        np.array([0, 1]),
                        np.array([0, 1]),
                        0.0
                    )

            roc_data = {
                'fpr': fpr_dict,
                'tpr': tpr_dict,
                'roc_auc': roc_auc_dict
            }
            np.save(f"{prefix}_xgboost_roc_data.npy", roc_data)

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
            plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=18, labelpad=10)
            plt.ylabel('True Positive Rate (Sensitivity)', fontsize=18, labelpad=10)
            plt.title('ROC Curves for XGBoost', fontsize=22, fontweight='bold', pad=15)
            plt.legend(loc="lower right", fontsize=14, title_fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout()
            plt.savefig(f'{prefix}_xgboost_roc_curve.png', dpi=300)
            plt.close()

            metrics = {'Accuracy': acc, 'F1 Score': f1, 'Sensitivity': sensitivity, 'Specificity': specificity}
            metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
            ax = metrics_df.plot(kind='bar', x='Metric', y='Value', legend=False)
            plt.title('Performance Metrics for XGBoost',fontsize=14,fontweight='bold')
            plt.ylabel('Value')
            plt.ylim(0, 1)
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f'{prefix}_xgboost_metrics.png', dpi=300)
            plt.close()

            predictions_df = pd.DataFrame({
                'SampleID': sample_ids,
                'Original Label': y,
                'Predicted Label': le.inverse_transform(y_pred_class)
            })
            predictions_df.to_csv(f"{prefix}_xgboost_predictions.csv", index=False)
            print(f"Predictions saved to {prefix}_xgboost_predictions.csv")

    if tsne_selected:
        pass
    else:
        pass

def main():
    parser = argparse.ArgumentParser(description='Run XGBoost with Nested Cross-Validation, Feature Selection (ElasticNet, PCA, KPCA, UMAP, t-SNE, PLS), and Optuna hyperparameter optimization.')
    parser.add_argument('-i', type=str, help='Input file in CSV format', required=True)
    parser.add_argument('-p', type=str, help='Prefix for output files', required=True)
    parser.add_argument(
        '-f',
        type=str,
        choices=['none', 'elasticnet', 'pca', 'kpca', 'umap', 'tsne', 'pls'],
        default='none',
        help='Feature selection method to use. Options: none, elasticnet, pca, kpca, umap, tsne, pls.'
    )

    args = parser.parse_args()

    if args.f == 'tsne':
        print("Warning: t-SNE does not support transforming new data. The entire dataset will be transformed before cross-validation, which may lead to data leakage.")

    xgboost_nested_cv(args.i, args.p, args.f)

if __name__ == '__main__':
    main()
