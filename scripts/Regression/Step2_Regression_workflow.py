import argparse
import pandas as pd
import numpy as np
import sys
import os
import json
import contextlib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.samplers import TPESampler
from joblib import Parallel, delayed, dump
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
import umap
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import BaseEstimator, TransformerMixin
import random
import matplotlib as mpl

# ----- Reproducibility & larger fonts (safe, no overlap) -----
def _set_global_seed(seed: int = 1234):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

_set_global_seed(1234)

mpl.rcParams['font.size'] = 16
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 14
# -------------------------------------------------------------

# Import custom feature selectors
from feature_selectors import PLSFeatureSelector, ElasticNetFeatureSelector
# Import custom wrappers from a separate module
from wrappers import UMAPPipelineWrapper, TSNEPipelineWrapper

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)

class SuppressOutput(contextlib.AbstractContextManager):
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._stdout
        sys.stderr = self._stderr

def get_feature_selection_transformer(method, trial, X_train, y_train, is_pls=False):
    import optuna

    if method == 'pca':
        pca_n_components = trial.suggest_int('pca_n_components', 1, min(X_train.shape[1], X_train.shape[0] - 1))
        return PCA(n_components=pca_n_components, random_state=1234)

    elif method == 'kpca':
        kpca_kernel = trial.suggest_categorical('kpca_kernel', ['linear', 'poly', 'rbf', 'sigmoid', 'cosine'])
        kpca_n_components = trial.suggest_int('kpca_n_components', 1, min(X_train.shape[1], X_train.shape[0] - 1))
        kpca_params = {
            'n_components': kpca_n_components,
            'kernel': kpca_kernel,
            'random_state': 1234,
            'eigen_solver': 'arpack',
            'max_iter': 5000,
            'alpha': 1e-7
        }
        if kpca_kernel in ['poly', 'rbf', 'sigmoid']:
            kpca_gamma = trial.suggest_float('kpca_gamma', 1e-4, 1e1, log=True)
            kpca_params['gamma'] = kpca_gamma
        if kpca_kernel in ['poly', 'sigmoid']:
            kpca_coef0 = trial.suggest_float('kpca_coef0', 0.0, 1.0)
            kpca_params['coef0'] = kpca_coef0
        if kpca_kernel == 'poly':
            kpca_degree = trial.suggest_int('kpca_degree', 2, 5)
            kpca_params['degree'] = kpca_degree
        return KernelPCA(**kpca_params)

    elif method == 'umap':
        umap_n_components = trial.suggest_int('umap_n_components', 2, min(100, X_train.shape[1]))
        umap_n_neighbors = trial.suggest_int('umap_n_neighbors', 5, min(50, X_train.shape[0] - 1))
        umap_min_dist = trial.suggest_float('umap_min_dist', 0.0, 0.99)
        return UMAPPipelineWrapper(
            n_components=umap_n_components,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            random_state=1234
        )

    elif method == 'tsne':
        return TSNEPipelineWrapper(
            n_components=2,
            perplexity=30.0,
            learning_rate=200.0,
            max_iter=1000,
            random_state=1234
        )

    elif method == 'elasticnet':
        elasticnet_alpha = trial.suggest_float('elasticnet_alpha', 1e-4, 1e1, log=True)
        l1_ratio = trial.suggest_float('elasticnet_l1_ratio', 0.0, 1.0)
        return ElasticNetFeatureSelector(
            alpha=elasticnet_alpha,
            l1_ratio=l1_ratio,
            max_iter=10000,
            tol=1e-4
        )

    elif method == 'pls':
        pls_n_components = trial.suggest_int('pls_n_components', 2, min(X_train.shape[1], X_train.shape[0] - 1))
        return PLSFeatureSelector(n_components=pls_n_components)

    else:
        return None

def get_param_distributions(name, num_features):
    import optuna

    if name == 'Neural_Network_reg':
        return {
            'n_layers': optuna.distributions.IntDistribution(low=1, high=15),
            'hidden_layer_size': optuna.distributions.IntDistribution(low=10, high=200),
            'alpha': optuna.distributions.FloatDistribution(low=1e-4, high=1e-1, log=True),
            'learning_rate_init': optuna.distributions.FloatDistribution(low=1e-4, high=1e-1, log=True)
        }
    elif name == 'Random_Forest_reg':
        return {
            'n_estimators': optuna.distributions.IntDistribution(low=100, high=1000),
            'max_depth': optuna.distributions.IntDistribution(low=5, high=50),
            'max_features': optuna.distributions.CategoricalDistribution(choices=['sqrt', 'log2', None]),
            'min_samples_split': optuna.distributions.IntDistribution(low=2, high=20),
            'min_samples_leaf': optuna.distributions.IntDistribution(low=1, high=20)
        }
    elif name == 'SVM_reg':
        return {
            'C': optuna.distributions.FloatDistribution(low=1e-3, high=1e3, log=True),
            'gamma': optuna.distributions.FloatDistribution(low=1e-4, high=1e1, log=True),
            'epsilon': optuna.distributions.FloatDistribution(low=1e-4, high=1e1, log=True)
        }
    elif name == 'XGBoost_reg':
        return {
            'n_estimators': optuna.distributions.IntDistribution(low=50, high=500),
            'max_depth': optuna.distributions.IntDistribution(low=3, high=15),
            'learning_rate': optuna.distributions.FloatDistribution(low=1e-3, high=1e-1, log=True),
            'subsample': optuna.distributions.FloatDistribution(low=0.5, high=1.0),
            'colsample_bytree': optuna.distributions.FloatDistribution(low=0.5, high=1.0),
            'reg_alpha': optuna.distributions.FloatDistribution(low=0.0, high=1.0),
            'reg_lambda': optuna.distributions.FloatDistribution(low=0.0, high=1.0)
        }
    elif name == 'PLS_reg':
        upper = max(2, num_features // 2)
        return {
            'n_components': optuna.distributions.IntDistribution(low=2, high=upper)
        }
    elif name == 'KNN_reg':
        return {
            'n_neighbors': optuna.distributions.IntDistribution(low=1, high=50),
            'weights': optuna.distributions.CategoricalDistribution(choices=['uniform', 'distance']),
            'p': optuna.distributions.IntDistribution(low=1, high=3)
        }
    elif name == 'LightGBM_reg':
        return {
            'n_estimators': optuna.distributions.IntDistribution(low=50, high=500),
            'max_depth': optuna.distributions.CategoricalDistribution(choices=[-1, 3, 5, 7, 10, 15, 20, 30, 50]),
            'learning_rate': optuna.distributions.FloatDistribution(low=1e-3, high=1e-1, log=True),
            'num_leaves': optuna.distributions.IntDistribution(low=20, high=150),
            'subsample': optuna.distributions.FloatDistribution(low=0.5, high=1.0),
            'colsample_bytree': optuna.distributions.FloatDistribution(low=0.5, high=1.0),
            'reg_alpha': optuna.distributions.FloatDistribution(low=0.0, high=1.0),
            'reg_lambda': optuna.distributions.FloatDistribution(low=0.0, high=1.0)
        }
    else:
        return {}

def build_final_model(name, params):
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.neighbors import KNeighborsRegressor
    from xgboost import XGBRegressor
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import RandomForestRegressor
    from lightgbm import LGBMRegressor

    if name == 'Neural_Network_reg':
        hidden_layer_sizes = tuple([params.get('hidden_layer_size', 100)] * params.get('n_layers', 1))
        alpha = params.get('alpha', 1e-4)
        learning_rate_init = params.get('learning_rate_init', 1e-3)
        return MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=200000,
            random_state=1234
        )
    elif name == 'Random_Forest_reg':
        return RandomForestRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 10),
            max_features=params.get('max_features', 'sqrt'),
            min_samples_split=params.get('min_samples_split', 2),
            min_samples_leaf=params.get('min_samples_leaf', 1),
            random_state=1234,
            n_jobs=-1
        )
    elif name == 'SVM_reg':
        return SVR(
            C=params.get('C', 1.0),
            gamma=params.get('gamma', 'scale'),
            epsilon=params.get('epsilon', 0.1)
        )
    elif name == 'XGBoost_reg':
        return XGBRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 5),
            learning_rate=params.get('learning_rate', 0.05),
            subsample=params.get('subsample', 1.0),
            colsample_bytree=params.get('colsample_bytree', 1.0),
            reg_alpha=params.get('reg_alpha', 0.0),
            reg_lambda=params.get('reg_lambda', 1.0),
            eval_metric='rmse',
            random_state=1234,
            verbosity=0,
            n_jobs=-1
        )
    elif name == 'PLS_reg':
        return PLSRegression(
            n_components=params.get('n_components', 2)
        )
    elif name == 'KNN_reg':
        return KNeighborsRegressor(
            n_neighbors=params.get('n_neighbors', 5),
            weights=params.get('weights', 'uniform'),
            p=params.get('p', 2),
            n_jobs=-1
        )
    elif name == 'LightGBM_reg':
        return LGBMRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', -1),
            learning_rate=params.get('learning_rate', 0.05),
            num_leaves=params.get('num_leaves', 31),
            subsample=params.get('subsample', 1.0),
            colsample_bytree=params.get('colsample_bytree', 1.0),
            reg_alpha=params.get('reg_alpha', 0.0),
            reg_lambda=params.get('reg_lambda', 0.0),
            random_state=1234,
            force_col_wise=True,
            verbosity=-1,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unsupported model: {name}")

class DummyTrial:
    def __init__(self, params):
        self.params = params
    def suggest_int(self, name, low, high):
        return self.params.get(name, low)
    def suggest_float(self, name, low, high, log=False):
        return self.params.get(name, low)
    def suggest_categorical(self, name, choices):
        return self.params.get(name, choices[0])

def _build_pipeline_for_trial(name, trial, X_train, y_train, feature_selection_method, RANDOM_SEED):
    steps = []
    steps.append(('scaler', StandardScaler()))
    apply_feature_selection = (feature_selection_method != 'none')
    if name == 'PLS_reg' and feature_selection_method not in ['elasticnet', 'umap', 'pca', 'kpca', 'pls', 'tsne']:
        apply_feature_selection = False
    if apply_feature_selection:
        transformer = get_feature_selection_transformer(
            feature_selection_method,
            trial,
            X_train,
            y_train,
            is_pls=(name == 'PLS_reg')
        )
        if transformer is not None:
            steps.append(('feature_selection', transformer))

    # Model per trial
    if name == 'Neural_Network_reg':
        n_layers = trial.suggest_int('n_layers', 1, 15)
        hidden_layer_size = trial.suggest_int('hidden_layer_size', 10, 200)
        alpha = trial.suggest_float('alpha', 1e-4, 1e-1, log=True)
        learning_rate_init = trial.suggest_float('learning_rate_init', 1e-4, 1e-1, log=True)
        model = MLPRegressor(
            hidden_layer_sizes=tuple([hidden_layer_size] * n_layers),
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=200000,
            random_state=RANDOM_SEED
        )
    elif name == 'Random_Forest_reg':
        n_estimators = trial.suggest_int('n_estimators', 100, 1000)
        max_depth = trial.suggest_int('max_depth', 5, 50)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
    elif name == 'SVM_reg':
        C = trial.suggest_float('C', 1e-3, 1e3, log=True)
        gamma = trial.suggest_float('gamma', 1e-4, 1e1, log=True)
        epsilon = trial.suggest_float('epsilon', 1e-4, 1e1, log=True)
        model = SVR(C=C, gamma=gamma, epsilon=epsilon)
    elif name == 'XGBoost_reg':
        n_estimators = trial.suggest_int('n_estimators', 50, 500)
        max_depth = trial.suggest_int('max_depth', 3, 15)
        learning_rate = trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True)
        subsample = trial.suggest_float('subsample', 0.5, 1.0)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
        reg_alpha = trial.suggest_float('reg_alpha', 0.0, 1.0)
        reg_lambda = trial.suggest_float('reg_lambda', 0.0, 1.0)
        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            eval_metric='rmse',
            random_state=RANDOM_SEED,
            verbosity=0,
            n_jobs=-1
        )
    elif name == 'PLS_reg':
        n_components = trial.suggest_int('n_components', 2, min(X_train.shape[1], X_train.shape[0]-1))
        model = PLSRegression(n_components=n_components)
    elif name == 'KNN_reg':
        n_neighbors = trial.suggest_int('n_neighbors', 1, 50)
        weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
        p = trial.suggest_int('p', 1, 3)
        model = KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights=weights,
            p=p,
            n_jobs=-1
        )
    elif name == 'LightGBM_reg':
        n_estimators = trial.suggest_int('n_estimators', 50, 500)
        max_depth = trial.suggest_categorical('max_depth', [-1, 3, 5, 7, 10, 15, 20, 30, 50])
        learning_rate = trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True)
        num_leaves = trial.suggest_int('num_leaves', 20, 150)
        subsample = trial.suggest_float('subsample', 0.5, 1.0)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
        reg_alpha = trial.suggest_float('reg_alpha', 0.0, 1.0)
        reg_lambda = trial.suggest_float('reg_lambda', 0.0, 1.0)
        model = LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=RANDOM_SEED,
            force_col_wise=True,
            verbosity=-1,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unsupported model: {name}")

    steps.append(('regressor', model))
    pipeline = Pipeline(steps)
    return pipeline

def run_model(name, X, y, sample_ids, prefix, feature_selection_method):
    import optuna
    RANDOM_SEED = 1234
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    cv_outer = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    regressors = {
        'Neural_Network_reg': MLPRegressor(max_iter=200000, random_state=RANDOM_SEED),
        'Random_Forest_reg': RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1),
        'SVM_reg': SVR(),
        'XGBoost_reg': XGBRegressor(eval_metric='rmse', random_state=RANDOM_SEED, verbosity=0, n_jobs=-1),
        'PLS_reg': PLSRegression(),
        'KNN_reg': KNeighborsRegressor(n_jobs=-1),
        'LightGBM_reg': LGBMRegressor(random_state=RANDOM_SEED, force_col_wise=True, verbosity=-1, n_jobs=-1)
    }
    if name not in regressors:
        print(f"Error: Model {name} not recognized.")
        return

    apply_feature_selection = (feature_selection_method != 'none')
    if name == 'PLS_reg' and feature_selection_method not in ['elasticnet', 'umap', 'pca', 'kpca', 'pls', 'tsne']:
        apply_feature_selection = False
        print(f"Skipping feature selection for {name} as it is incompatible with '{feature_selection_method}'.")

    param_distributions = get_param_distributions(name, X.shape[1])

    outer_metrics = []
    y_preds = np.zeros_like(y, dtype=float)

    fold_idx = 1
    for train_idx, test_idx in cv_outer.split(X, y):
        print(f"Model: {name}, Fold: {fold_idx}")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        sampler = TPESampler(seed=RANDOM_SEED)
        study = optuna.create_study(direction='minimize', sampler=sampler)

        def objective(trial):
            pipeline = _build_pipeline_for_trial(
                name=name,
                trial=trial,
                X_train=X_train,
                y_train=y_train,
                feature_selection_method=feature_selection_method,
                RANDOM_SEED=RANDOM_SEED
            )
            with SuppressOutput():
                cv_inner = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
                mse_scores = []
                for in_train_idx, in_valid_idx in cv_inner.split(X_train, y_train):
                    X_in_train, X_in_valid = X_train.iloc[in_train_idx], X_train.iloc[in_valid_idx]
                    y_in_train, y_in_valid = y_train[in_train_idx], y_train[in_valid_idx]
                    try:
                        pipeline.fit(X_in_train, y_in_train)
                        preds_val = pipeline.predict(X_in_valid)
                        if np.any(np.isnan(preds_val)):
                            mse_scores.append(float('inf'))
                        else:
                            mse_val = mean_squared_error(y_in_valid, preds_val)
                            mse_scores.append(mse_val)
                    except Exception:
                        mse_scores.append(float('inf'))
                return np.mean(mse_scores)

        try:
            study.optimize(objective, n_trials=50, n_jobs=1, show_progress_bar=False)
            best_params = study.best_params
        except Exception as e:
            print(f"Optuna optimization failed on fold {fold_idx} for {name}: {e}. Using default parameters.")
            best_params = {}

        # Final pipeline for this fold (scaler + optional feature_selection + regressor)
        steps_final = []
        steps_final.append(('scaler', StandardScaler()))
        if apply_feature_selection and study.best_trial is not None:
            transformer = get_feature_selection_transformer(
                feature_selection_method,
                study.best_trial,
                X_train,
                y_train,
                is_pls=(name == 'PLS_reg')
            )
            if transformer is not None:
                steps_final.append(('feature_selection', transformer))

        final_model = build_final_model(name, best_params)
        steps_final.append(('regressor', final_model))
        final_pipeline = Pipeline(steps_final)

        with SuppressOutput():
            try:
                final_pipeline.fit(X_train, y_train)
            except Exception as e:
                print(f"Error fitting the model in fold {fold_idx}: {e}")
                outer_metrics.append({
                    'Fold': fold_idx,
                    'MSE': float('inf'),
                    'RMSE': float('inf'),
                    'MAE': float('inf'),
                    'R2': float('-inf')
                })
                fold_idx += 1
                continue

        try:
            y_pred = final_pipeline.predict(X_test)
            if np.any(np.isnan(y_pred)):
                print(f"Error predicting in fold {fold_idx}: TSNE or other transforms produced NaN.")
                y_pred = np.full(len(X_test), np.nan)
        except Exception as e:
            print(f"Error predicting in fold {fold_idx}: {e}")
            y_pred = np.full(len(X_test), np.nan)

        y_preds[test_idx] = y_pred

        if np.any(np.isnan(y_pred)):
            mse_fold = float('inf')
            rmse_fold = float('inf')
            mae_fold = float('inf')
            r2_fold = float('-inf')
        else:
            mse_fold = mean_squared_error(y_test, y_pred)
            rmse_fold = np.sqrt(mse_fold)
            mae_fold = mean_absolute_error(y_test, y_pred)
            r2_fold = r2_score(y_test, y_pred)

        outer_metrics.append({
            'Fold': fold_idx,
            'MSE': mse_fold,
            'RMSE': rmse_fold,
            'MAE': mae_fold,
            'R2': r2_fold
        })

        model_params = {}
        for step_name, step_obj in final_pipeline.named_steps.items():
            if hasattr(step_obj, 'get_params'):
                model_params[step_name] = step_obj.get_params()
        with open(f"{prefix}_{name}_fold{fold_idx}_model_params.json", 'w') as f:
            json.dump(model_params, f, indent=4)

        fold_predictions = pd.DataFrame({
            'SampleID': sample_ids.iloc[test_idx],
            'Original Label': y_test,
            f'{name}_Predicted': y_pred
        })
        fold_predictions.to_csv(f"{prefix}_{name}_fold{fold_idx}_predictions.csv", index=False)

        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Actual vs Predicted for {name} - Fold {fold_idx}')
        plt.tight_layout()
        plt.savefig(f"{prefix}_{name}_fold{fold_idx}_prediction.png", dpi=300)
        plt.close()

        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True, bins=30)
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title(f'Residuals for {name} - Fold {fold_idx}')
        plt.tight_layout()
        plt.savefig(f"{prefix}_{name}_fold{fold_idx}_residuals.png", dpi=300)
        plt.close()

        fold_idx += 1

    metrics_df = pd.DataFrame(outer_metrics)
    metrics_df.to_csv(f"{prefix}_{name}_model_performance_metrics.csv", index=False)
    print(f"Model performance metrics saved to {prefix}_{name}_model_performance_metrics.csv")

    np.save(f"{prefix}_{name}_X.npy", X.values)
    np.save(f"{prefix}_{name}_y_true.npy", y)
    np.save(f"{prefix}_{name}_y_pred.npy", y_preds)
    np.save(f"{prefix}_{name}_feature_names.npy", X.columns.values)

    overall_residuals = y - y_preds

    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df['Fold'], metrics_df['MSE'], marker='o', label='MSE')
    plt.plot(metrics_df['Fold'], metrics_df['RMSE'], marker='o', label='RMSE')
    plt.plot(metrics_df['Fold'], metrics_df['MAE'], marker='o', label='MAE')
    plt.plot(metrics_df['Fold'], metrics_df['R2'], marker='o', label='R2')
    plt.xlabel('Fold')
    plt.ylabel('Metric Value')
    plt.title(f'{name} Performance Metrics Across Folds')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_{name}_metrics_line_plot.png", dpi=300)
    plt.close()

    predictions_df = pd.DataFrame({
        'SampleID': sample_ids,
        'Original Label': y,
        f'{name}_Predicted': y_preds
    })
    predictions_df.to_csv(f"{prefix}_{name}_predictions.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_preds, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Overall Actual vs Predicted for {name}')
    plt.tight_layout()
    plt.savefig(f"{prefix}_{name}_overall_prediction.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(overall_residuals, kde=True, bins=30)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title(f'Overall Residuals for {name}')
    plt.tight_layout()
    plt.savefig(f"{prefix}_{name}_overall_residuals.png", dpi=300)
    plt.close()

    # ===== Post-evaluation: full-data CV to select final hyperparameters and fit final model =====
    print(f"Running full-data CV hyperparameter search for final {name} model...")
    sampler_full = TPESampler(seed=RANDOM_SEED)
    study_full = optuna.create_study(direction='minimize', sampler=sampler_full)

    def objective_full(trial):
        pipeline = _build_pipeline_for_trial(
            name=name,
            trial=trial,
            X_train=X,
            y_train=y,
            feature_selection_method=feature_selection_method,
            RANDOM_SEED=RANDOM_SEED
        )
        with SuppressOutput():
            cv_inner = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
            mse_scores = []
            for in_train_idx, in_valid_idx in cv_inner.split(X, y):
                X_in_train, X_in_valid = X.iloc[in_train_idx], X.iloc[in_valid_idx]
                y_in_train, y_in_valid = y[in_train_idx], y[in_valid_idx]
                try:
                    pipeline.fit(X_in_train, y_in_train)
                    preds_val = pipeline.predict(X_in_valid)
                    if np.any(np.isnan(preds_val)):
                        mse_scores.append(float('inf'))
                    else:
                        mse_val = mean_squared_error(y_in_valid, preds_val)
                        mse_scores.append(mse_val)
                except Exception:
                    mse_scores.append(float('inf'))
            return np.mean(mse_scores)

    try:
        study_full.optimize(objective_full, n_trials=50, n_jobs=1, show_progress_bar=False)
        final_best_params = study_full.best_params
    except Exception as e:
        print(f"Optuna final optimization failed for {name}: {e}. Using default parameters.")
        final_best_params = {}

    steps_final_full = []
    steps_final_full.append(('scaler', StandardScaler()))
    # feature selection transformer built with final_best_params through DummyTrial to keep consistency
    if apply_feature_selection and len(final_best_params) > 0:
        dummy_trial = DummyTrial(final_best_params)
        transformer_final = get_feature_selection_transformer(
            feature_selection_method,
            dummy_trial,
            X,
            y,
            is_pls=(name == 'PLS_reg')
        )
        # For t-SNE we still avoid adding it to the final predictive pipeline
        if feature_selection_method == 'tsne':
            transformer_final = None
        if transformer_final is not None:
            try:
                transformer_final.fit(StandardScaler().fit(X).transform(X), y)  # ensure it can fit, actual fit will be in pipeline
            except Exception:
                pass
            steps_final_full.append(('feature_selection', transformer_final))

    final_model_full = build_final_model(name, final_best_params)
    steps_final_full.append(('regressor', final_model_full))
    final_pipeline_full = Pipeline(steps_final_full)

    with SuppressOutput():
        try:
            final_pipeline_full.fit(X, y)
        except Exception as e:
            print(f"Error fitting the final full-data model for {name}: {e}")

    model_pkl_path = f"{prefix}_{name}_best_model.pkl"
    dump(final_pipeline_full, model_pkl_path)
    print(f"Final best model pipeline saved to: {model_pkl_path}")

    # ===== Consistent exports based on the fitted final pipeline (except t-SNE kept as-is) =====
    if apply_feature_selection and 'feature_selection' in final_pipeline_full.named_steps:
        fs = final_pipeline_full.named_steps['feature_selection']
        scaler_fitted = final_pipeline_full.named_steps.get('scaler', None)
        try:
            if scaler_fitted is not None:
                X_scaled_full = scaler_fitted.transform(X)
            else:
                X_scaled_full = X.values
        except Exception:
            X_scaled_full = X.values

        try:
            X_fs = fs.transform(X_scaled_full)
        except Exception as e:
            print(f"Error transforming data with final feature selection: {e}")
            X_fs = None

        if X_fs is not None:
            if feature_selection_method == 'elasticnet':
                if hasattr(fs.selector, 'get_support'):
                    selected_mask = fs.selector.get_support()
                    selected_features = X.columns[selected_mask]
                    X_fs_df = pd.DataFrame(X_fs, columns=selected_features)
                    X_fs_df.insert(0, 'SampleID', sample_ids)
                    X_fs_df['Label'] = y
                    enet_csv_path = f"{prefix}_{name}_elasticnet_selected_features_X.csv"
                    X_fs_df.to_csv(enet_csv_path, index=False)
                    print(f"ElasticNet selected features file saved to {enet_csv_path}")

            elif feature_selection_method == 'pca':
                comp_count = getattr(fs, 'n_components', None)
                if comp_count is None and hasattr(fs, 'n_components_'):
                    comp_count = fs.n_components_
                comp_cols = [f"PCA_Component_{i+1}" for i in range(comp_count)]
                X_pca_df = pd.DataFrame(X_fs, columns=comp_cols)
                X_pca_df.insert(0, 'SampleID', sample_ids)
                X_pca_df['Label'] = y
                pca_csv_path = f"{prefix}_{name}_best_pca_transformed_X.csv"
                X_pca_df.to_csv(pca_csv_path, index=False)
                print(f"Best PCA transformed data saved to {pca_csv_path}")

                if hasattr(fs, 'explained_variance_ratio_'):
                    var_csv_path = f"{prefix}_{name}_pca_explained_variance.csv"
                    evr = fs.explained_variance_ratio_
                    evr_df = pd.DataFrame({
                        'Component': np.arange(1, len(evr) + 1),
                        'Explained Variance Ratio': evr
                    })
                    evr_df.to_csv(var_csv_path, index=False)
                    print(f"PCA explained variance ratios saved to {var_csv_path}")

            elif feature_selection_method == 'kpca':
                comp_count = getattr(fs, 'n_components', None)
                if comp_count is None and hasattr(fs, 'n_components_'):
                    comp_count = fs.n_components_
                comp_cols = [f"KPCA_Component_{i+1}" for i in range(comp_count)]
                X_kpca_df = pd.DataFrame(X_fs, columns=comp_cols)
                X_kpca_df.insert(0, 'SampleID', sample_ids)
                X_kpca_df['Label'] = y
                kpca_csv_path = f"{prefix}_{name}_best_kpca_transformed_X.csv"
                X_kpca_df.to_csv(kpca_csv_path, index=False)
                print(f"Best KPCA transformed data saved to {kpca_csv_path}")

                var_csv_path = f"{prefix}_{name}_kpca_explained_variance.csv"
                with open(var_csv_path, 'w') as f:
                    f.write("KernelPCA does not provide explained variance info.\n")
                print(f"No variance info for KPCA. File created at {var_csv_path}")

            elif feature_selection_method == 'pls':
                comp_count = getattr(fs, 'n_components', None)
                if comp_count is None:
                    comp_count = getattr(fs, 'n_components_', None)
                if comp_count is None:
                    comp_count = getattr(getattr(fs, 'pls', None), 'n_components', None)
                if comp_count is None:
                    comp_count = 2
                comp_cols = [f"PLS_Component_{i+1}" for i in range(comp_count)]
                X_pls_df = pd.DataFrame(X_fs, columns=comp_cols)
                X_pls_df.insert(0, 'SampleID', sample_ids)
                X_pls_df['Label'] = y
                pls_csv_path = f"{prefix}_{name}_best_pls_transformed_X.csv"
                X_pls_df.to_csv(pls_csv_path, index=False)
                print(f"Best PLS transformed data saved to {pls_csv_path}")

                var_csv_path = f"{prefix}_{name}_pls_explained_variance.csv"
                try:
                    if hasattr(fs, 'pls') and hasattr(fs.pls, 'x_scores_'):
                        x_scores = fs.pls.x_scores_
                        explained_var = np.var(x_scores, axis=0) / np.var(X_scaled_full, axis=0).sum()
                        var_df = pd.DataFrame({
                            'Component': np.arange(1, len(explained_var) + 1),
                            'Explained Variance Ratio': explained_var
                        })
                        var_df.to_csv(var_csv_path, index=False)
                        print(f"PLS explained variance ratios saved to {var_csv_path}")
                except Exception:
                    pass

            elif feature_selection_method == 'umap':
                if hasattr(fs, '_umap'):
                    comp_count = fs.n_components_
                else:
                    comp_count = getattr(fs, 'n_components', None)
                    if comp_count is None:
                        comp_count = X_fs.shape[1]
                comp_cols = [f"UMAP_Component_{i+1}" for i in range(comp_count)]
                X_umap_df = pd.DataFrame(X_fs, columns=comp_cols)
                X_umap_df.insert(0, 'SampleID', sample_ids)
                X_umap_df['Label'] = y
                umap_csv_path = f"{prefix}_{name}_best_umap_transformed_X.csv"
                X_umap_df.to_csv(umap_csv_path, index=False)
                print(f"Best UMAP transformed data saved to {umap_csv_path}")

                var_csv_path = f"{prefix}_{name}_umap_variance.csv"
                with open(var_csv_path, 'w') as f:
                    f.write("UMAP does not provide explained variance info.\n")
                print(f"No variance info for UMAP. File created at {var_csv_path}")

    # Keep t-SNE export as-is (not part of predictive pipeline)
    if feature_selection_method == 'tsne':
        tsne_csv_path = f"{prefix}_{name}_tsne_transformed_X.csv"
        tsne_transformer = TSNE(
            n_components=2,
            perplexity=30.0,
            learning_rate=200.0,
            max_iter=1000,
            random_state=1234
        )
        X_tsne = tsne_transformer.fit_transform(X)
        tsne_cols = [f"TSNE_Component_{i+1}" for i in range(2)]
        X_tsne_df = pd.DataFrame(X_tsne, columns=tsne_cols)
        X_tsne_df.insert(0, 'SampleID', sample_ids)
        X_tsne_df['Label'] = y
        X_tsne_df.to_csv(tsne_csv_path, index=False)
        print(f"t-SNE transformed data saved to {tsne_csv_path}")

def main_regression():
    parser = argparse.ArgumentParser(description='Run regression models with Nested Cross-Validation, Feature Selection, and Optuna hyperparameter optimization.')
    parser.add_argument('-i', '--csv', type=str, required=True, help='Input CSV file')
    parser.add_argument('-p', '--prefix', type=str, required=True, help='Output prefix')
    parser.add_argument('-f', '--feature_selection', type=str,
                        choices=['none', 'elasticnet', 'pca', 'kpca', 'umap', 'tsne', 'pls'],
                        default='none',
                        help='Feature selection method to use')
    parser.add_argument('-m', '--models', type=str, nargs='+',
                        default=['Neural_Network_reg', 'Random_Forest_reg', 'SVM_reg', 'XGBoost_reg',
                                 'PLS_reg', 'KNN_reg', 'LightGBM_reg'],
                        help='List of models to run')

    args = parser.parse_args()

    # Model aliases
    model_aliases = {
        "NN_reg": "Neural_Network_reg",
        "RF_reg": "Random_Forest_reg",
        "SVM_reg": "SVM_reg",
        "XGB_reg": "XGBoost_reg",
        "PLS_reg": "PLS_reg",
        "KNN_reg": "KNN_reg",
        "LGBM_reg": "LightGBM_reg"
    }

    parsed_models = []
    for m in args.models:
        if m in model_aliases:
            parsed_models.append(model_aliases[m])
        else:
            parsed_models.append(m)

    data = pd.read_csv(args.csv).dropna()
    if 'SampleID' not in data.columns or 'Label' not in data.columns:
        raise ValueError("Input data must have 'SampleID' and 'Label' columns.")

    sample_ids = data['SampleID']
    X_original = data.drop(columns=['SampleID', 'Label']).copy()
    y = data['Label'].astype(float).values.ravel()

    # DO NOT scale globally here. Scaling is inside the Pipeline per CV fold and for the final model.
    X = X_original  # keep as-is

    feature_selection_method = args.feature_selection

    regressors_available = [
        'Neural_Network_reg',
        'Random_Forest_reg',
        'SVM_reg',
        'XGBoost_reg',
        'PLS_reg',
        'KNN_reg',
        'LightGBM_reg'
    ]
    selected_models = [m for m in parsed_models if m in regressors_available]
    if not selected_models:
        print("No valid models selected. Exiting.")
        sys.exit(1)

    Parallel(n_jobs=-1)(
        delayed(run_model)(model_name, X, y, sample_ids, args.prefix, feature_selection_method)
        for model_name in selected_models
    )

    all_models_metrics = []
    for model_name in selected_models:
        metrics_file = f"{args.prefix}_{model_name}_model_performance_metrics.csv"
        if os.path.exists(metrics_file):
            df_metrics = pd.read_csv(metrics_file)
            avg_mse = df_metrics['MSE'].mean()
            avg_mae = df_metrics['MAE'].mean()
            avg_rmse = df_metrics['RMSE'].mean()
            avg_r2 = df_metrics['R2'].mean()
            all_models_metrics.append({
                'Model': model_name,
                'MSE': round(avg_mse, 3),
                'MAE': round(avg_mae, 3),
                'RMSE': round(avg_rmse, 3),
                'R2': round(avg_r2, 3)
            })
        else:
            print(f"Metrics file not found for {model_name}. Skipping summary.")

    if all_models_metrics:
        summary_df = pd.DataFrame(all_models_metrics).sort_values(by='Model').reset_index(drop=True)
        summary_csv_path = f"{args.prefix}_models_summary_metrics.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"Summary of model performance saved to {summary_csv_path}")

        metrics = ['MSE', 'MAE', 'RMSE', 'R2']
        bar_width = 0.2
        index_vals = np.arange(len(summary_df['Model']))
        
        plt.figure(figsize=(12, 8))
        colors = ['skyblue', 'salmon', 'lightgreen', 'orange']
        for i, metric in enumerate(metrics):
            plt.bar(index_vals + i * bar_width, summary_df[metric], bar_width, label=metric, color=colors[i])
        
        max_vals = [summary_df[m].max() for m in metrics]
        min_vals = [summary_df[m].min() for m in metrics]
        global_max = max(max_vals) if max_vals else 0.0
        global_min = min(min_vals) if min_vals else 0.0
        unit = (global_max - min(0.0, global_min)) or 1.0
        
        # staggered but all upward; larger offsets so they don't collide
        series_offsets = [0.04, 0.06, 0.08, 0.10]  # scaled by 'unit'
        min_pad = 0.02 * global_max if global_max > 0 else 0.1  # ensures small bars (e.g., R2) still get visible padding
        
        for i, metric in enumerate(metrics):
            delta = max(series_offsets[i] * unit, min_pad)
            for j, val in enumerate(summary_df[metric]):
                x = index_vals[j] + i * bar_width
                y_pos = float(val) + delta       # always above bar
                plt.text(
                    x, y_pos, f"{float(val):.2f}",
                    ha='center', va='bottom', fontsize=10, zorder=3, clip_on=False
                )
        
        # ---- y-limits: add enough headroom for the tallest label ----
        headroom = max(series_offsets) * unit + min_pad + 0.05 * (global_max if global_max > 0 else 1.0)
        ylim_low = 0.0  # keep baseline at zero so nothing goes below axis
        ylim_high = max(global_max * 1.15, global_max + headroom) if global_max != 0 else 1.0
        if ylim_high <= ylim_low:
            ylim_high = ylim_low + 1.0
        plt.ylim(ylim_low, ylim_high)
        
        plt.xlabel('Model')
        plt.ylabel('Metric Value')
        plt.title('Average Performance Metrics for All Models')
        plt.xticks(index_vals + bar_width * (len(metrics) - 1) / 2, summary_df['Model'], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        summary_chart_path = f"{args.prefix}_models_summary_bar_chart.png"
        plt.savefig(summary_chart_path, dpi=300)
        plt.close()
        print(f"Summary bar chart saved to {summary_chart_path}")
    else:
        print("No metrics available to create a summary bar chart.")

if __name__ == '__main__':
    main_regression()
