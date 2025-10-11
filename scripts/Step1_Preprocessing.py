import argparse
import sys
import warnings
import random
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed

import pandas as pd
import matplotlib
# Set the backend before importing pyplot to avoid issues in multiprocessing
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin

from umap import UMAP
import optuna
from optuna.samplers import TPESampler
from sklearn.exceptions import ConvergenceWarning
from matplotlib.colors import Normalize

# ---------------- Reproducibility ----------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="use_inf_as_na option is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, message="n_jobs value 1 overridden to 1 by setting random_state")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script to perform dimensionality reduction or feature selection on the dataset.")
    parser.add_argument('-i', '--csv', type=str, help='Input file in CSV format', required=True)
    parser.add_argument('-m', '--methods', type=str, nargs='+', choices=['pca', 'umap', 'tsne', 'kpca', 'pls', 'elasticnet'], help='Name of the method(s)', required=True)
    parser.add_argument('-p', '--prefix', type=str, help='Output prefix')
    parser.add_argument('-d', '--dimensions', type=int, help='Number of dimensions (for dimensionality reduction methods)', default=3)
    return parser.parse_args()

class UMAPEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=3, n_neighbors=15, min_dist=0.1, random_state=42):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state
        self.umap_model = UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            random_state=self.random_state
        )

    def fit(self, X, y=None):
        self.umap_model.fit(X)
        return self

    def transform(self, X):
        return self.umap_model.transform(X)
    
    def fit_transform(self, X, y=None):
        return self.umap_model.fit_transform(X)

def plot_pairplot(data, plot_prefix, labels, method, dims, label_type):
    sns.set(style="white", palette="muted", context="talk", font_scale=2)
    dim_columns = [f"{method}{i+1}" for i in range(dims)]
    df = pd.DataFrame(data, columns=dim_columns)
    df['Label'] = labels
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    if label_type == 'categorical':
        g = sns.PairGrid(df, vars=dim_columns, hue='Label', corner=True)
        g.map_lower(sns.scatterplot, s=40, edgecolor="w", linewidth=0.5)
        g.map_diag(sns.histplot, kde=True, fill=True, alpha=0.6)
        g.add_legend(fontsize=22, title_fontsize=24)
    else:
        g = sns.PairGrid(df, vars=dim_columns, corner=True)
        def scatter_continuous(x, y, **kwargs):
            idx = x.index
            plt.scatter(x, y, c=df.loc[idx, 'Label'], cmap='viridis', s=40, edgecolor="w", linewidth=0.5, alpha=0.6)
        g.map_lower(scatter_continuous)
        g.map_diag(sns.histplot, kde=True, fill=True, alpha=0.6)
        norm = Normalize(df['Label'].min(), df['Label'].max())
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        g.fig.subplots_adjust(right=0.92)
        cbar_ax = g.fig.add_axes([0.94, 0.15, 0.02, 0.7])
        g.fig.colorbar(sm, cax=cbar_ax, orientation="vertical", label="Label").ax.tick_params(labelsize=18)
        cbar_ax.set_ylabel("Label", fontsize=22)

    g.fig.set_size_inches(12, 12)
    plt.subplots_adjust(top=0.95, wspace=0.3, hspace=0.3)
    g.fig.suptitle(f"Dimensions' view from {method.upper()}", fontsize=30, fontweight='bold')

    if method == 'kpca':
        for ax in g.axes.flatten():
            if ax is not None:
                ax.ticklabel_format(style='sci', scilimits=(0,0))
                ax.xaxis.offsetText.set_visible(False)
                ax.yaxis.offsetText.set_visible(False)
                ax.set_xlabel(ax.get_xlabel().split('[')[0])
                ax.set_ylabel(ax.get_ylabel().split('[')[0])
    
    plt.savefig(f"{plot_prefix}_{method}_result.png", bbox_inches='tight', dpi=300)
    plt.close()

def detect_label_type(labels, threshold=10):
    if labels.dtype == 'object':
        return 'categorical'
    elif pd.api.types.is_numeric_dtype(labels):
        unique_values = labels.nunique()
        if unique_values <= threshold:
            return 'categorical'
        else:
            return 'continuous'
    else:
        return 'categorical'

def perform_pca(data, out_prefix, labels, sample_ids, dims=3, random_state=42, label_type='categorical'):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    pca = PCA(n_components=dims, svd_solver='full')
    pca_result = pca.fit_transform(data_scaled)
    plot_pairplot(pca_result, out_prefix, labels, "pca", dims, label_type)
    df_result = pd.DataFrame(pca_result, columns=[f"PC{i+1}" for i in range(dims)])
    if sample_ids is not None:
        df_result['SampleID'] = sample_ids
    df_result['Label'] = labels
    df_result.to_csv(f"{out_prefix}_pca_result.csv", index=False)
    explained_var = pca.explained_variance_ratio_
    print(f"PCA explained variance by component: {explained_var}")

def perform_umap(data, out_prefix, labels, sample_ids, dims=3, random_state=42, label_type='categorical'):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    n_samples = data_scaled.shape[0]
    max_neighbors = max(2, min(50, n_samples - 1))
    
    def objective(trial):
        try:
            n_neighbors = trial.suggest_int('n_neighbors', 2, max_neighbors)
            min_dist = trial.suggest_float('min_dist', 0.1, 0.9)
            umap_model = UMAP(n_components=dims, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
            umap_result = umap_model.fit_transform(data_scaled)
            score = umap_result.var(axis=0).sum()
            if not np.isfinite(score):
                raise ValueError("Non-finite score")
            return score
        except Exception as e:
            print(f"UMAP trial failed with parameters: n_neighbors={locals().get('n_neighbors', None)}, min_dist={locals().get('min_dist', None)}. Error: {e}")
            return -1e10
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=50)
    
    if not study.best_params:
        print("UMAP optimization failed. Using default parameters.")
        best_params = {}
    else:
        best_params = study.best_params
    
    if 'n_neighbors' in best_params:
        best_params['n_neighbors'] = min(max_neighbors, max(2, best_params['n_neighbors']))
    umap_model = UMAP(n_components=dims, **best_params, random_state=random_state)
    umap_result = umap_model.fit_transform(data_scaled)
    plot_pairplot(umap_result, out_prefix, labels, "umap", dims, label_type)
    df_result = pd.DataFrame(umap_result, columns=[f"UMAP{i+1}" for i in range(dims)])
    if sample_ids is not None:
        df_result['SampleID'] = sample_ids
    df_result['Label'] = labels
    df_result.to_csv(f"{out_prefix}_umap_result.csv", index=False)
    print(f"Best UMAP params: {best_params}")

def perform_tsne(data, out_prefix, labels, sample_ids, dims=3, random_state=42, label_type='categorical'):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    n_samples = data_scaled.shape[0]
    max_perp = max(2, min(50, (n_samples - 1) // 3 if n_samples > 3 else n_samples - 1))
    
    def objective(trial):
        try:
            perplexity = trial.suggest_int('perplexity', 2, max_perp)
            tsne = TSNE(n_components=dims, perplexity=perplexity, random_state=random_state, method='exact')
            tsne_result = tsne.fit_transform(data_scaled)
            score = tsne_result.var(axis=0).sum()
            if not np.isfinite(score):
                raise ValueError("Non-finite score")
            return score
        except Exception as e:
            print(f"t-SNE trial failed with perplexity={locals().get('perplexity', None)}. Error: {e}")
            return -1e10
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=50)
    
    if not study.best_params:
        print("t-SNE optimization failed. Using default perplexity=30.")
        best_perplexity = min(max_perp, 30)
    else:
        best_perplexity = study.best_params['perplexity']
    
    best_perplexity = min(max_perp, max(2, best_perplexity))
    tsne = TSNE(n_components=dims, perplexity=best_perplexity, random_state=random_state, method='exact')
    tsne_result = tsne.fit_transform(data_scaled)
    plot_pairplot(tsne_result, out_prefix, labels, "tsne", dims, label_type)
    df_result = pd.DataFrame(tsne_result, columns=[f"t-SNE{i+1}" for i in range(dims)])
    if sample_ids is not None:
        df_result['SampleID'] = sample_ids
    df_result['Label'] = labels
    df_result.to_csv(f"{out_prefix}_tsne_result.csv", index=False)
    print(f"Best t-SNE perplexity: {best_perplexity}")

def perform_kernel_pca(data, out_prefix, labels, sample_ids, dims=3, random_state=42, label_type='categorical'):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    def objective(trial):
        try:
            gamma = trial.suggest_float('gamma', 1e-3, 1e-1, log=True)
            kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'cosine'])
            kpca = KernelPCA(n_components=dims, kernel=kernel, gamma=gamma)
            kpca_result = kpca.fit_transform(data_scaled)
            score = kpca_result.var(axis=0).sum()
            if not np.isfinite(score):
                raise ValueError("Non-finite score")
            return score
        except Exception as e:
            print(f"KernelPCA trial failed with parameters: gamma={locals().get('gamma', None)}, kernel={locals().get('kernel', None)}. Error: {e}")
            return -1e10
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=50)
    
    if not study.best_params:
        print("KernelPCA optimization failed. Using default parameters.")
        best_params = {}
    else:
        best_params = study.best_params
    
    try:
        kpca = KernelPCA(n_components=dims, **best_params)
        kpca_result = kpca.fit_transform(data_scaled)
        kpca_result = StandardScaler().fit_transform(kpca_result)
    except Exception as e:
        print(f"KernelPCA fitting failed with best parameters: {best_params}. Error: {e}")
        return
    
    plot_pairplot(kpca_result, out_prefix, labels, "kpca", dims, label_type)
    df_result = pd.DataFrame(kpca_result, columns=[f"KPCA{i+1}" for i in range(dims)])
    if sample_ids is not None:
        df_result['SampleID'] = sample_ids
    df_result['Label'] = labels
    df_result.to_csv(f"{out_prefix}_kpca_result.csv", index=False)
    print(f"Best KPCA params: {best_params}")

def perform_pls(data, out_prefix, labels, sample_ids, dims=3, random_state=42, label_type='categorical'):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    if label_type == 'categorical':
        le = LabelEncoder()
        numeric_labels = le.fit_transform(labels)
    else:
        numeric_labels = labels

    n_comp = min(dims, data_scaled.shape[1], data_scaled.shape[0])
    pls = PLSRegression(n_components=n_comp)
    pls.fit(data_scaled, numeric_labels)
    pls_result = pls.transform(data_scaled)

    plot_pairplot(pls_result, out_prefix, labels, "pls", n_comp, label_type)
    df_result = pd.DataFrame(pls_result, columns=[f"PLS{i+1}" for i in range(n_comp)])
    if sample_ids is not None:
        df_result['SampleID'] = sample_ids
    df_result['Label'] = labels
    df_result.to_csv(f"{out_prefix}_pls_result.csv", index=False)

def perform_elastic_net(data, out_prefix, labels, sample_ids, random_state=42, label_type='categorical'):
    if label_type == 'categorical':
        le = LabelEncoder()
        y = le.fit_transform(labels)
        print("ElasticNet is being used for feature selection with categorical labels treated as numeric.")
    else:
        y = labels
    
    def objective(trial):
        try:
            l1_ratio = trial.suggest_float('l1_ratio', 0.1, 0.9)
            alpha = trial.suggest_float('alpha', 1e-4, 10, log=True)
            elastic_net = ElasticNetCV(l1_ratio=[l1_ratio], alphas=[alpha], cv=5, random_state=random_state, max_iter=100000, tol=1e-5)
            score = cross_val_score(elastic_net, data, y, cv=5, scoring='neg_mean_squared_error').mean()
            if not np.isfinite(score):
                raise ValueError("Non-finite score")
            return score
        except Exception as e:
            print(f"ElasticNet trial failed with parameters: l1_ratio={locals().get('l1_ratio', None)}, alpha={locals().get('alpha', None)}. Error: {e}")
            return -1e10
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=50)
    
    if not study.best_params:
        print("ElasticNet optimization failed. Using default parameters.")
        best_alpha = 1.0
        best_l1_ratio = 0.5
    else:
        best_alpha = study.best_params['alpha']
        best_l1_ratio = study.best_params['l1_ratio']
    
    try:
        elastic_net = ElasticNetCV(l1_ratio=[best_l1_ratio], alphas=[best_alpha], cv=5, random_state=random_state, max_iter=100000, tol=1e-5)
        elastic_net.fit(data, y)
    except Exception as e:
        print(f"ElasticNet fitting failed with best parameters: l1_ratio={best_l1_ratio}, alpha={best_alpha}. Error: {e}")
        return
    
    coef = elastic_net.coef_
    min_features = max(int(0.1 * data.shape[1]), 30)
    important_features = np.argsort(np.abs(coef))[-min_features:]
    important_features = np.unique(important_features)
    if len(important_features) > min_features:
        important_features = important_features[-min_features:]
    
    X_selected = data.iloc[:, important_features]
    scaler = StandardScaler()
    X_selected_scaled = scaler.fit_transform(X_selected)
    
    print(f'Original number of features: {data.shape[1]}')
    print(f'Number of selected features after Elastic Net: {X_selected_scaled.shape[1]}')
    print(f'Best alpha: {best_alpha}')
    print(f'Best l1_ratio: {best_l1_ratio}')
    
    original_feature_names = data.columns if isinstance(data, pd.DataFrame) else [f"Feature_{i}" for i in range(data.shape[1])]
    selected_feature_names = original_feature_names[important_features]
    selected_features_df = pd.DataFrame(X_selected_scaled, columns=selected_feature_names)
    if sample_ids is not None:
        selected_features_df['SampleID'] = sample_ids
    selected_features_df['Label'] = labels
    selected_features_df.to_csv(f'{out_prefix}_elasticnet_result.csv', index=False)

def perform_dimensionality_reduction(file_path, file_prefix, method, dims=3, standardize=True, random_state=42):
    data = pd.read_csv(file_path)
    
    if 'SampleID' in data.columns:
        sample_ids = data['SampleID']
    else:
        sample_ids = None
    
    if 'Label' in data.columns:
        labels = data['Label']
    else:
        raise ValueError("The input CSV must contain a 'Label' column.")
    
    if 'SampleID' in data.columns:
        feature_data = data.drop(columns=['Label', 'SampleID'])
    else:
        feature_data = data.drop(columns=['Label'])
    
    combined = feature_data.copy()
    combined['Label'] = labels
    
    label_type = detect_label_type(combined['Label'])
    print(f"Detected label type: {label_type}")
    
    combined.replace([np.inf, -np.inf], np.nan, inplace=True)
    combined.dropna(inplace=True)
    
    data_clean = combined.drop(columns=['Label'])
    labels_clean = combined['Label']
    
    if sample_ids is not None:
        sample_ids_clean = sample_ids.loc[combined.index]
    else:
        sample_ids_clean = None
    
    if method == 'pca':
        perform_pca(data_clean, file_prefix, labels_clean, sample_ids_clean, dims, random_state, label_type)
    elif method == 'umap':
        perform_umap(data_clean, file_prefix, labels_clean, sample_ids_clean, dims, random_state, label_type)
    elif method == 'tsne':
        perform_tsne(data_clean, file_prefix, labels_clean, sample_ids_clean, dims, random_state, label_type)
    elif method == 'kpca':
        perform_kernel_pca(data_clean, file_prefix, labels_clean, sample_ids_clean, dims, random_state, label_type)
    elif method == 'pls':
        perform_pls(data_clean, file_prefix, labels_clean, sample_ids_clean, dims, random_state, label_type)
    elif method == 'elasticnet':
        if label_type == 'categorical':
            print("Warning: ElasticNet is being applied to categorical labels treated as numeric.")
        perform_elastic_net(data_clean, file_prefix, labels_clean, sample_ids_clean, random_state, label_type)
    else:
        raise ValueError("Unsupported method. Choose from 'pca', 'umap', 'tsne', 'kpca', 'pls', or 'elasticnet'.")

if __name__ == "__main__":
    args = parse_arguments()
    prefix = Path(args.csv).stem
    if args.prefix:
        prefix = args.prefix
    methods = args.methods
    dims = args.dimensions
    
    results = Parallel(n_jobs=-1, backend='multiprocessing', verbose=100)(
        delayed(perform_dimensionality_reduction)(
            args.csv, 
            prefix, 
            method=method_name, 
            dims=dims, 
            standardize=True,
            random_state=42
        ) for method_name in methods
    )
