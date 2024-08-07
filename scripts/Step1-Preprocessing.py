import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import make_scorer
import umap

class UMAPEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=3, n_neighbors=15, min_dist=0.1, random_state=42):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state
        self.umap_model = umap.UMAP(n_components=self.n_components, 
                                    n_neighbors=self.n_neighbors, 
                                    min_dist=self.min_dist,
                                    random_state=self.random_state)

    def fit(self, X, y=None):
        self.umap_model.fit(X)
        return self

    def transform(self, X):
        return self.umap_model.transform(X)
    
    def fit_transform(self, X, y=None):
        return self.umap_model.fit_transform(X)

def umap_score(estimator, X):
    transformed = estimator.fit_transform(X)
    return transformed.var(axis=0).sum()

def plot_pairplot(data, labels, method, dims):
    sns.set(style="white", palette="muted", context="talk")
    df = pd.DataFrame(data, columns=[f"{method}{i+1}" for i in range(dims)])
    df['label'] = labels
    g = sns.PairGrid(df, hue='label', corner=True)
    g.map_lower(sns.scatterplot, s=40, edgecolor="w", linewidth=0.5)
    g.map_diag(sns.histplot, kde=True, fill=True, alpha=0.6)
    g.add_legend()
    for ax in g.axes.flatten():
        if ax:
            ax.set_title(ax.get_title(), size=14)
            ax.set_xlabel(ax.get_xlabel(), size=12)
            ax.set_ylabel(ax.get_ylabel(), size=12)
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle(f"Dimensions' view from {method}", size=16)
    plt.savefig(f"{method}_result.png", bbox_inches='tight')
    plt.close()

def perform_pca(data, labels, dims=15, standardize=True, random_state=42):
    if standardize:
        data = StandardScaler().fit_transform(data)
    pca = PCA()
    param_grid = {'n_components': np.arange(2, dims+1)}
    grid_search = GridSearchCV(pca, param_grid, cv=5)
    grid_search.fit(data)
    best_params = grid_search.best_params_
    pca = PCA(n_components=best_params['n_components'], random_state=random_state)
    pca_result = pca.fit_transform(data)
    plot_pairplot(pca_result, labels, "PC", best_params['n_components'])
    df_result = pd.DataFrame(pca_result, columns=[f"PC{i+1}" for i in range(best_params['n_components'])])
    df_result['Label'] = labels
    df_result.to_csv("PCA_result.csv", index=False)
    explained_var = pca.explained_variance_ratio_
    print(f"Best PCA params: {best_params}")
    print(f"Explained variance by component: {explained_var}")

def perform_umap(data, labels, dims=3, standardize=True, random_state=42):
    if standardize:
        data = StandardScaler().fit_transform(data)
    n_samples = data.shape[0]
    param_grid = {'n_neighbors': np.arange(5, min(50, n_samples), 5), 'min_dist': np.linspace(0.1, 0.9, 9)}
    umap_model = UMAPEstimator(n_components=dims, random_state=random_state)
    grid_search = GridSearchCV(umap_model, param_grid, scoring=make_scorer(umap_score), cv=5)
    grid_search.fit(data)
    best_params = grid_search.best_params_
    umap_model = UMAPEstimator(n_components=dims, **best_params, random_state=random_state)
    umap_result = umap_model.fit_transform(data)
    plot_pairplot(umap_result, labels, "UMAP", dims)
    df_result = pd.DataFrame(umap_result, columns=[f"UMAP{i+1}" for i in range(dims)])
    df_result['Label'] = labels
    df_result.to_csv("UMAP_result.csv", index=False)
    print(f"Best UMAP params: {best_params}")

def perform_tsne(data, labels, dims=3, standardize=True, random_state=42):
    if standardize:
        data = StandardScaler().fit_transform(data)
    param_grid = {'perplexity': np.arange(5, min(50, data.shape[0]), 5)}
    tsne_results = []
    for perplexity in param_grid['perplexity']:
        tsne = TSNE(n_components=dims, perplexity=perplexity, random_state=random_state)
        tsne_result = tsne.fit_transform(data)
        tsne_results.append((tsne_result, perplexity, tsne_result.var(axis=0).sum()))
    best_result = max(tsne_results, key=lambda x: x[2])
    tsne_result, best_perplexity, _ = best_result
    plot_pairplot(tsne_result, labels, "t-SNE", dims)
    df_result = pd.DataFrame(tsne_result, columns=[f"t-SNE{i+1}" for i in range(dims)])
    df_result['Label'] = labels
    df_result.to_csv("t-SNE_result.csv", index=False)
    print(f"Best t-SNE params: {{'perplexity': {best_perplexity}}}")

def perform_kernel_pca(data, labels, dims=3, standardize=True, random_state=42):
    if standardize:
        data = StandardScaler().fit_transform(data)
    param_grid = {'gamma': np.logspace(-4, 4, 20), 'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']}
    best_score = -1
    best_params = None
    kpca_results = []
    for kernel in param_grid['kernel']:
        for gamma in param_grid['gamma']:
            kpca = KernelPCA(n_components=dims, kernel=kernel, gamma=gamma, random_state=random_state)
            kpca_result = kpca.fit_transform(data)
            score = kpca_result.var(axis=0).sum()
            kpca_results.append((kpca_result, {'kernel': kernel, 'gamma': gamma}, score))
    best_result = max(kpca_results, key=lambda x: x[2])
    kpca_result, best_params, _ = best_result
    plot_pairplot(kpca_result, labels, "KPCA", dims)
    df_result = pd.DataFrame(kpca_result, columns=[f"KPCA{i+1}" for i in range(dims)])
    df_result['Label'] = labels
    df_result.to_csv("KPCA_result.csv", index=False)
    print(f"Best KPCA params: {best_params}")

def perform_pls(data, labels, dims=3, standardize=True, random_state=42):
    if standardize:
        data = StandardScaler().fit_transform(data)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    param_grid = {'n_components': np.arange(2, dims+1)}
    pls = PLSRegression()
    grid_search = GridSearchCV(pls, param_grid, scoring='neg_mean_squared_error', cv=5)
    grid_search.fit(data, labels)
    best_params = grid_search.best_params_
    pls = PLSRegression(n_components=best_params['n_components'])
    pls_result = pls.fit_transform(data, labels)[0]
    plot_pairplot(pls_result, labels, "PLS", best_params['n_components'])
    df_result = pd.DataFrame(pls_result, columns=[f"PLS{i+1}" for i in range(best_params['n_components'])])
    df_result['Label'] = labels
    df_result.to_csv("PLS_result.csv", index=False)
    print(f"Best PLS params: {best_params}")

def perform_dimensionality_reduction(file_path, method, dims=3, standardize=True):
    data = pd.read_csv(file_path)
    labels = data['Label']
    data = data.drop(columns=['Label'])
    
    if method == 'PCA':
        perform_pca(data, labels, dims, standardize)
    elif method == 'UMAP':
        perform_umap(data, labels, dims, standardize)
    elif method == 't-SNE':
        perform_tsne(data, labels, dims, standardize)
    elif method == 'KPCA':
        perform_kernel_pca(data, labels, dims, standardize)
    elif method == 'PLS':
        perform_pls(data, labels, dims, standardize)
    else:
        raise ValueError("Unsupported method. Choose from 'PCA', 'UMAP', 't-SNE', 'KPCA', or 'PLS'.")

# Example usage
file_path = "/Users/yuhan/Downloads/proDataLabel.csv"
perform_dimensionality_reduction(file_path, method='PCA', dims=3, standardize=True)
perform_dimensionality_reduction(file_path, method='UMAP', dims=3, standardize=True)
perform_dimensionality_reduction(file_path, method='t-SNE', dims=3, standardize=True)
perform_dimensionality_reduction(file_path, method='KPCA', dims=3, standardize=True)
perform_dimensionality_reduction(file_path, method='PLS', dims=3, standardize=True)
