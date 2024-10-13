import argparse
import sys
import warnings
from pathlib import Path
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import umap
import optuna
from sklearn.exceptions import ConvergenceWarning  # Import ConvergenceWarning

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="use_inf_as_na option is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, message="n_jobs value 1 overridden to 1 by setting random_state")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def parse_arguments():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Script to perform dimensionality reduction or feature selection on the dataset.")
    parser.add_argument('-i', '--csv', type=str, help='Input file in CSV format', required=True)
    parser.add_argument('-m', '--methods', type=str, nargs='+', choices=['PCA', 'UMAP', 't-SNE', 'KPCA', 'PLS', 'ElasticNet'], help='Name of the method(s)', required=True)
    parser.add_argument('-p', '--prefix', type=str, help='Output prefix')
    parser.add_argument('-d', '--dimensions', type=int, help='Number of dimensions (for dimensionality reduction methods)', default=3)
    return parser.parse_args()

class UMAPEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=3, n_neighbors=15, min_dist=0.1, random_state=42):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state
        self.umap_model = umap.UMAP(n_components=self.n_components, 
                                    n_neighbors=self.n_neighbors, 
                                    min_dist=self.min_dist,
                                    random_state=self.random_state,
                                    n_jobs=1)

    def fit(self, X, y=None):
        self.umap_model.fit(X)
        return self

    def transform(self, X):
        return self.umap_model.transform(X)
    
    def fit_transform(self, X, y=None):
        return self.umap_model.fit_transform(X)

def plot_pairplot(data, plot_prefix, labels, method, dims):
    sns.set(style="white", palette="muted", context="talk")
    df = pd.DataFrame(data, columns=[f"{method}{i+1}" for i in range(dims)])
    df['label'] = labels
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
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
    plt.savefig(f"{plot_prefix}_{method}_result.png", bbox_inches='tight')
    plt.close()

def perform_pca(data, out_prefix, labels, sample_ids, dims=3, random_state=42):
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Perform PCA
    pca = PCA(n_components=dims, svd_solver='full', random_state=random_state)
    pca_result = pca.fit_transform(data_scaled)
    plot_pairplot(pca_result, out_prefix, labels, "PCA", dims)
    df_result = pd.DataFrame(pca_result, columns=[f"PC{i+1}" for i in range(dims)])
    df_result['SampleID'] = sample_ids
    df_result['Label'] = labels
    df_result.to_csv(f"{out_prefix}_PCA_result.csv", index=False)
    explained_var = pca.explained_variance_ratio_
    print(f"PCA explained variance by component: {explained_var}")

def perform_umap(data, out_prefix, labels, sample_ids, dims=3, random_state=42):
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    n_samples = data_scaled.shape[0]
    
    def objective(trial):
        n_neighbors = trial.suggest_int('n_neighbors', 5, 50)  # UMAP neighbor range
        min_dist = trial.suggest_float('min_dist', 0.1, 0.9)
        umap_model = umap.UMAP(n_components=dims, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state, n_jobs=1)
        umap_result = umap_model.fit_transform(data_scaled)
        score = umap_result.var(axis=0).sum()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    best_params = study.best_params
    umap_model = umap.UMAP(n_components=dims, **best_params, random_state=random_state, n_jobs=1)
    umap_result = umap_model.fit_transform(data_scaled)
    plot_pairplot(umap_result, out_prefix, labels, "UMAP", dims)
    df_result = pd.DataFrame(umap_result, columns=[f"UMAP{i+1}" for i in range(dims)])
    df_result['SampleID'] = sample_ids
    df_result['Label'] = labels
    df_result.to_csv(f"{out_prefix}_UMAP_result.csv", index=False)
    print(f"Best UMAP params: {best_params}")

def perform_tsne(data, out_prefix, labels, sample_ids, dims=3, random_state=42):
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    def objective(trial):
        perplexity = trial.suggest_int('perplexity', 5, 50)
        tsne = TSNE(n_components=dims, perplexity=perplexity, random_state=random_state, method='exact')
        tsne_result = tsne.fit_transform(data_scaled)
        score = tsne_result.var(axis=0).sum()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    best_perplexity = study.best_params['perplexity']
    tsne = TSNE(n_components=dims, perplexity=best_perplexity, random_state=random_state, method='exact')
    tsne_result = tsne.fit_transform(data_scaled)
    plot_pairplot(tsne_result, out_prefix, labels, "t-SNE", dims)
    df_result = pd.DataFrame(tsne_result, columns=[f"t-SNE{i+1}" for i in range(dims)])
    df_result['SampleID'] = sample_ids
    df_result['Label'] = labels
    df_result.to_csv(f"{out_prefix}_t-SNE_result.csv", index=False)
    print(f"Best t-SNE perplexity: {best_perplexity}")

def perform_kernel_pca(data, out_prefix, labels, sample_ids, dims=3, random_state=42):
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    def objective(trial):
        gamma = trial.suggest_float('gamma', 1e-4, 1e4, log=True)
        kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid', 'cosine'])
        kpca = KernelPCA(n_components=dims, kernel=kernel, gamma=gamma, random_state=random_state)
        kpca_result = kpca.fit_transform(data_scaled)
        score = kpca_result.var(axis=0).sum()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    best_params = study.best_params
    kpca = KernelPCA(n_components=dims, **best_params, random_state=random_state)
    kpca_result = kpca.fit_transform(data_scaled)
    plot_pairplot(kpca_result, out_prefix, labels, "KPCA", dims)
    df_result = pd.DataFrame(kpca_result, columns=[f"KPCA{i+1}" for i in range(dims)])
    df_result['SampleID'] = sample_ids
    df_result['Label'] = labels
    df_result.to_csv(f"{out_prefix}_KPCA_result.csv", index=False)
    print(f"Best KPCA params: {best_params}")

def perform_pls(data, out_prefix, labels, sample_ids, dims=3, random_state=42):
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Convert labels to numeric values if necessary
    if labels.dtype == 'object' or not np.issubdtype(labels.dtype, np.number):
        le = LabelEncoder()
        numeric_labels = le.fit_transform(labels)
    else:
        numeric_labels = labels

    # Perform PLS with user-specified dimensions
    pls = PLSRegression(n_components=dims)
    pls.fit(data_scaled, numeric_labels)
    pls_result = pls.transform(data_scaled)
    
    # Check if the transformed PLS result has the expected shape
    if pls_result.shape[1] != dims:
        raise ValueError(f"PLS result dimensions ({pls_result.shape[1]}) do not match the expected dimensions ({dims})")

    # Plot using original labels
    plot_pairplot(pls_result, out_prefix, labels, "PLS", dims)  # Use original labels for the plot
    df_result = pd.DataFrame(pls_result, columns=[f"PLS{i+1}" for i in range(dims)])
    df_result['SampleID'] = sample_ids
    df_result['Label'] = labels  # Use original labels in the output file
    df_result.to_csv(f"{out_prefix}_PLS_result.csv", index=False)

def perform_elastic_net(data, out_prefix, labels, sample_ids, random_state=42):
    # Convert labels to numeric values if necessary
    label_encoder_used = False
    if labels.dtype == 'object' or not np.issubdtype(labels.dtype, np.number):
        le = LabelEncoder()
        y = le.fit_transform(labels)
        label_encoder_used = True
    else:
        y = labels
    
    # Use cross-validation for feature selection
    def objective(trial):
        l1_ratio = trial.suggest_float('l1_ratio', 0.1, 0.9)
        alpha = trial.suggest_float('alpha', 1e-4, 10, log=True)  # Reasonable alpha range
        elastic_net = ElasticNetCV(l1_ratio=[l1_ratio], alphas=[alpha], cv=5, random_state=random_state, max_iter=90000000, tol=1e-5)
        score = cross_val_score(elastic_net, data, y, cv=5, scoring='neg_mean_squared_error').mean()
        return score

    study = optuna.create_study(direction='minimize')  # Minimize ElasticNet error
    study.optimize(objective, n_trials=50)

    best_alpha = study.best_params['alpha']
    best_l1_ratio = study.best_params['l1_ratio']
    
    # Train final ElasticNet model
    elastic_net = ElasticNetCV(l1_ratio=[best_l1_ratio], alphas=[best_alpha], cv=5, random_state=random_state, max_iter=90000000, tol=1e-5)
    elastic_net.fit(data, y)
    
    # Get ElasticNet coefficients
    coef = elastic_net.coef_
    
    # Set minimum number of features to retain, at least 30 or 10% of total features
    min_features = max(int(0.1 * data.shape[1]), 30)
    
    # Select features based on coefficients
    important_features = np.argsort(np.abs(coef))[-min_features:]
    
    # Ensure selected features are unique and sorted
    important_features = np.unique(important_features)
    if len(important_features) > min_features:
        important_features = important_features[-min_features:]
    
    # Fix invalid indexing for Pandas DataFrame
    X_selected = data.iloc[:, important_features]  # Use iloc for Pandas indexing
    
    # Standardize selected features and save
    scaler = StandardScaler()
    X_selected_scaled = scaler.fit_transform(X_selected)
    
    print(f'Original number of features: {data.shape[1]}')
    print(f'Number of selected features after Elastic Net: {X_selected_scaled.shape[1]}')
    print(f'Best alpha: {best_alpha}')
    print(f'Best l1_ratio: {best_l1_ratio}')
    
    # Save selected features
    original_feature_names = data.columns if isinstance(data, pd.DataFrame) else [f"Feature_{i}" for i in range(data.shape[1])]
    selected_feature_names = original_feature_names[important_features]
    selected_features_df = pd.DataFrame(X_selected_scaled, columns=selected_feature_names)
    selected_features_df['SampleID'] = sample_ids
    selected_features_df['Label'] = labels
    selected_features_df.to_csv(f'{out_prefix}_ElasticNet_selected_features_with_labels.csv', index=False)

def perform_dimensionality_reduction(file_path, file_prefix, method, dims=3, standardize=True, random_state=42):
    # Load data and exclude 'SampleID' column if present
    data = pd.read_csv(file_path)
    sample_ids = data['SampleID'] if 'SampleID' in data.columns else None
    if 'SampleID' in data.columns:
        labels = data['Label']
        data = data.drop(columns=['Label', 'SampleID'])  # Exclude 'SampleID' column
    else:
        labels = data['Label']
        data = data.drop(columns=['Label'])  # If 'SampleID' is not present
    
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)  # Drop rows with NaN values to ensure valid input
    
    if method == 'PCA':
        perform_pca(data, file_prefix, labels, sample_ids, dims, random_state)
    elif method == 'UMAP':
        perform_umap(data, file_prefix, labels, sample_ids, dims, random_state)
    elif method == 't-SNE':
        perform_tsne(data, file_prefix, labels, sample_ids, dims, random_state)
    elif method == 'KPCA':
        perform_kernel_pca(data, file_prefix, labels, sample_ids, dims, random_state)
    elif method == 'PLS':
        perform_pls(data, file_prefix, labels, sample_ids, dims, random_state)
    elif method == 'ElasticNet':
        perform_elastic_net(data, file_prefix, labels, sample_ids, random_state)
    else:
        raise ValueError("Unsupported method. Choose from 'PCA', 'UMAP', 't-SNE', 'KPCA', 'PLS', or 'ElasticNet'.")

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
            standardize=True,  # Enforce standardization
            random_state=42  # Fix random seed
        ) for method_name in methods
    )
