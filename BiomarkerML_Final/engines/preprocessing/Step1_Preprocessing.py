import argparse
import sys
import warnings
import random
import numpy as np
from pathlib import Path
from sklearn.utils.parallel import Parallel, delayed

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except Exception:
    sns = None
from core.publication_utils import (
    BorutaProteinSelector,
    DEFAULT_RANDOM_STATE,
    configure_publication_plots,
    set_global_seed,
    task_run_context,
)
from core.adaptive_optuna import run_adaptive_study, tolerance_for_objective

configure_publication_plots()

from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from umap import UMAP
except Exception:
    UMAP = None
import optuna
from optuna.samplers import TPESampler
from sklearn.exceptions import ConvergenceWarning
from matplotlib.colors import Normalize

def set_seed(seed: int = DEFAULT_RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)

set_seed(DEFAULT_RANDOM_STATE)

warnings.filterwarnings("ignore", category=FutureWarning, message="use_inf_as_na option is no longer recommended")
warnings.filterwarnings("ignore", category=UserWarning, message="n_jobs value 1 overridden to 1 by setting random_state")
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _safe_umap_n_neighbors(data, n_neighbors: int) -> int:
    matrix = np.asarray(data)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return 1
    return min(max(1, int(n_neighbors)), matrix.shape[0] - 1)


def _umap_initialization(data, n_neighbors: int) -> str:
    """Use deterministic non-spectral UMAP initialization to avoid disconnected-graph spectral warnings."""
    return "random"


def _build_umap(data, *, n_components: int, n_neighbors: int, min_dist: float, random_state: int):
    if UMAP is None:
        raise ImportError("UMAP requires the optional 'umap-learn' package.")
    safe_neighbors = _safe_umap_n_neighbors(data, n_neighbors)
    return UMAP(
        n_components=n_components,
        n_neighbors=safe_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        init=_umap_initialization(data, safe_neighbors),
    )

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate exploratory embeddings or standalone protein feature-selection outputs.")
    parser.add_argument('--input', '--csv', '-i', dest='csv', type=str, help='Input CSV file.', required=True)
    parser.add_argument('-m', '--methods', type=str, nargs='+', choices=['pca', 'umap', 'tsne', 'kpca', 'pls', 'elasticnet', 'boruta'], help='Method name(s)', required=True)
    parser.add_argument('--prefix', '-p', type=str, help='Output prefix')
    parser.add_argument('--outdir', type=str, default='.', help='Base output directory.')
    parser.add_argument('--feature_selection', choices=['yes', 'no'], default='no', help='Accepted for CLI consistency; preprocessing methods are selected with --methods.')
    parser.add_argument('--jobs', type=int, default=1, help='Parallel workers.')
    parser.add_argument('--seed', type=int, default=DEFAULT_RANDOM_STATE, help='Random seed.')
    parser.add_argument('-d', '--dimensions', type=int, help='Number of dimensions (for dimensionality reduction methods)', default=3)
    return parser.parse_args()

class UMAPEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=3, n_neighbors=15, min_dist=0.1, random_state=DEFAULT_RANDOM_STATE):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state
        if UMAP is None:
            raise ImportError("UMAP requires the optional 'umap-learn' package.")
        self.umap_model = None

    def _model_for(self, X):
        return _build_umap(
            X,
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            random_state=self.random_state,
        )

    def fit(self, X, y=None):
        self.umap_model = self._model_for(X)
        self.umap_model.fit(X)
        return self

    def transform(self, X):
        if self.umap_model is None:
            raise ValueError("UMAPEstimator must be fitted before transform.")
        return self.umap_model.transform(X)
    
    def fit_transform(self, X, y=None):
        self.umap_model = self._model_for(X)
        return self.umap_model.fit_transform(X)

def plot_pairplot(data, plot_prefix, labels, method, dims, label_type):
    dim_columns = [f"{method}{i+1}" for i in range(dims)]
    df = pd.DataFrame(data, columns=dim_columns)
    df['Label'] = labels
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    if sns is None:
        fig, axes = plt.subplots(dims, dims, figsize=(12, 12))
        for i, y_col in enumerate(dim_columns):
            for j, x_col in enumerate(dim_columns):
                ax = axes[i, j]
                if i < j:
                    ax.axis("off")
                elif i == j:
                    ax.hist(df[x_col], bins=25, color="#4C78A8", alpha=0.75)
                else:
                    if label_type == 'categorical':
                        for label in pd.unique(df['Label']):
                            subset = df[df['Label'] == label]
                            ax.scatter(subset[x_col], subset[y_col], s=18, alpha=0.7, label=str(label))
                    else:
                        ax.scatter(df[x_col], df[y_col], c=df['Label'], cmap='viridis', s=18, alpha=0.7)
                if i == dims - 1:
                    ax.set_xlabel(x_col)
                if j == 0 and i > 0:
                    ax.set_ylabel(y_col)
        fig.suptitle(f"Dimensions' view from {method.upper()}", fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{plot_prefix}_{method}_result.png", bbox_inches='tight', dpi=300)
        plt.close()
        return

    sns.set(style="white", palette="muted", context="talk", font_scale=2)
    
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

def perform_pca(data, out_prefix, labels, sample_ids, dims=3, random_state=DEFAULT_RANDOM_STATE, label_type='categorical'):
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

def perform_umap(data, out_prefix, labels, sample_ids, dims=3, random_state=DEFAULT_RANDOM_STATE, label_type='categorical'):
    if UMAP is None:
        raise ImportError("UMAP requires the optional 'umap-learn' package.")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    n_samples = data_scaled.shape[0]
    max_neighbors = max(2, min(50, n_samples - 1))
    
    def objective(trial):
        try:
            n_neighbors = trial.suggest_int('n_neighbors', 2, max_neighbors)
            min_dist = trial.suggest_float('min_dist', 0.1, 0.9)
            umap_model = _build_umap(
                data_scaled,
                n_components=dims,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=random_state,
            )
            umap_result = umap_model.fit_transform(data_scaled)
            score = umap_result.var(axis=0).sum()
            if not np.isfinite(score):
                raise ValueError("Non-finite score")
            return score
        except Exception as e:
            print(f"UMAP trial failed with parameters: n_neighbors={locals().get('n_neighbors', None)}, min_dist={locals().get('min_dist', None)}. Error: {e}")
            raise
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=random_state))
    study_artifact = Path(f"{out_prefix}_umap_optuna_study.json").resolve()
    run_adaptive_study(
        study,
        objective,
        task="preprocessing",
        model="UMAP",
        study_scope="full_input_embedding",
        objective_name="umap_embedding_variance",
        direction="maximize",
        tolerance=tolerance_for_objective(
            "umap_embedding_variance",
            scale_floor=max(float(np.var(data_scaled, axis=0).sum()), np.finfo(float).eps),
        ),
        seed=random_state,
        study_id=f"preprocessing-umap-{Path(out_prefix).name}",
        provenance_path=study_artifact,
        catch=(Exception,),
    )
    
    if not study.best_params:
        print("UMAP optimization failed. Using default parameters.")
        best_params = {}
    else:
        best_params = study.best_params
    
    if 'n_neighbors' in best_params:
        best_params['n_neighbors'] = min(max_neighbors, max(2, best_params['n_neighbors']))
    umap_model = _build_umap(
        data_scaled,
        n_components=dims,
        n_neighbors=int(best_params.get("n_neighbors", 15)),
        min_dist=float(best_params.get("min_dist", 0.1)),
        random_state=random_state,
    )
    umap_result = umap_model.fit_transform(data_scaled)
    plot_pairplot(umap_result, out_prefix, labels, "umap", dims, label_type)
    df_result = pd.DataFrame(umap_result, columns=[f"UMAP{i+1}" for i in range(dims)])
    if sample_ids is not None:
        df_result['SampleID'] = sample_ids
    df_result['Label'] = labels
    df_result.to_csv(f"{out_prefix}_umap_result.csv", index=False)
    print(f"Best UMAP params: {best_params}")
    return study_artifact

def perform_tsne(data, out_prefix, labels, sample_ids, dims=3, random_state=DEFAULT_RANDOM_STATE, label_type='categorical'):
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
            raise
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=random_state))
    study_artifact = Path(f"{out_prefix}_tsne_optuna_study.json").resolve()
    run_adaptive_study(
        study,
        objective,
        task="preprocessing",
        model="t-SNE",
        study_scope="full_input_embedding",
        objective_name="tsne_embedding_variance",
        direction="maximize",
        tolerance=tolerance_for_objective(
            "tsne_embedding_variance",
            scale_floor=max(float(np.var(data_scaled, axis=0).sum()), np.finfo(float).eps),
        ),
        seed=random_state,
        study_id=f"preprocessing-tsne-{Path(out_prefix).name}",
        provenance_path=study_artifact,
        catch=(Exception,),
    )
    
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
    return study_artifact

def _fit_kernel_pca(data, dims, kernel, gamma, random_state):
    params = {
        "n_components": dims,
        "kernel": kernel,
        "gamma": gamma,
    }
    try:
        return KernelPCA(**params).fit_transform(data)
    except ValueError as exc:
        if "zero-size array to reduction operation maximum" not in str(exc):
            raise
        return KernelPCA(
            **params,
            eigen_solver="arpack",
            random_state=random_state,
        ).fit_transform(data)


def perform_kernel_pca(data, out_prefix, labels, sample_ids, dims=3, random_state=DEFAULT_RANDOM_STATE, label_type='categorical'):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    def objective(trial):
        try:
            gamma = trial.suggest_float('gamma', 1e-3, 1e-1, log=True)
            kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'cosine'])
            kpca_result = _fit_kernel_pca(data_scaled, dims, kernel, gamma, random_state)
            score = kpca_result.var(axis=0).sum()
            if not np.isfinite(score):
                raise ValueError("Non-finite score")
            return score
        except Exception as e:
            print(f"KernelPCA trial failed with parameters: gamma={locals().get('gamma', None)}, kernel={locals().get('kernel', None)}. Error: {e}")
            raise
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=random_state))
    study_artifact = Path(f"{out_prefix}_kpca_optuna_study.json").resolve()
    run_adaptive_study(
        study,
        objective,
        task="preprocessing",
        model="KernelPCA",
        study_scope="full_input_embedding",
        objective_name="kernel_pca_embedding_variance",
        direction="maximize",
        tolerance=tolerance_for_objective(
            "kernel_pca_embedding_variance",
            scale_floor=max(float(np.var(data_scaled, axis=0).sum()), np.finfo(float).eps),
        ),
        seed=random_state,
        study_id=f"preprocessing-kernel-pca-{Path(out_prefix).name}",
        provenance_path=study_artifact,
        catch=(Exception,),
    )
    
    if not study.best_params:
        print("KernelPCA optimization failed. Using default parameters.")
        best_params = {}
    else:
        best_params = study.best_params
    
    try:
        kpca_result = _fit_kernel_pca(
            data_scaled,
            dims,
            best_params.get("kernel", "linear"),
            best_params.get("gamma"),
            random_state,
        )
        kpca_result = StandardScaler().fit_transform(kpca_result)
    except Exception as e:
        print(f"KernelPCA fitting failed with best parameters: {best_params}. Error: {e}")
        return study_artifact
    
    plot_pairplot(kpca_result, out_prefix, labels, "kpca", dims, label_type)
    df_result = pd.DataFrame(kpca_result, columns=[f"KPCA{i+1}" for i in range(dims)])
    if sample_ids is not None:
        df_result['SampleID'] = sample_ids
    df_result['Label'] = labels
    df_result.to_csv(f"{out_prefix}_kpca_result.csv", index=False)
    print(f"Best KPCA params: {best_params}")
    return study_artifact

def perform_pls(data, out_prefix, labels, sample_ids, dims=3, random_state=DEFAULT_RANDOM_STATE, label_type='categorical'):
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

def perform_elastic_net(data, out_prefix, labels, sample_ids, random_state=DEFAULT_RANDOM_STATE, label_type='categorical'):
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
            raise
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=random_state))
    study_artifact = Path(f"{out_prefix}_elasticnet_optuna_study.json").resolve()
    run_adaptive_study(
        study,
        objective,
        task="preprocessing",
        model="ElasticNet",
        study_scope="full_input_feature_selection",
        objective_name="elastic_net_neg_mean_squared_error",
        direction="maximize",
        tolerance=tolerance_for_objective(
            "elastic_net_neg_mean_squared_error",
            scale_floor=max(float(np.nanvar(np.asarray(y, dtype=float))), np.finfo(float).eps),
        ),
        seed=random_state,
        study_id=f"preprocessing-elastic-net-{Path(out_prefix).name}",
        provenance_path=study_artifact,
        catch=(Exception,),
    )
    
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
        return study_artifact
    
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
    return study_artifact

def perform_boruta(data, out_prefix, labels, sample_ids, random_state=DEFAULT_RANDOM_STATE, label_type='categorical'):
    selector = BorutaProteinSelector(
        task='regression' if label_type == 'continuous' else 'classification',
        random_state=random_state,
    )
    y = pd.to_numeric(labels, errors='coerce').values if label_type == 'continuous' else LabelEncoder().fit_transform(labels)
    selector.fit(data, y)
    support = selector.get_support()
    if not support.any():
        support[:] = True
    selected_feature_names = data.columns[support]
    selected_scaled = StandardScaler().fit_transform(data[selected_feature_names])
    selected_features_df = pd.DataFrame(selected_scaled, columns=selected_feature_names)
    if sample_ids is not None:
        selected_features_df['SampleID'] = sample_ids.values
    selected_features_df['Label'] = labels.values
    selected_features_df.to_csv(f'{out_prefix}_boruta_result.csv', index=False)

    ranking_df = pd.DataFrame({
        'Feature': data.columns,
        'Selected': support,
        'Boruta_Rank': selector.ranking_,
    }).sort_values(['Selected', 'Boruta_Rank'], ascending=[False, True])
    ranking_df.to_csv(f'{out_prefix}_boruta_feature_ranking.csv', index=False)

def perform_dimensionality_reduction(file_path, file_prefix, method, dims=3, standardize=True, random_state=DEFAULT_RANDOM_STATE):
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
    
    study_artifact = None
    if method == 'pca':
        perform_pca(data_clean, file_prefix, labels_clean, sample_ids_clean, dims, random_state, label_type)
    elif method == 'umap':
        study_artifact = perform_umap(data_clean, file_prefix, labels_clean, sample_ids_clean, dims, random_state, label_type)
    elif method == 'tsne':
        study_artifact = perform_tsne(data_clean, file_prefix, labels_clean, sample_ids_clean, dims, random_state, label_type)
    elif method == 'kpca':
        study_artifact = perform_kernel_pca(data_clean, file_prefix, labels_clean, sample_ids_clean, dims, random_state, label_type)
    elif method == 'pls':
        perform_pls(data_clean, file_prefix, labels_clean, sample_ids_clean, dims, random_state, label_type)
    elif method == 'elasticnet':
        if label_type == 'categorical':
            print("Warning: ElasticNet is being applied to categorical labels treated as numeric.")
        study_artifact = perform_elastic_net(data_clean, file_prefix, labels_clean, sample_ids_clean, random_state, label_type)
    elif method == 'boruta':
        perform_boruta(data_clean, file_prefix, labels_clean, sample_ids_clean, random_state, label_type)
    else:
        raise ValueError("Unsupported method.")

    return study_artifact

if __name__ == "__main__":
    args = parse_arguments()
    set_global_seed(args.seed)
    prefix = Path(args.csv).stem
    if args.prefix:
        prefix = args.prefix
    methods = args.methods
    dims = args.dimensions

    with task_run_context(args.outdir, "preprocess"):
        results = Parallel(n_jobs=args.jobs, backend='multiprocessing', verbose=100)(
            delayed(perform_dimensionality_reduction)(
                args.csv,
                prefix,
                method=method_name,
                dims=dims,
                standardize=True,
                random_state=args.seed
            ) for method_name in methods
        )
