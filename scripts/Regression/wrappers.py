import umap
from sklearn.manifold import TSNE
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class UMAPPipelineWrapper(BaseEstimator, TransformerMixin):
    """
    A wrapper for UMAP so it can be used as a transformer within a Pipeline,
    allowing usage in inner CV. UMAP supports transform on new data, but
    we store the fitted model in _umap.
    """
    def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1, random_state=1234):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state
        self._umap = None

    def fit(self, X, y=None):
        self._umap = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            random_state=self.random_state
        )
        self._umap.fit(X)
        return self

    def transform(self, X):
        if self._umap is None:
            raise ValueError("UMAPPipelineWrapper is not fitted.")
        return self._umap.transform(X)

    @property
    def n_components_(self):
        return self.n_components


class TSNEPipelineWrapper(BaseEstimator, TransformerMixin):
    """
    A wrapper for t-SNE so it can be used as a transformer within a Pipeline,
    allowing usage in inner CV. This wrapper fits t-SNE on the input data and
    transforms it accordingly, ensuring the number of samples remains consistent.
    """
    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200.0, max_iter=1000, random_state=1234):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self._tsne = None

    def fit(self, X, y=None):
        self._tsne = TSNE(
            n_components=self.n_components,
            perplexity=self.perplexity,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        self._tsne.fit(X)
        return self

    def transform(self, X):
        if self._tsne is None:
            raise ValueError("TSNEPipelineWrapper is not fitted.")
        return self._tsne.fit_transform(X)

    @property
    def n_components_(self):
        return self.n_components
