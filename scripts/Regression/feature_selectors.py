from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import ElasticNet
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
import umap

# Custom PLS Feature Selector
class PLSFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, max_iter=1000, tol=1e-06):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.pls = None

    def fit(self, X, y):
        self.pls = PLSRegression(n_components=self.n_components, max_iter=self.max_iter, tol=self.tol)
        self.pls.fit(X, y)
        return self

    def transform(self, X):
        return self.pls.transform(X)


# ElasticNet Feature Selector
class ElasticNetFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=10000, tol=1e-4):
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


# Custom TSNE Transformer (Not used within Pipeline)
class TSNETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, perplexity=30, learning_rate=200.0, max_iter=1000, random_state=1234):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.tsne = None
        self.embedding_ = None

    def fit(self, X, y=None):
        self.tsne = TSNE(n_components=self.n_components,
                         perplexity=self.perplexity,
                         learning_rate=self.learning_rate,
                         max_iter=self.max_iter,
                         random_state=self.random_state)
        self.embedding_ = self.tsne.fit_transform(X)
        self.train_shape_ = X.shape
        return self

    def transform(self, X):
        if X.shape == self.train_shape_:
            return self.embedding_
        else:
            raise RuntimeError("TSNETransformer cannot transform new/unseen data. Use on the same training data only.")
