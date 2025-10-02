import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, label_binarize, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.cross_decomposition import PLSRegression
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
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
import umap
from scipy.sparse.linalg import ArpackError

# Suppress specific warnings except ConvergenceWarning
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)


def enlarge_fonts(ax, label=22, ticks=18, title=24, legend=20, legend_title=22):
    ax.xaxis.label.set_size(label)
    ax.yaxis.label.set_size(label)
    if hasattr(ax, "title"):
        ax.title.set_size(title)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontsize(ticks)
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


def multiclass_specificity(cm: np.ndarray) -> float:
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


class PLSDAClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=2, max_iter=1000):
        self.n_components = n_components
        self.max_iter = max_iter
        self.pls = None
        self.classes_ = None

    def fit(self, X, y):
        # y is expected to be encoded integers (0..n_classes-1)
        self.classes_ = np.unique(y)
        y_onehot = label_binarize(y, classes=self.classes_)
        if y_onehot.ndim == 1:
            y_onehot = np.vstack([1 - y_onehot, y_onehot]).T
        # clamp n_components to valid range
        n_targets = y_onehot.shape[1]
        max_comp = max(1, min(X.shape[0] - 1, X.shape[1], n_targets))
        n_comp = max(1, min(self.n_components, max_comp))
        self.pls = PLSRegression(n_components=n_comp, max_iter=self.max_iter, tol=1e-06)
        self.pls.fit(X, y_onehot)
        return self

    def predict(self, X):
        y_pred_cont = self.pls.predict(X)
        if y_pred_cont.shape[1] > 1:
            y_idx = np.argmax(y_pred_cont, axis=1)
            return self.classes_[y_idx]
        else:
            y_bin = (y_pred_cont >= 0.5).astype(int).ravel()
            return self.classes_[y_bin]

    def predict_proba(self, X):
        y_pred_cont = self.pls.predict(X)
        y_pred_proba = np.maximum(y_pred_cont, 0.0)
        eps = 1e-12
        if y_pred_proba.ndim == 1 or y_pred_proba.shape[1] == 1:
            p1 = np.clip(y_pred_proba.ravel(), 0.0, 1.0)
            p0 = 1.0 - p1
            probs = np.vstack([p0, p1]).T
            probs = np.clip(probs, 0.0, None)
            s = probs.sum(axis=1, keepdims=True)
            s = np.where(s <= eps, 1.0, s)
            probs = probs / s
            return probs
        else:
            probs = y_pred_proba
            s = probs.sum(axis=1, keepdims=True)
            zero_mask = (s <= eps).ravel()
            s = np.clip(s, eps, None)
            probs = probs / s
            if np.any(zero_mask):
                k = probs.shape[1]
                probs[zero_mask, :] = 1.0 / k
            return probs


class ElasticNetFeatureSelector(BaseEstimator, TransformerMixin):
    # LogisticRegression with elastic-net penalty for classification feature selection
    def __init__(self, C=1.0, l1_ratio=0.5, max_iter=10000, tol=1e-4):
        self.C = C
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.selector = SelectFromModel(
            LogisticRegression(
                penalty='elasticnet',
                solver='saga',
                l1_ratio=self.l1_ratio,
                C=self.C,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=1234,
                class_weight='balanced'
            )
        )

    def fit(self, X, y):
        self.selector.fit(X, y)
        return self

    def transform(self, X):
        return self.selector.transform(X)


def safe_umap(n_components, n_neighbors, min_dist, X, random_state=1234):
    n_samples = X.shape[0]
    n_components = min(n_components, max(1, min(n_samples - 1, X.shape[1])))
    n_neighbors = min(n_neighbors, max(2, n_samples - 2))
    return umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        init='random'
    )


def max_pls_components(X, y_encoded):
    n_classes = len(np.unique(y_encoded))
    upper = max(1, min(X.shape[0] - 1, X.shape[1], max(1, n_classes - 1)))
    return upper


def plsda_nested_cv(inp, prefix, feature_selection_method):
    data = pd.read_csv(inp)

    if 'SampleID' not in data.columns or 'Label' not in data.columns:
        raise ValueError("Input data must contain 'SampleID' and 'Label' columns.")

    sample_ids = data['SampleID']
    X = data.drop(columns=['SampleID', 'Label']).copy()
    y = data['Label']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_binarized = label_binarize(y_encoded, classes=np.unique(y_encoded))
    if y_binarized.ndim == 1:
        y_binarized = np.vstack([1 - y_binarized, y_binarized]).T
    num_classes = len(np.unique(y_encoded))

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
            steps = [('scaler', StandardScaler())]

            if feature_selection_method == 'elasticnet':
                lr_C = trial.suggest_loguniform('lr_C', 1e-3, 1e3)
                lr_l1 = trial.suggest_uniform('lr_l1_ratio', 0.0, 1.0)
                steps.append(('feature_selection', ElasticNetFeatureSelector(
                    C=lr_C,
                    l1_ratio=lr_l1,
                    max_iter=10000,
                    tol=1e-4
                )))
            elif feature_selection_method == 'umap':
                umap_n_components = trial.suggest_int(
                    'umap_n_components',
                    2,
                    min(100, X_train_outer.shape[1], X_train_outer.shape[0] - 1)
                )
                umap_n_neighbors = trial.suggest_int(
                    'umap_n_neighbors',
                    5,
                    min(50, X_train_outer.shape[0] - 1)
                )
                umap_min_dist = trial.suggest_uniform('umap_min_dist', 0.0, 0.99)
                steps.append(('feature_selection', safe_umap(
                    n_components=umap_n_components,
                    n_neighbors=umap_n_neighbors,
                    min_dist=umap_min_dist,
                    X=X_train_outer
                )))

            max_comp = max_pls_components(X_train_outer, y_train_outer)
            plsda_n_components = trial.suggest_int('plsda_n_components', 1, max_comp)
            steps.append(('plsda', PLSDAClassifier(n_components=plsda_n_components, max_iter=1000)))

            pipeline = Pipeline(steps)

            with SuppressOutput():
                f1_scores = []
                for inner_train_idx, inner_valid_idx in cv_inner.split(X_train_outer, y_train_outer):
                    X_train_inner, X_valid_inner = X_train_outer.iloc[inner_train_idx], X_train_outer.iloc[inner_valid_idx]
                    y_train_inner, y_valid_inner = y_train_outer[inner_train_idx], y_train_outer[inner_valid_idx]
                    try:
                        pipeline.fit(X_train_inner, y_train_inner)
                        y_pred_inner = pipeline.predict(X_valid_inner)
                        f1 = f1_score(y_valid_inner, y_pred_inner, average='weighted')
                        f1_scores.append(f1)
                    except (ValueError, ArpackError):
                        f1_scores.append(0.0)
                return np.mean(f1_scores)

        study_inner = optuna.create_study(direction='maximize', sampler=TPESampler(seed=1234))
        study_inner.optimize(objective_inner, n_trials=50, show_progress_bar=False)

        best_params_inner = study_inner.best_params

        steps = [('scaler', StandardScaler())]

        if feature_selection_method == 'elasticnet':
            best_lr_C = best_params_inner.get('lr_C', 1.0)
            best_lr_l1 = best_params_inner.get('lr_l1_ratio', 0.5)
            steps.append(('feature_selection', ElasticNetFeatureSelector(
                C=best_lr_C,
                l1_ratio=best_lr_l1,
                max_iter=10000,
                tol=1e-4
            )))
        elif feature_selection_method == 'umap':
            best_umap_n_components = best_params_inner.get('umap_n_components', 2)
            best_umap_n_neighbors = best_params_inner.get('umap_n_neighbors', 15)
            best_umap_min_dist = best_params_inner.get('umap_min_dist', 0.1)
            steps.append(('feature_selection', safe_umap(
                n_components=best_umap_n_components,
                n_neighbors=best_umap_n_neighbors,
                min_dist=best_umap_min_dist,
                X=X_train_outer
            )))

        best_plsda_n_components = best_params_inner.get('plsda_n_components', 2)
        best_plsda_n_components = max(1, min(best_plsda_n_components, max_pls_components(X_train_outer, y_train_outer)))
        steps.append(('plsda', PLSDAClassifier(n_components=best_plsda_n_components, max_iter=1000)))

        best_model_inner = Pipeline(steps)

        with SuppressOutput():
            try:
                best_model_inner.fit(X_train_outer, y_train_outer)
            except (ValueError, ArpackError) as e:
                print(f"Error fitting the model in outer fold {fold_idx}: {e}")
                outer_f1_scores.append(0)
                outer_auc_scores.append(0)
                fold_idx += 1
                continue

        try:
            y_pred_prob_outer = best_model_inner.predict_proba(X_test_outer)
            y_pred_class_outer = best_model_inner.predict(X_test_outer)
        except (ValueError, ArpackError) as e:
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
                if y_binarized_test.ndim == 1:
                    y_binarized_test = np.vstack([1 - y_binarized_test, y_binarized_test]).T
                fpr, tpr, _ = roc_curve(y_binarized_test.ravel(), y_pred_prob_outer.ravel())
                auc_outer = auc(fpr, tpr)
            except ValueError:
                auc_outer = 0.0
        outer_auc_scores.append(auc_outer)

        print(f"Fold {fold_idx} - F1 Score: {f1_outer:.4f}, AUC: {auc_outer:.4f}")
        fold_idx += 1

    plt.figure(figsize=(10, 6))
    folds = range(1, cv_outer.get_n_splits() + 1)
    plt.plot(folds, outer_f1_scores, marker='o', linestyle='-', label='F1 Score')
    plt.plot(folds, outer_auc_scores, marker='s', linestyle='-', label='AUC Score')
    plt.title('F1 and AUC Scores per Outer Fold', fontsize=26, fontweight='bold', pad=14)
    plt.xlabel('Outer Fold Number', fontsize=22, labelpad=12)
    plt.ylabel('Score (F1 / AUC)', fontsize=22, labelpad=12)
    plt.xticks(folds, fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18, title_fontsize=20)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    ax_cv = plt.gca()
    enlarge_fonts(ax_cv)
    plt.tight_layout()
    plt.savefig(f"{prefix}_plsda_nested_cv_f1_auc.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Nested cross-validation completed.")
    print(f"Average F1 Score: {np.mean(outer_f1_scores):.4f} ± {np.std(outer_f1_scores):.4f}")
    print(f"Average AUC: {np.mean(outer_auc_scores):.4f} ± {np.std(outer_auc_scores):.4f}")

    print("Starting hyperparameter tuning on the entire dataset...")

    def objective_full(trial):
        steps = [('scaler', StandardScaler())]

        if feature_selection_method == 'elasticnet':
            lr_C = trial.suggest_loguniform('lr_C', 1e-3, 1e3)
            lr_l1 = trial.suggest_uniform('lr_l1_ratio', 0.0, 1.0)
            steps.append(('feature_selection', ElasticNetFeatureSelector(
                C=lr_C,
                l1_ratio=lr_l1,
                max_iter=10000,
                tol=1e-4
            )))
        elif feature_selection_method == 'umap':
            umap_n_components = trial.suggest_int(
                'umap_n_components',
                2,
                min(100, X.shape[1], X.shape[0] - 1)
            )
            umap_n_neighbors = trial.suggest_int(
                'umap_n_neighbors',
                5,
                min(50, X.shape[0] - 1)
            )
            umap_min_dist = trial.suggest_uniform('umap_min_dist', 0.0, 0.99)
            steps.append(('feature_selection', safe_umap(
                n_components=umap_n_components,
                n_neighbors=umap_n_neighbors,
                min_dist=umap_min_dist,
                X=X
            )))

        max_comp_full = max_pls_components(X, y_encoded)
        plsda_n_components = trial.suggest_int('plsda_n_components', 1, max_comp_full)
        steps.append(('plsda', PLSDAClassifier(n_components=plsda_n_components, max_iter=1000)))

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
                except (ValueError, ArpackError):
                    f1_scores.append(0.0)
            return np.mean(f1_scores)

    study_full = optuna.create_study(direction='maximize', sampler=TPESampler(seed=1234))
    study_full.optimize(objective_full, n_trials=50, show_progress_bar=True)

    best_params_full = study_full.best_params
    print(f"Best parameters for PLSDA: {best_params_full}")

    steps = [('scaler', StandardScaler())]

    if feature_selection_method == 'elasticnet':
        best_lr_C_full = best_params_full.get('lr_C', 1.0)
        best_lr_l1_full = best_params_full.get('lr_l1_ratio', 0.5)
        steps.append(('feature_selection', ElasticNetFeatureSelector(
            C=best_lr_C_full,
            l1_ratio=best_lr_l1_full,
            max_iter=10000,
            tol=1e-4
        )))
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

    best_plsda_n_components_full = best_params_full.get('plsda_n_components', 2)
    best_plsda_n_components_full = max(1, min(best_plsda_n_components_full, max_pls_components(X, y_encoded)))
    steps.append(('plsda', PLSDAClassifier(n_components=best_plsda_n_components_full, max_iter=1000)))

    best_model = Pipeline(steps)

    with SuppressOutput():
        try:
            best_model.fit(X, y_encoded)
        except (ValueError, ArpackError) as e:
            print(f"Error fitting the final model: {e}")
            sys.exit(1)

    joblib.dump(best_model, f"{prefix}_plsda_model.pkl")
    joblib.dump((X, y_encoded, le), f"{prefix}_plsda_data.pkl")

    print(f"Best parameters for PLSDA: {best_params_full}")

    if feature_selection_method != 'none':
        if 'feature_selection' in best_model.named_steps:
            try:
                X_transformed = best_model.named_steps['feature_selection'].transform(X)
            except (NotImplementedError, ArpackError, ValueError):
                X_transformed = None
        else:
            X_transformed = None

        if X_transformed is not None:
            if feature_selection_method == 'umap':
                n_components = X_transformed.shape[1]
                transformed_columns = [f"UMAP_Component_{i+1}" for i in range(n_components)]
                X_transformed_df = pd.DataFrame(X_transformed, columns=transformed_columns)
            elif feature_selection_method == 'elasticnet':
                selected_features = X.columns[best_model.named_steps['feature_selection'].selector.get_support()]
                X_transformed_df = X[selected_features].copy()
            else:
                X_transformed_df = pd.DataFrame(X_transformed)

            X_transformed_df.insert(0, 'SampleID', sample_ids)
            X_transformed_df['Label'] = y
            transformed_csv_path = f"{prefix}_plsda_transformed_X.csv"
            X_transformed_df.to_csv(transformed_csv_path, index=False)
            print(f"Transformed data saved to {transformed_csv_path}")

            variance_csv_path = f"{prefix}_plsda_variance.csv"
            if feature_selection_method == 'elasticnet':
                with open(variance_csv_path, 'w') as f:
                    f.write("ElasticNet does not provide explained variance information.\n")
                print(f"No variance information available for ElasticNet. File created at {variance_csv_path}")
            elif feature_selection_method == 'umap':
                with open(variance_csv_path, 'w') as f:
                    f.write("UMAP does not provide explained variance information.\n")
                print(f"No variance information available for UMAP. File created at {variance_csv_path}")
            else:
                with open(variance_csv_path, 'w') as f:
                    f.write(f"{feature_selection_method.upper()} does not provide explained variance information.\n")
                print(f"No variance information available for {feature_selection_method.upper()}. File created at {variance_csv_path}")
        else:
            print("Transformed data is not available.")
    else:
        print("No feature selection method selected. Skipping transformed data and variance information saving.")

    try:
        y_pred_prob = cross_val_predict(best_model, X, y_encoded, cv=cv_outer, method='predict_proba', n_jobs=-1)
        y_pred_class = np.argmax(y_pred_prob, axis=1)
    except (ValueError, ArpackError) as e:
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
            sensitivity = np.mean(
                np.divide(
                    np.diag(cm),
                    np.sum(cm, axis=1),
                    out=np.zeros_like(np.diag(cm), dtype=float),
                    where=np.sum(cm, axis=1) != 0
                )
            )
        specificity = multiclass_specificity(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for PLSDA', fontsize=16, fontweight='bold', pad=12)
    enlarge_fonts(disp.ax_)
    plt.savefig(f"{prefix}_plsda_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    fpr = {}
    tpr = {}
    roc_auc = {}

    if num_classes == 2:
        try:
            fpr[0], tpr[0], _ = roc_curve(y_encoded, y_pred_prob[:, 1])
            roc_auc[0] = auc(fpr[0], tpr[0])
        except ValueError:
            roc_auc[0] = 0.0
    else:
        for i in range(y_binarized.shape[1]):
            try:
                fpr[i], tpr[i], _ = roc_curve(y_binarized[:, i], y_pred_prob[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            except ValueError:
                fpr[i], tpr[i], roc_auc[i] = np.array([0, 1]), np.array([0, 1]), 0.0

        try:
            fpr["micro"], tpr["micro"], _ = roc_curve(y_binarized.ravel(), y_pred_prob.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        except ValueError:
            fpr["micro"], tpr["micro"], roc_auc["micro"] = np.array([0, 1]), np.array([0, 1]), 0.0

        try:
            class_keys = [k for k in fpr.keys() if isinstance(k, (int, np.integer))]
            valid_keys = [k for k in class_keys if len(fpr[k]) > 1 and len(tpr[k]) > 1]
            if len(valid_keys) > 0:
                all_fpr = np.unique(np.concatenate([fpr[k] for k in valid_keys]))
                mean_tpr = np.zeros_like(all_fpr)
                for k in valid_keys:
                    mean_tpr += np.interp(all_fpr, fpr[k], tpr[k])
                mean_tpr /= len(valid_keys)
                fpr["macro"] = all_fpr
                tpr["macro"] = mean_tpr
                roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
            else:
                fpr["macro"], tpr["macro"], roc_auc["macro"] = np.array([0, 1]), np.array([0, 1]), 0.0
        except Exception:
            fpr["macro"], tpr["macro"], roc_auc["macro"] = np.array([0, 1]), np.array([0, 1]), 0.0

    roc_data = {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}
    np.save(f"{prefix}_plsda_roc_data.npy", roc_data, allow_pickle=True)

    plt.figure(figsize=(10, 8))
    if num_classes == 2:
        plt.plot(fpr[0], tpr[0], label=f'AUC = {roc_auc[0]:.2f}')
    else:
        for i in range(len(le.classes_)):
            if i in roc_auc and roc_auc[i] > 0.0:
                plt.plot(fpr[i], tpr[i], label=f'{le.inverse_transform([i])[0]} (AUC = {roc_auc[i]:.2f})')
        if "micro" in roc_auc and roc_auc["micro"] > 0.0:
            plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})', linestyle='--')
        if "macro" in roc_auc and roc_auc["macro"] > 0.0:
            plt.plot(fpr["macro"], tpr["macro"], label=f'Macro-average (AUC = {roc_auc["macro"]:.2f})', linestyle=':')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=22, labelpad=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=22, labelpad=12)
    plt.title('ROC Curves for PLSDA', fontsize=26, fontweight='bold', pad=14)
    plt.legend(loc="lower right", fontsize=18, title_fontsize=20)
    ax_roc = plt.gca()
    enlarge_fonts(ax_roc)
    plt.tight_layout()
    plt.savefig(f'{prefix}_plsda_roc_curve.png', dpi=300, bbox_inches="tight")
    plt.close()

    metrics = {'Accuracy': acc, 'F1 Score': f1, 'Sensitivity': sensitivity, 'Specificity': specificity}
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    ax = metrics_df.plot(kind='bar', x='Metric', y='Value', legend=False)
    plt.title('Performance Metrics for PLSDA', fontsize=20, fontweight='bold', pad=10)
    plt.ylabel('Value', fontsize=22, labelpad=12)
    plt.ylim(0, 1.1)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=5)
    plt.xticks(rotation=45, ha='right', fontsize=18)
    ax.tick_params(axis='y', labelsize=18)
    enlarge_fonts(ax)
    plt.tight_layout()
    plt.savefig(f'{prefix}_plsda_metrics.png', dpi=300, bbox_inches="tight")
    plt.close()

    predictions_df = pd.DataFrame({
        'SampleID': sample_ids,
        'Original Label': y,
        'Predicted Label': le.inverse_transform(y_pred_class)
    })
    predictions_df.to_csv(f"{prefix}_plsda_predictions.csv", index=False)
    print(f"Predictions saved to {prefix}_plsda_predictions.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run PLSDA with Nested Cross-Validation, Feature Selection (UMAP or ElasticNet), and Optuna hyperparameter optimization.'
    )
    parser.add_argument('-i', type=str, required=True, help='Input file in CSV format')
    parser.add_argument('-p', type=str, required=True, help='Output prefix')
    parser.add_argument('-f', type=str, choices=['none', 'elasticnet', 'umap'], default='none',
                        help='Feature selection method to use. Options: none, elasticnet, umap.')
    args = parser.parse_args()

    plsda_nested_cv(args.i, args.p, args.f)
