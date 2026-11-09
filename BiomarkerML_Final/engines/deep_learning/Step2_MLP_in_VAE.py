import argparse
import random
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import optuna
import joblib
from sklearn.utils.class_weight import compute_class_weight

SCRIPT_ROOT = str(Path(__file__).resolve().parents[1])
if SCRIPT_ROOT not in sys.path:
    sys.path.insert(0, SCRIPT_ROOT)
from core.data_contracts import (
    RoleAwarePreprocessor, modeling_frame, resolve_feature_roles,
    write_missingness_qc, write_reproducibility_manifest, imputation_fallback_reasons,
)
from core.classification_metrics import sensitivity_specificity_from_confusion_matrix
from core.adaptive_optuna import run_adaptive_study, tolerance_for_objective
from core.publication_utils import (
    DEFAULT_RANDOM_STATE,
    build_output_prefix,
    configure_publication_plots,
    SuppressOutput,
    EarlyStopping,
    plot_classification_nested_cv,
    plot_confusion_matrix,
    plot_metric_bars,
    plot_roc_curves,
    task_run_context,
    export_outer_fold_biomarkers,
    estimator_importance_with_source,
    generate_training_size_curve,
    write_training_history,
)

configure_publication_plots()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Script to run classifiers')
    parser.add_argument('--input', '-i', '--csv', dest='csv', type=str, help='Input CSV file', required=True)
    parser.add_argument('-p', '--prefix', type=str, help='Output prefix')
    parser.add_argument('--outdir', default='.', help='Base output directory')
    parser.add_argument('--feature_selection', choices=['yes', 'no'], default='no', help='Accepted for CLI consistency')
    parser.add_argument('--jobs', type=int, default=1, help='Accepted for CLI consistency')
    parser.add_argument('--seed', type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument('--imputation', choices=['median', 'knn'], default='median')
    parser.add_argument('--covariates', nargs='*', default=None)
    parser.add_argument('--categorical-covariates', nargs='*', default=None)
    return parser.parse_args()

def set_seed(seed=DEFAULT_RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()
warnings.filterwarnings('ignore')

TITLE_FONTSIZE = 22
LABEL_FONTSIZE = 18
TICK_FONTSIZE = 14
LEGEND_FONTSIZE = 14

class NormLayer(nn.Module):
    def __init__(self, norm_type: str, units: int):
        super().__init__()
        norm_type = (norm_type or 'none').lower()
        if norm_type == 'batch':
            self.norm = nn.BatchNorm1d(units)
        elif norm_type == 'layer':
            self.norm = nn.LayerNorm(units)
        else:
            self.norm = nn.Identity()
    def forward(self, x):
        return self.norm(x)

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, encoder_layers, enc_dropout, enc_norm):
        super(Encoder, self).__init__()
        layers = []
        prev_dim = input_dim
        for units in encoder_layers:
            layers.append(nn.Linear(prev_dim, units))
            layers.append(nn.LeakyReLU())
            layers.append(NormLayer(enc_norm, units))
            layers.append(nn.Dropout(enc_dropout))
            prev_dim = units
        layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*layers)
    def forward(self, x):
        return self.encoder(x)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, mlp_layers, mlp_dropout, mlp_norm):
        super(MLPClassifier, self).__init__()
        layers = []
        prev_dim = input_dim
        for units in mlp_layers:
            layers.append(nn.Linear(prev_dim, units))
            layers.append(nn.LeakyReLU())
            layers.append(NormLayer(mlp_norm, units))
            layers.append(nn.Dropout(mlp_dropout))
            prev_dim = units
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)

class VAE_MLP(BaseEstimator, ClassifierMixin):
    """
    Encoder + MLP classifier (no decoder). The encoder reduces dimensionality;
    the MLP classifies on the latent space.
    """
    def __init__(self,
                 input_dim=30,
                 num_classes=2,
                 latent_dim=2,
                 encoder_layers=[64, 32],
                 mlp_layers=[32, 16],
                 enc_dropout=0.5,
                 mlp_dropout=0.2,
                 enc_norm='layer',
                 mlp_norm='layer',
                 early_stopping_metric='loss',  # 'loss' or 'macro_f1'
                 early_stopping_patience=10,
                 early_stopping_min_delta=0.0,
                 encoder_learning_rate=0.001,
                 mlp_learning_rate=0.001,
                 enc_weight_decay=1e-5,
                 mlp_weight_decay=1e-5,
                 scheduler_factor=0.5,
                 scheduler_patience=5,
                 scheduler_min_lr=1e-6,
                 max_grad_norm=None,
                 epochs=50,
                 batch_size=32):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.encoder_layers = encoder_layers
        self.mlp_layers = mlp_layers
        self.enc_dropout = enc_dropout
        self.mlp_dropout = mlp_dropout
        self.enc_norm = enc_norm
        self.mlp_norm = mlp_norm
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.encoder_learning_rate = encoder_learning_rate
        self.mlp_learning_rate = mlp_learning_rate
        self.enc_weight_decay = enc_weight_decay
        self.mlp_weight_decay = mlp_weight_decay
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.scheduler_min_lr = scheduler_min_lr
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes_ = None
        self.encoder = None
        self.classifier = None
        self.optimizer = None
        self.scheduler = None

    def create_models(self):
        self.encoder = Encoder(self.input_dim, self.latent_dim, self.encoder_layers,
                               self.enc_dropout, self.enc_norm).to(self.device)
        self.classifier = MLPClassifier(self.latent_dim, self.num_classes, self.mlp_layers,
                                        self.mlp_dropout, self.mlp_norm).to(self.device)
        self.optimizer = optim.Adam([
            {'params': self.encoder.parameters(), 'lr': self.encoder_learning_rate, 'weight_decay': self.enc_weight_decay},
            {'params': self.classifier.parameters(), 'lr': self.mlp_learning_rate, 'weight_decay': self.mlp_weight_decay}
        ])
        mode = 'min' if self.early_stopping_metric == 'loss' else 'max'
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode=mode, factor=self.scheduler_factor,
            patience=self.scheduler_patience, min_lr=self.scheduler_min_lr, verbose=False
        )

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.create_models()
        self.training_history_ = []

        # Train/val split for early stopping
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=DEFAULT_RANDOM_STATE, stratify=y
        )
        X_train_t = torch.from_numpy(X_train).float().to(self.device)
        y_train_t = torch.from_numpy(y_train).long().to(self.device)
        X_val_t = torch.from_numpy(X_val).float().to(self.device)
        y_val_t = torch.from_numpy(y_val).long().to(self.device)

        train_batch_size = min(self.batch_size, max(1, len(X_train)))
        dl_train = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train_t, y_train_t),
            batch_size=train_batch_size, shuffle=True, drop_last=True
        )
        dl_val = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_val_t, y_val_t),
            batch_size=self.batch_size, shuffle=False, drop_last=False
        )

        # Class weights aligned with indices 0..num_classes-1
        weights_full = np.ones(self.num_classes, dtype=np.float32)
        classes_present = np.unique(y)
        weights_present = compute_class_weight(class_weight='balanced', classes=classes_present, y=y)
        for idx, cls in enumerate(classes_present):
            weights_full[int(cls)] = float(weights_present[idx])
        assert len(weights_full) == self.num_classes
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights_full, dtype=torch.float32, device=self.device))

        mode = 'min' if self.early_stopping_metric == 'loss' else 'max'
        early_stopping = EarlyStopping(
            patience=self.early_stopping_patience,
            min_delta=self.early_stopping_min_delta,
            mode=mode,
            verbose=False
        )

        self.encoder.train()
        self.classifier.train()
        for epoch in range(self.epochs):
            train_loss_sum = 0.0
            n_train = 0
            for xb, yb in dl_train:
                self.optimizer.zero_grad()
                logits = self.classifier(self.encoder(xb))
                loss = criterion(logits, yb)
                loss.backward()
                if self.max_grad_norm is not None and self.max_grad_norm > 0:
                    clip_grad_norm_(list(self.encoder.parameters()) + list(self.classifier.parameters()),
                                    max_norm=self.max_grad_norm)
                self.optimizer.step()
                bs = yb.size(0)
                train_loss_sum += loss.item() * bs
                n_train += bs
            train_loss = train_loss_sum / max(n_train, 1)

            self.encoder.eval()
            self.classifier.eval()
            val_loss_sum = 0.0
            n_val = 0
            all_pred = []
            all_true = []
            with torch.no_grad():
                for xb, yb in dl_val:
                    logits = self.classifier(self.encoder(xb))
                    loss = criterion(logits, yb)
                    bs = yb.size(0)
                    val_loss_sum += loss.item() * bs
                    n_val += bs
                    preds = torch.argmax(logits, dim=1)
                    all_pred.append(preds.cpu().numpy())
                    all_true.append(yb.cpu().numpy())
            val_loss = val_loss_sum / max(n_val, 1)
            all_pred = np.concatenate(all_pred) if all_pred else np.array([])
            all_true = np.concatenate(all_true) if all_true else np.array([])
            if all_true.size > 0:
                val_macro_f1 = f1_score(all_true, all_pred, average='macro')
            else:
                val_macro_f1 = 0.0

            monitor_value = val_loss if self.early_stopping_metric == 'loss' else val_macro_f1
            self.scheduler.step(monitor_value)
            stopped = early_stopping(
                monitor_value,
                epoch + 1,
                checkpoint_modules=(self.encoder, self.classifier),
            )
            self.training_history_.append({
                "Epoch": epoch + 1,
                "TrainingLoss": train_loss,
                "ValidationLoss": val_loss,
                "LearningRate": self.optimizer.param_groups[0]["lr"],
                "EarlyStoppingTriggered": stopped,
                "BestEpoch": early_stopping.best_epoch,
            })
            if stopped:
                break
            self.encoder.train()
            self.classifier.train()
        early_stopping.restore_checkpoint(self.encoder, self.classifier)
        return self

    def _inference_logits(self, X, inference_batch_size: int | None = None):
        batch_size = int(inference_batch_size or self.batch_size)
        X_tensor = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.device)
        logits = []
        # Match the base-seed draw made by the legacy single inference DataLoader.
        torch.empty((), dtype=torch.int64).random_()
        self.encoder.eval()
        self.classifier.eval()
        with torch.inference_mode():
            for start in range(0, X_tensor.shape[0], batch_size):
                logits.append(self.classifier(self.encoder(X_tensor[start:start + batch_size])).cpu())
        if not logits:
            return torch.empty((0, self.num_classes), dtype=torch.float32)
        return torch.cat(logits, dim=0)

    def predict(self, X, inference_batch_size: int | None = None):
        logits = self._inference_logits(X, inference_batch_size)
        return torch.argmax(logits, dim=1).numpy() if len(logits) else np.array([])

    def predict_proba(self, X, inference_batch_size: int | None = None):
        logits = self._inference_logits(X, inference_batch_size)
        return torch.softmax(logits, dim=1).numpy() if len(logits) else np.empty((0, self.num_classes))

def compute_macro_roc(y_true_bin: np.ndarray, y_score: np.ndarray):
    n_classes = y_true_bin.shape[1]
    fpr_d, tpr_d, auc_d = {}, {}, {}
    for i in range(n_classes):
        yi = y_true_bin[:, i]
        si = y_score[:, i]
        if (yi.sum() == 0) or (yi.sum() == len(yi)):
            fpr_d[i], tpr_d[i] = np.array([0, 1]), np.array([0, 1])
            auc_d[i] = 0.0
        else:
            fpr_i, tpr_i, _ = roc_curve(yi, si)
            fpr_d[i], tpr_d[i] = fpr_i, tpr_i
            auc_d[i] = auc(fpr_i, tpr_i)
    all_fpr = np.unique(np.concatenate([fpr_d[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr_d[i], tpr_d[i])
    mean_tpr /= n_classes
    auc_macro = auc(all_fpr, mean_tpr)
    return all_fpr, mean_tpr, auc_macro, (fpr_d, tpr_d, auc_d)

def vae(inp, prefix, imputation="median", covariates=None, categorical_covariates=None):
    data = pd.read_csv(inp)
    roles = resolve_feature_roles(data, target="Label", identifier="SampleID", covariates=covariates, categorical_covariates=categorical_covariates)
    sample_ids = data['SampleID'].values if 'SampleID' in data.columns else np.asarray([f"Sample_{i+1}" for i in range(len(data))])
    X = modeling_frame(data, roles)
    y = data['Label'].values
    feature_names = list(roles.biomarkers)

    def make_preprocessor():
        return RoleAwarePreprocessor(
            biomarker_columns=roles.biomarkers,
            numeric_covariates=roles.numeric_covariates,
            categorical_covariates=roles.categorical_covariates,
            imputation=imputation,
            selector=None,
        )

    le = LabelEncoder()
    y = le.fit_transform(y)
    num_classes = len(np.unique(y))

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
    outer_fold = 0

    outer_accuracies = []
    outer_f1_scores = []
    outer_weighted_f1_scores = []
    outer_aucs = []
    outer_sensitivities = []
    outer_specificities = []
    outer_cm_total = np.zeros((num_classes, num_classes), dtype=float)

    all_X = []
    all_y_true = []
    all_y_pred = []
    all_y_pred_proba = []
    per_fold_f1 = []
    per_fold_auc = []
    best_params_by_fold = []
    missingness_folds = []
    tuning_reports = []

    # Outer-fold parameters are validation provenance only; final parameters are selected separately.
    latent_dim = 2
    enc_dropout = 0.5
    mlp_dropout = 0.2
    enc_norm = 'layer'
    mlp_norm = 'layer'
    early_stopping_metric_best = 'loss'
    early_stopping_patience = 10
    early_stopping_min_delta = 0.0
    encoder_layers = [64, 32]
    mlp_layers = [32, 16]
    encoder_learning_rate = 1e-3
    mlp_learning_rate = 1e-3
    enc_weight_decay = 1e-5
    mlp_weight_decay = 1e-5
    scheduler_factor = 0.5
    scheduler_patience = 5
    scheduler_min_lr = 1e-6
    max_grad_norm = None
    epochs = 50
    batch_size = 32

    for train_idx, test_idx in outer_cv.split(X, y):
        outer_fold += 1
        print(f"Starting Outer Fold {outer_fold}")
        X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
        y_train_outer, y_test_outer = y[train_idx], y[test_idx]

        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=DEFAULT_RANDOM_STATE)

        def create_model_from_params(params, input_dim):
            return VAE_MLP(
                input_dim=input_dim,
                num_classes=num_classes,
                latent_dim=params['latent_dim'],
                encoder_layers=[params[f'encoder_units_l{i}'] for i in range(params['num_encoder_layers'])],
                mlp_layers=[params[f'mlp_units_l{i}'] for i in range(params['num_mlp_layers'])],
                enc_dropout=params['enc_dropout'],
                mlp_dropout=params['mlp_dropout'],
                enc_norm=params['enc_norm'],
                mlp_norm=params['mlp_norm'],
                early_stopping_metric=params['early_stopping_metric'],
                early_stopping_patience=params['early_stopping_patience'],
                early_stopping_min_delta=params['early_stopping_min_delta'],
                encoder_learning_rate=params['encoder_learning_rate'],
                mlp_learning_rate=params['mlp_learning_rate'],
                enc_weight_decay=params['enc_weight_decay'],
                mlp_weight_decay=params['mlp_weight_decay'],
                scheduler_factor=params['scheduler_factor'],
                scheduler_patience=params['scheduler_patience'],
                scheduler_min_lr=params['scheduler_min_lr'],
                max_grad_norm=params['max_grad_norm'],
                epochs=params['epochs'],
                batch_size=params['batch_size']
            )

        def objective(trial):
            # seeds per trial/fold will be set inside evaluate function
            latent_dim_ = trial.suggest_int('latent_dim', 2, 512, log=True)
            enc_dropout_ = trial.suggest_float('enc_dropout', 0.2, 0.6)
            mlp_dropout_ = trial.suggest_float('mlp_dropout', 0.0, 0.5)
            early_stopping_metric_ = trial.suggest_categorical('early_stopping_metric', ['loss', 'macro_f1'])
            early_stopping_patience_ = trial.suggest_int('early_stopping_patience', 5, 20)
            early_stopping_min_delta_ = trial.suggest_float('early_stopping_min_delta', 0.0, 0.1, step=0.01)

            num_encoder_layers_ = trial.suggest_int('num_encoder_layers', 1, 5)
            encoder_layers_ = [trial.suggest_int(f'encoder_units_l{i}', 16, 512, log=True) for i in range(num_encoder_layers_)]

            num_mlp_layers_ = trial.suggest_int('num_mlp_layers', 1, 5)
            mlp_layers_ = [trial.suggest_int(f'mlp_units_l{i}', 16, 512, log=True) for i in range(num_mlp_layers_)]

            enc_norm_ = trial.suggest_categorical('enc_norm', ['batch', 'layer', 'none'])
            mlp_norm_ = trial.suggest_categorical('mlp_norm', ['batch', 'layer', 'none'])

            encoder_learning_rate_ = trial.suggest_float('encoder_learning_rate', 1e-5, 5e-2, log=True)
            mlp_learning_rate_ = trial.suggest_float('mlp_learning_rate', 1e-5, 5e-2, log=True)
            epochs_ = trial.suggest_int('epochs', 10, 200)
            batch_size_ = trial.suggest_int('batch_size', 2, 256, log=True)

            enc_weight_decay_ = trial.suggest_float('enc_weight_decay', 1e-6, 1e-3, log=True)
            mlp_weight_decay_ = trial.suggest_float('mlp_weight_decay', 1e-6, 1e-3, log=True)
            max_grad_norm_ = trial.suggest_categorical('max_grad_norm', [None, 1.0, 5.0])

            scheduler_factor_ = trial.suggest_float('scheduler_factor', 0.1, 0.7)
            scheduler_patience_ = trial.suggest_int('scheduler_patience', 2, 10)
            scheduler_min_lr_ = trial.suggest_float('scheduler_min_lr', 1e-6, 1e-4, log=True)

            params = {
                'latent_dim': latent_dim_,
                'enc_dropout': enc_dropout_,
                'mlp_dropout': mlp_dropout_,
                'early_stopping_metric': early_stopping_metric_,
                'early_stopping_patience': early_stopping_patience_,
                'early_stopping_min_delta': early_stopping_min_delta_,
                'num_encoder_layers': num_encoder_layers_,
                'num_mlp_layers': num_mlp_layers_,
                'encoder_learning_rate': encoder_learning_rate_,
                'mlp_learning_rate': mlp_learning_rate_,
                'epochs': epochs_,
                'batch_size': batch_size_,
                'enc_norm': enc_norm_,
                'mlp_norm': mlp_norm_,
                'enc_weight_decay': enc_weight_decay_,
                'mlp_weight_decay': mlp_weight_decay_,
                'max_grad_norm': max_grad_norm_,
                'scheduler_factor': scheduler_factor_,
                'scheduler_patience': scheduler_patience_,
                'scheduler_min_lr': scheduler_min_lr_
            }
            # fill layer sizes
            for i in range(num_encoder_layers_):
                params[f'encoder_units_l{i}'] = encoder_layers_[i]
            for i in range(num_mlp_layers_):
                params[f'mlp_units_l{i}'] = mlp_layers_[i]

            #splits = list(inner_cv.split(X_train_outer, y_train_outer))
            def evaluate_inner_fold(fold_id, inner_train_idx, inner_val_idx):
                set_seed(DEFAULT_RANDOM_STATE)
                X_tr_raw, X_va_raw = X_train_outer.iloc[inner_train_idx], X_train_outer.iloc[inner_val_idx]
                y_tr, y_va = y_train_outer[inner_train_idx], y_train_outer[inner_val_idx]
                inner_preprocessor = make_preprocessor()
                X_tr = inner_preprocessor.fit_transform(X_tr_raw, y_tr)
                X_va = inner_preprocessor.transform(X_va_raw)
                model = create_model_from_params(params, X_tr.shape[1])
                model.fit(np.asarray(X_tr), y_tr)
                y_pred = model.predict(np.asarray(X_va))
                if num_classes > 2:
                    return f1_score(y_va, y_pred, average='macro')
                else:
                    return f1_score(y_va, y_pred, average='binary')

            scores = []
            for k, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(X_train_outer, y_train_outer)):
                score = evaluate_inner_fold(k,inner_train_idx, inner_val_idx)
                scores.append(score)

            return float(np.mean(scores))

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=DEFAULT_RANDOM_STATE))
        with SuppressOutput():
            tuning_reports.append(
                run_adaptive_study(
                    study,
                    objective,
                    task="classification",
                    model="MLPVAE",
                    study_scope="outer_fold",
                    objective_name="mlpvae_classification_f1",
                    direction="maximize",
                    tolerance=tolerance_for_objective("mlpvae_classification_f1"),
                    seed=DEFAULT_RANDOM_STATE,
                    study_id=f"classification-mlpvae-outer-fold-{outer_fold}",
                    outer_fold=outer_fold,
                    provenance_path=f"{prefix}_vaemlp_fold{outer_fold}_optuna_study.json",
                )
            )

        best_params = study.best_params
        best_params_by_fold.append(dict(best_params))
        print(f"Best hyperparameters for Outer Fold {outer_fold} found by Optuna:")
        for k, v in best_params.items():
            print(f"{k}: {v}")

        # materialize best params for this fold (also stored for final SHAP later)
        latent_dim = best_params['latent_dim']
        enc_dropout = best_params['enc_dropout']
        mlp_dropout = best_params['mlp_dropout']
        early_stopping_metric_best = best_params['early_stopping_metric']
        early_stopping_patience = best_params['early_stopping_patience']
        early_stopping_min_delta = best_params['early_stopping_min_delta']
        encoder_layers = [best_params[f'encoder_units_l{i}'] for i in range(best_params['num_encoder_layers'])]
        mlp_layers = [best_params[f'mlp_units_l{i}'] for i in range(best_params['num_mlp_layers'])]
        encoder_learning_rate = best_params['encoder_learning_rate']
        mlp_learning_rate = best_params['mlp_learning_rate']
        epochs = best_params['epochs']
        batch_size = best_params['batch_size']
        enc_norm = best_params['enc_norm']
        mlp_norm = best_params['mlp_norm']
        enc_weight_decay = best_params['enc_weight_decay']
        mlp_weight_decay = best_params['mlp_weight_decay']
        max_grad_norm = best_params['max_grad_norm']
        scheduler_factor = best_params['scheduler_factor']
        scheduler_patience = best_params['scheduler_patience']
        scheduler_min_lr = best_params['scheduler_min_lr']

        outer_preprocessor = make_preprocessor()
        X_train_outer_scaled = outer_preprocessor.fit_transform(X_train_outer, y_train_outer)
        X_test_outer_scaled = outer_preprocessor.transform(X_test_outer)
        outer_preprocessor_width = X_train_outer_scaled.shape[1]
        missingness_folds.append(outer_preprocessor.biomarker_diagnostics(outer_fold))

        best_model = VAE_MLP(
            input_dim=outer_preprocessor_width,
            num_classes=num_classes,
            latent_dim=latent_dim,
            encoder_layers=encoder_layers,
            mlp_layers=mlp_layers,
            enc_dropout=enc_dropout,
            mlp_dropout=mlp_dropout,
            enc_norm=enc_norm,
            mlp_norm=mlp_norm,
            early_stopping_metric=early_stopping_metric_best,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            encoder_learning_rate=encoder_learning_rate,
            mlp_learning_rate=mlp_learning_rate,
            enc_weight_decay=enc_weight_decay,
            mlp_weight_decay=mlp_weight_decay,
            scheduler_factor=scheduler_factor,
            scheduler_patience=scheduler_patience,
            scheduler_min_lr=scheduler_min_lr,
            max_grad_norm=max_grad_norm,
            epochs=epochs,
            batch_size=batch_size
        )
        best_model.fit(np.asarray(X_train_outer_scaled), y_train_outer)
        fold_importance, fold_importance_type = estimator_importance_with_source(best_model, outer_preprocessor_width)
        fold_biomarkers = outer_preprocessor.selected_biomarker_names_
        export_outer_fold_biomarkers(
            prefix, "vaemlp", outer_fold, None, fold_biomarkers, selection_method="None",
            importance_values=np.asarray(fold_importance)[:len(fold_biomarkers)],
            importance_type=fold_importance_type,
        )

        y_pred_outer = best_model.predict(np.asarray(X_test_outer_scaled))
        y_pred_proba_outer = best_model.predict_proba(np.asarray(X_test_outer_scaled))

        accuracy = accuracy_score(y_test_outer, y_pred_outer)
        if num_classes > 2:
            f1_val = f1_score(y_test_outer, y_pred_outer, average='macro')
        else:
            f1_val = f1_score(y_test_outer, y_pred_outer, average='binary')
        weighted_f1_val = f1_score(y_test_outer, y_pred_outer, average='weighted', zero_division=0)

        cm = confusion_matrix(y_test_outer, y_pred_outer, labels=np.arange(num_classes))
        cm_sum = np.zeros((num_classes, num_classes), dtype=float)
        cm_sum[:cm.shape[0], :cm.shape[1]] = cm

        sensitivity, specificity = sensitivity_specificity_from_confusion_matrix(cm)

        if num_classes == 2:
            if y_pred_proba_outer.shape[1] >= 2:
                fpr_val, tpr_val, _ = roc_curve(y_test_outer, y_pred_proba_outer[:, 1])
                auc_val = auc(fpr_val, tpr_val)
            else:
                auc_val = 0.0
        else:
            aucs_cls = []
            y_test_bin_outer = np.eye(num_classes)[y_test_outer]
            for i in range(num_classes):
                yi = y_test_bin_outer[:, i]
                if (yi.sum() == 0) or (yi.sum() == len(yi)):
                    continue
                fpr_i, tpr_i, _ = roc_curve(yi, y_pred_proba_outer[:, i])
                aucs_cls.append(auc(fpr_i, tpr_i))
            auc_val = float(np.mean(aucs_cls)) if len(aucs_cls) > 0 else 0.0

        outer_accuracies.append(accuracy)
        outer_f1_scores.append(f1_val)
        outer_weighted_f1_scores.append(weighted_f1_val)
        outer_aucs.append(auc_val)
        outer_sensitivities.append(sensitivity)
        outer_specificities.append(specificity)
        outer_cm_total += cm_sum

        all_X.append(X_test_outer.to_numpy())
        all_y_true.append(y_test_outer)
        all_y_pred.append(y_pred_outer)
        all_y_pred_proba.append(y_pred_proba_outer)
        per_fold_f1.append(f1_val)
        per_fold_auc.append(auc_val)

    plot_confusion_matrix(outer_cm_total, le.classes_, "VAE MLP Confusion Matrix", f"{prefix}_vaemlp_confusion_matrix.png")

    if all_X:
        X_all = np.vstack(all_X)
        y_all_true = np.hstack(all_y_true)
        y_all_pred = np.hstack(all_y_pred)
        y_all_pred_proba = np.vstack(all_y_pred_proba)
    else:
        print("No predictions were made. The predictions DataFrame will not be created.")
        X_all = np.array([])
        y_all_true = np.array([])
        y_all_pred = np.array([])
        y_all_pred_proba = np.array([])

    if X_all.size > 0:
        test_indices = []
        for _, (tr_i, te_i) in enumerate(outer_cv.split(X, y)):
            test_indices.extend(te_i)
        test_indices = np.array(test_indices)
        sorted_order = np.argsort(test_indices)
        sorted_test_indices = test_indices[sorted_order]
        sorted_pred = y_all_pred[sorted_order]
        results_df = pd.DataFrame({
            'SampleID': sample_ids[sorted_test_indices],
            'Original Label': data['Label'].values[sorted_test_indices],
            'Predicted Label': le.inverse_transform(sorted_pred.astype(int))
        })
        sorted_proba = y_all_pred_proba[sorted_order]
        for class_index, class_name in enumerate(le.classes_):
            results_df[f"Probability_{class_name}"] = sorted_proba[:, class_index]
        results_df["Fold"] = np.asarray([fold for fold, (_, te) in enumerate(outer_cv.split(X, y), 1) for _ in te])[sorted_order]
        results_df.to_csv(f"{prefix}_vaemlp_predictions.csv", index=False)
    else:
        print("No predictions were made. The predictions DataFrame will not be created.")

    fpr = {}
    tpr = {}
    roc_auc = {}
    if y_all_pred_proba.size > 0:
        if num_classes == 2:
            if y_all_pred_proba.shape[1] >= 2:
                fpr[0], tpr[0], _ = roc_curve(y_all_true, y_all_pred_proba[:, 1])
                roc_auc[0] = auc(fpr[0], tpr[0])
        else:
            y_all_true_bin = np.eye(num_classes)[y_all_true]
            fpr_macro, tpr_macro, auc_macro, (fpr_c, tpr_c, auc_c) = compute_macro_roc(y_all_true_bin, y_all_pred_proba)
            for i in range(num_classes):
                fpr[i], tpr[i], roc_auc[i] = fpr_c[i], tpr_c[i], auc_c[i]
            fpr["macro"], tpr["macro"], roc_auc["macro"] = fpr_macro, tpr_macro, auc_macro
            try:
                fpr["micro"], tpr["micro"], _ = roc_curve(y_all_true_bin.ravel(), y_all_pred_proba.ravel())
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            except ValueError:
                fpr["micro"], tpr["micro"], roc_auc["micro"] = np.array([0, 1]), np.array([0, 1]), 0.0

    roc_data = {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}
    np.save(f"{prefix}_vaemlp_roc_data.npy", roc_data, allow_pickle=True)

    mean_accuracy = float(np.mean(outer_accuracies)) if outer_accuracies else 0.0
    mean_f1 = float(np.mean(outer_f1_scores)) if outer_f1_scores else 0.0
    mean_auc = float(np.mean(outer_aucs)) if outer_aucs else 0.0
    mean_sensitivity = float(np.mean(outer_sensitivities)) if outer_sensitivities else 0.0
    mean_specificity = float(np.mean(outer_specificities)) if outer_specificities else 0.0

    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'F1 Score', 'Sensitivity', 'Specificity'],
        'Score': [mean_accuracy, mean_f1, mean_sensitivity, mean_specificity]
    })
    print(metrics_df)

    plot_metric_bars(metrics_df["Metric"], metrics_df["Score"], "VAE MLP Performance", f"{prefix}_vaemlp_metrics.png", ylim=(0.0, 1.12))

    curves = []
    if num_classes == 2 and 0 in roc_auc:
        curves.append({"fpr": fpr[0], "tpr": tpr[0], "auc": roc_auc[0], "label": "ROC"})
    else:
        for i in range(num_classes):
            if i in roc_auc:
                curves.append({"fpr": fpr[i], "tpr": tpr[i], "auc": roc_auc[i], "label": str(le.inverse_transform([i])[0])})
        if "micro" in roc_auc:
            curves.append({"fpr": fpr["micro"], "tpr": tpr["micro"], "auc": roc_auc["micro"], "label": "Micro-average", "linestyle": "--"})
        if "macro" in roc_auc:
            curves.append({"fpr": fpr["macro"], "tpr": tpr["macro"], "auc": roc_auc["macro"], "label": "Macro-average", "linestyle": "-."})
    plot_roc_curves(curves, "VAE MLP ROC", f"{prefix}_vaemlp_roc_curve.png")

    folds = np.arange(1, outer_fold + 1)
    plot_classification_nested_cv(folds, per_fold_f1, per_fold_auc, "VAE MLP Nested CV", f"{prefix}_vaemlp_nested_cv_f1_auc.png", f1_label="F1")
    pd.DataFrame(
        {
            "Model": "vaemlp",
            "Fold": folds,
            "AUC": outer_aucs,
            "weighted_F1": outer_weighted_f1_scores,
            "Accuracy": outer_accuracies,
        }
    ).to_csv(f"{prefix}_vaemlp_outer_fold_metrics.csv", index=False)
    pooled_auc = float(roc_auc.get("macro", roc_auc.get(0, np.nan)))
    pooled_micro_auc = float(roc_auc.get("micro", pooled_auc))
    pd.DataFrame([{
        "Model": "vaemlp",
        "Accuracy": float(accuracy_score(y_all_true, y_all_pred)),
        "weighted_F1": float(f1_score(y_all_true, y_all_pred, average="weighted", zero_division=0)),
        "OOF_AUC": pooled_auc,
        "Macro_AUC": pooled_auc,
        "Micro_AUC": pooled_micro_auc,
        "MeanFold_AUC": float(np.mean(outer_aucs)),
        "SDFold_AUC": float(np.std(outer_aucs, ddof=1)) if len(outer_aucs) > 1 else 0.0,
        "MetricScope": "Pooled outer-fold out-of-fold predictions",
    }]).to_csv(f"{prefix}_vaemlp_final_validation_metrics.csv", index=False)

    try:
        write_missingness_qc(prefix + "_vaemlp", X, roles, missingness_folds)
        # Final-model tuning is a new full-development CV procedure; outer OOF results above remain frozen.
        X_train_outer, y_train_outer = X, y
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
        final_study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=DEFAULT_RANDOM_STATE))
        with SuppressOutput():
            tuning_reports.append(
                run_adaptive_study(
                    final_study,
                    objective,
                    task="classification",
                    model="MLPVAE",
                    study_scope="final_full_development",
                    objective_name="mlpvae_classification_f1",
                    direction="maximize",
                    tolerance=tolerance_for_objective("mlpvae_classification_f1"),
                    seed=DEFAULT_RANDOM_STATE,
                    study_id="classification-mlpvae-final",
                    provenance_path=f"{prefix}_vaemlp_final_optuna_study.json",
                )
            )
        final_params = dict(final_study.best_params)
        final_preprocessor = make_preprocessor()
        X_scaled_final = final_preprocessor.fit_transform(X, y)

        latent_dim = final_params['latent_dim']
        enc_dropout = final_params['enc_dropout']
        mlp_dropout = final_params['mlp_dropout']
        early_stopping_metric_best = final_params['early_stopping_metric']
        early_stopping_patience = final_params['early_stopping_patience']
        early_stopping_min_delta = final_params['early_stopping_min_delta']
        encoder_layers = [final_params[f'encoder_units_l{i}'] for i in range(final_params['num_encoder_layers'])]
        mlp_layers = [final_params[f'mlp_units_l{i}'] for i in range(final_params['num_mlp_layers'])]
        encoder_learning_rate = final_params['encoder_learning_rate']
        mlp_learning_rate = final_params['mlp_learning_rate']
        epochs = final_params['epochs']
        batch_size = final_params['batch_size']
        enc_norm = final_params['enc_norm']
        mlp_norm = final_params['mlp_norm']
        enc_weight_decay = final_params['enc_weight_decay']
        mlp_weight_decay = final_params['mlp_weight_decay']
        max_grad_norm = final_params['max_grad_norm']
        scheduler_factor = final_params['scheduler_factor']
        scheduler_patience = final_params['scheduler_patience']
        scheduler_min_lr = final_params['scheduler_min_lr']

        def make_final_model():
            set_seed(DEFAULT_RANDOM_STATE)
            return VAE_MLP(
                input_dim=X_scaled_final.shape[1],
                num_classes=num_classes,
                latent_dim=latent_dim,
                encoder_layers=encoder_layers,
                mlp_layers=mlp_layers,
                enc_dropout=enc_dropout,
                mlp_dropout=mlp_dropout,
                enc_norm=enc_norm,
                mlp_norm=mlp_norm,
                early_stopping_metric=early_stopping_metric_best,
                early_stopping_patience=early_stopping_patience,
                early_stopping_min_delta=early_stopping_min_delta,
                encoder_learning_rate=encoder_learning_rate,
                mlp_learning_rate=mlp_learning_rate,
                enc_weight_decay=enc_weight_decay,
                mlp_weight_decay=mlp_weight_decay,
                scheduler_factor=scheduler_factor,
                scheduler_patience=scheduler_patience,
                scheduler_min_lr=scheduler_min_lr,
                max_grad_norm=max_grad_norm,
                epochs=epochs,
                batch_size=batch_size
            )

        final_model = make_final_model()
        final_model.fit(np.asarray(X_scaled_final), y)
        write_training_history(getattr(final_model, "training_history_", []), prefix, "vaemlp")
        generate_training_size_curve(np.asarray(X_scaled_final), y, make_final_model, prefix, "vaemlp", task="classification")
        joblib.dump(final_model, f"{prefix}_vaemlp_model.pkl")
        joblib.dump(final_preprocessor, f"{prefix}_vaemlp_scaler.pkl")
        joblib.dump((X, y, le, list(final_preprocessor.get_feature_names_out())), f"{prefix}_vaemlp_data.pkl")
        joblib.dump(
            {
                "model": final_model,
                "preprocessor": final_preprocessor,
                "feature_order": list(X.columns),
                "model_feature_order": list(final_preprocessor.get_feature_names_out()),
                "X_raw": X,
                "y": y,
                "label_encoder": le,
                "feature_roles": roles,
                "selected_biomarker_features": list(final_preprocessor.selected_biomarker_names_),
                "final_hyperparameters": final_params,
                "architecture": {
                    key: value
                    for key, value in final_params.items()
                    if "layer" in key or "units" in key or key == "latent_dim"
                },
                "random_seed": DEFAULT_RANDOM_STATE,
                "scope": "final_full_development_model",
            },
            f"{prefix}_vaemlp_final_artifact.pkl",
        )
        with open(f"{prefix}_vaemlp_best_params.json", "w", encoding="utf-8") as handle:
            import json
            json.dump({"final": final_params, "outer_folds": best_params_by_fold, "provenance": "explicit full-development CV after OOF validation", "adaptive_tuning": tuning_reports}, handle, indent=2)
        write_reproducibility_manifest(f"{prefix}_vaemlp_manifest.json", input_path=inp, task="classification", model="vaemlp", seed=DEFAULT_RANDOM_STATE, roles=roles, imputation=imputation, final_hyperparameters=final_params, artifacts=[f"{prefix}_vaemlp_predictions.csv", f"{prefix}_vaemlp_final_artifact.pkl"], imputation_fallbacks=imputation_fallback_reasons(missingness_folds), tuning_studies=tuning_reports)
    except Exception as exc:
        raise RuntimeError("Final VAE-MLP tuning or artifact persistence failed.") from exc

if __name__ == "__main__":
    args = parse_arguments()
    set_seed(args.seed)
    prefix = build_output_prefix(args.csv, args.prefix)
    with task_run_context(args.outdir, "classification"):
        vae(args.csv, prefix, args.imputation, args.covariates, args.categorical_covariates)
