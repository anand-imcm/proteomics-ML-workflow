import argparse
import random
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
import shap
import matplotlib.pyplot as plt
plt.style.use('default')
import warnings
import sys
import contextlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import optuna
import joblib
from joblib import Parallel, delayed
import matplotlib.ticker as mticker
from sklearn.utils.class_weight import compute_class_weight

# ------------------------------ Reproducibility ------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description='Script to run classifiers')
    parser.add_argument('-i', '--csv', type=str, help='Input CSV file', required=True)
    parser.add_argument('-p', '--prefix', type=str, help='Output prefix')
    return parser.parse_args()

def set_seed(seed=42):
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

# ------------------------------ Plot fonts -----------------------------------
TITLE_FONTSIZE = 22
LABEL_FONTSIZE = 18
TICK_FONTSIZE = 14
LEGEND_FONTSIZE = 14

# ------------------------------ IO suppression --------------------------------
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

# ------------------------------ Early Stopping --------------------------------
class EarlyStopping:
    """
    mode: 'min' for loss, 'max' for metrics like macro_f1
    """
    def __init__(self, patience=10, min_delta=0.0, mode='min', verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.best = None
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, current):
        if self.best is None:
            self.best = current
            return False
        improved = False
        if self.mode == 'min':
            improved = current < self.best - self.min_delta
        else:  # 'max'
            improved = current > self.best + self.min_delta
        if improved:
            self.best = current
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False

# ------------------------------ Norm selector ---------------------------------
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

# ------------------------------ Models ----------------------------------------
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

        # Train/val split for early stopping
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train_t = torch.from_numpy(X_train).float().to(self.device)
        y_train_t = torch.from_numpy(y_train).long().to(self.device)
        X_val_t = torch.from_numpy(X_val).float().to(self.device)
        y_val_t = torch.from_numpy(y_val).long().to(self.device)

        dl_train = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train_t, y_train_t),
            batch_size=self.batch_size, shuffle=True, drop_last=True
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
        for _ in range(self.epochs):
            # Train
            for xb, yb in dl_train:
                self.optimizer.zero_grad()
                logits = self.classifier(self.encoder(xb))
                loss = criterion(logits, yb)
                loss.backward()
                if self.max_grad_norm is not None and self.max_grad_norm > 0:
                    clip_grad_norm_(list(self.encoder.parameters()) + list(self.classifier.parameters()),
                                    max_norm=self.max_grad_norm)
                self.optimizer.step()

            # Validation
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
            if early_stopping(monitor_value):
                break
            self.encoder.train()
            self.classifier.train()
        return self

    def predict(self, X):
        self.encoder.eval()
        self.classifier.eval()
        ds = torch.utils.data.TensorDataset(torch.from_numpy(X).float().to(self.device))
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=False, drop_last=False)
        preds = []
        with torch.no_grad():
            for (xb,) in dl:
                logits = self.classifier(self.encoder(xb))
                preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        return np.concatenate(preds) if len(preds) > 0 else np.array([])

    def predict_proba(self, X):
        self.encoder.eval()
        self.classifier.eval()
        ds = torch.utils.data.TensorDataset(torch.from_numpy(X).float().to(self.device))
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=False, drop_last=False)
        probas = []
        with torch.no_grad():
            for (xb,) in dl:
                logits = self.classifier(self.encoder(xb))
                probas.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.vstack(probas) if len(probas) > 0 else np.array([])

# ------------------------------ ROC helpers -----------------------------------
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

# ------------------------------ Main ------------------------------------------
def vae(inp, prefix):
    data = pd.read_csv(inp)
    sample_ids = data['SampleID'].values
    X = data.drop(columns=['SampleID', 'Label']).values
    y = data['Label'].values
    feature_names = data.drop(columns=['SampleID', 'Label']).columns.tolist()

    le = LabelEncoder()
    y = le.fit_transform(y)
    num_classes = len(np.unique(y))

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    outer_fold = 0

    outer_accuracies = []
    outer_f1_scores = []
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

    # hyperparams placeholders for final SHAP model (from last fold best)
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
        X_train_outer, X_test_outer = X[train_idx], X[test_idx]
        y_train_outer, y_test_outer = y[train_idx], y[test_idx]

        scaler_outer = StandardScaler()
        X_train_outer_scaled = scaler_outer.fit_transform(X_train_outer)
        X_test_outer_scaled = scaler_outer.transform(X_test_outer)

        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        def create_model_from_params(params):
            return VAE_MLP(
                input_dim=X.shape[1],
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

            #splits = list(inner_cv.split(X_train_outer_scaled, y_train_outer))
            def evaluate_inner_fold(fold_id, inner_train_idx, inner_val_idx):
                set_seed(12345 + fold_id)
                X_tr, X_va = X_train_outer_scaled[inner_train_idx], X_train_outer_scaled[inner_val_idx]
                y_tr, y_va = y_train_outer[inner_train_idx], y_train_outer[inner_val_idx]
                model = create_model_from_params(params)
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_va)
                if num_classes > 2:
                    return f1_score(y_va, y_pred, average='macro')
                else:
                    return f1_score(y_va, y_pred, average='binary')

            # scores = Parallel(n_jobs=-1)(
            #     delayed(evaluate_inner_fold)(k, tr, va) for k, (tr, va) in enumerate(splits)
            # )
            scores = []
            for k, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(X_train_outer_scaled, y_train_outer)):
                #set_seed(12345 + k)
                score = evaluate_inner_fold(k,inner_train_idx, inner_val_idx)
                scores.append(score)

            return float(np.mean(scores))

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        with SuppressOutput():
            study.optimize(objective, n_trials=20, n_jobs=1)

        best_params = study.best_params
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

        best_model = VAE_MLP(
            input_dim=X.shape[1],
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
        best_model.fit(X_train_outer_scaled, y_train_outer)

        y_pred_outer = best_model.predict(X_test_outer_scaled)
        y_pred_proba_outer = best_model.predict_proba(X_test_outer_scaled)

        accuracy = accuracy_score(y_test_outer, y_pred_outer)
        if num_classes > 2:
            f1_val = f1_score(y_test_outer, y_pred_outer, average='macro')
        else:
            f1_val = f1_score(y_test_outer, y_pred_outer, average='binary')

        cm = confusion_matrix(y_test_outer, y_pred_outer, labels=np.arange(num_classes))
        cm_sum = np.zeros((num_classes, num_classes), dtype=float)
        cm_sum[:cm.shape[0], :cm.shape[1]] = cm

        if num_classes == 2:
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            else:
                sensitivity = 0.0
                specificity = 0.0
        else:
            sens_list, spec_list = [], []
            for i in range(num_classes):
                tp = cm[i, i]
                fn = np.sum(cm[i, :]) - tp
                fp = np.sum(cm[:, i]) - tp
                tn = np.sum(cm) - tp - fn - fp
                sens_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
                spec_list.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
            sensitivity = float(np.mean(sens_list))
            specificity = float(np.mean(spec_list))

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
        outer_aucs.append(auc_val)
        outer_sensitivities.append(sensitivity)
        outer_specificities.append(specificity)
        outer_cm_total += cm_sum

        all_X.append(X_test_outer)
        all_y_true.append(y_test_outer)
        all_y_pred.append(y_pred_outer)
        all_y_pred_proba.append(y_pred_proba_outer)
        per_fold_f1.append(f1_val)
        per_fold_auc.append(auc_val)

    # ----------------- Confusion Matrix (average) -----------------
    disp = ConfusionMatrixDisplay(confusion_matrix=outer_cm_total, display_labels=le.classes_)
    disp.plot(cmap=plt.cm.Blues, values_format='.0f')
    disp.plot(cmap=plt.cm.Blues, values_format='.0f')

    ax = plt.gca()
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.ticklabel_format(style='plain', axis='both')
    plt.title('Confusion Matrix for VAE_MLP', fontsize=TITLE_FONTSIZE, fontweight='bold')
    plt.xlabel('Predicted label', fontsize=LABEL_FONTSIZE)
    plt.ylabel('True label', fontsize=LABEL_FONTSIZE)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.tight_layout()
    plt.savefig(f"{prefix}_vaemlp_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ----------------- Aggregate predictions -----------------
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
        results_df.to_csv(f"{prefix}_vaemlp_predictions.csv", index=False)
    else:
        print("No predictions were made. The predictions DataFrame will not be created.")

    # ----------------- ROC Curves (aggregate) -----------------
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

    # ----------------- Metrics summary ----------------------------------------
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

    plt.figure(figsize=(12, 8))
    bars = plt.bar(metrics_df['Metric'], metrics_df['Score'])
    max_yval = max(metrics_df['Score']) if len(metrics_df['Score']) > 0 else 1.0
    plt.ylim(0, max_yval + 0.05)
    for bar, score in zip(bars, metrics_df['Score']):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f"{score:.3f}", ha='center', va='bottom', fontsize=TICK_FONTSIZE)
    plt.title('Performance Metrics for VAE_MLP', fontsize=TITLE_FONTSIZE, fontweight='bold', pad=15)
    plt.ylabel('Score', fontsize=LABEL_FONTSIZE, labelpad=10)
    plt.xlabel('Metric', fontsize=LABEL_FONTSIZE, labelpad=10)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.tight_layout()
    plt.savefig(f"{prefix}_vaemlp_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ----------------- ROC plot ------------------------------------------------
    plt.figure(figsize=(10, 8))
    if num_classes == 2 and 0 in roc_auc:
        plt.plot(fpr[0], tpr[0], label=f'ROC curve (AUC = {roc_auc[0]:.2f})')
    else:
        for i in range(num_classes):
            if i in roc_auc:
                plt.plot(fpr[i], tpr[i], label=f'Class {le.inverse_transform([i])[0]} (AUC = {roc_auc[i]:.2f})')
        if "micro" in roc_auc:
            plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})', linestyle='--')
        if "macro" in roc_auc:
            plt.plot(fpr["macro"], tpr["macro"], label=f'Macro-average (AUC = {roc_auc["macro"]:.2f})', linestyle='-.')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=LABEL_FONTSIZE, labelpad=10)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=LABEL_FONTSIZE, labelpad=10)
    plt.title('ROC Curves for VAE_MLP', fontsize=TITLE_FONTSIZE, fontweight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_FONTSIZE)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    plt.ticklabel_format(style='plain', axis='both')
    plt.tight_layout()
    plt.savefig(f"{prefix}_vaemlp_roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ----------------- F1 and AUC per fold ------------------------------------
    folds = np.arange(1, outer_fold + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(folds, per_fold_f1, marker='o', linestyle='-', label='F1 Score')
    plt.plot(folds, per_fold_auc, marker='s', linestyle='-', label='AUC Score')
    plt.title('F1 and AUC Scores per Outer Fold', fontsize=TITLE_FONTSIZE, fontweight='bold', pad=15)
    plt.xlabel('Outer Fold Number', fontsize=LABEL_FONTSIZE, labelpad=10)
    plt.ylabel('Score', fontsize=LABEL_FONTSIZE, labelpad=10)
    plt.xticks(folds, fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_FONTSIZE)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{prefix}_vaemlp_nested_cv_f1_auc.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ----------------- SHAP with consistent scaling ---------------------------
    try:
        scaler_final = StandardScaler()
        X_scaled_final = scaler_final.fit_transform(X)

        final_model = VAE_MLP(
            input_dim=X.shape[1],
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
        final_model.fit(X_scaled_final, y)

        background_size = 100
        if X_scaled_final.shape[0] < background_size:
            background_size = X_scaled_final.shape[0]
        background_indices = np.random.choice(X_scaled_final.shape[0], background_size, replace=False)
        background = X_scaled_final[background_indices]
        X_explain = X_scaled_final

        def model_predict_proba_scaled(X_input):
            return final_model.predict_proba(X_input)

        explainer = shap.PermutationExplainer(model_predict_proba_scaled, background, n_repeats=10, random_state=42, n_jobs=-1)
        shap_values = explainer.shap_values(X_explain)

        if isinstance(shap_values, list):
            shap_values = np.stack(shap_values, axis=-1)

        if num_classes > 2:
            class_feature_tuples = [(class_idx, feature_idx) for class_idx in range(num_classes) for feature_idx in range(len(feature_names))]
            def compute_mean_shap(shap_values_arr, class_idx, feature_idx):
                shap_val = shap_values_arr[:, feature_idx, class_idx]
                mean_shap = float(np.mean(shap_val))
                return (class_idx, feature_names[feature_idx], mean_shap)
        
            mean_shap_results = Parallel(n_jobs=-1)(
                delayed(compute_mean_shap)(shap_values, class_idx, feature_idx)
                for class_idx, feature_idx in class_feature_tuples
            )
        
            shap_dict = {feature: {} for feature in feature_names}
            for class_idx, feature_name, mean_shap in mean_shap_results:
                class_name = le.inverse_transform([class_idx])[0]
                shap_dict[feature_name][class_name] = mean_shap
        
            shap_df = pd.DataFrame(shap_dict).T.reset_index().rename(columns={'index': 'Feature'})
            shap_df = shap_df[['Feature'] + list(le.classes_)]
            global_mean = np.mean(np.abs(shap_values), axis=(0, 2))
            shap_df["Global Mean"] = global_mean
            shap_df.to_csv(f"{prefix}_vaemlp_shap_values.csv", index=False)
            print(f"SHAP values per class have been saved to {prefix}_vaemlp_shap_values.csv")
        else:
            shap_val = shap_values[:, :, 1]
            mean_shap = np.mean(shap_val, axis=0)
            shap_df = pd.DataFrame({'Feature': feature_names, 'Mean SHAP Value': mean_shap})
            shap_df['Global Mean'] = np.mean(np.abs(shap_val), axis=0)
            shap_df.to_csv(f"{prefix}_vaemlp_shap_values.csv", index=False)
            print(f"SHAP values have been saved to {prefix}_vaemlp_shap_values.csv")
    except Exception as e:
        print(f"SHAP computation was skipped due to an error: {e}")

if __name__ == "__main__":
    args = parse_arguments()
    prefix = Path(args.csv).stem
    if args.prefix:
        prefix = args.prefix
    vae(args.csv, prefix)
