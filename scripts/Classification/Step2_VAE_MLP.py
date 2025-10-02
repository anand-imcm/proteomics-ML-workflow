import argparse
import random
from pathlib import Path
import warnings
import sys
import contextlib

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

import optuna
import joblib
from joblib import Parallel, delayed  # not used for training to keep reproducibility

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import shap


# ------------------------------ Reproducibility ------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="VAE -> reconstruct X, then MLP classification")
    parser.add_argument("-i", "--csv", type=str, help="Input CSV file", required=True)
    parser.add_argument("-p", "--prefix", type=str, help="Output prefix")
    return parser.parse_args()


def set_seed(seed: int = 42):
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
warnings.filterwarnings("ignore")

# ------------------------------ Plot fonts -----------------------------------
TITLE_FONTSIZE = 24
LABEL_FONTSIZE = 20
TICK_FONTSIZE = 16
LEGEND_FONTSIZE = 16

# ------------------------------ IO suppression --------------------------------
class SuppressOutput(contextlib.AbstractContextManager):
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = open("/dev/null", "w")
        sys.stderr = open("/dev/null", "w")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            sys.stdout.close()
            sys.stderr.close()
        finally:
            sys.stdout = self._stdout
            sys.stderr = self._stderr


# ------------------------------ Early Stopping --------------------------------
class EarlyStopping:
    """Stop when monitored value stops improving."""

    def __init__(self, patience=10, min_delta=0.0, mode: str = "min", verbose=False):
        assert mode in ("min", "max")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.best = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_value: float) -> bool:
        if self.best is None:
            self.best = current_value
            return False
        improved = (current_value < (self.best - self.min_delta)) if self.mode == "min" else (
            current_value > (self.best + self.min_delta)
        )
        if improved:
            self.best = current_value
            self.counter = 0
            return False
        self.counter += 1
        if self.verbose:
            print(f"EarlyStopping counter: {self.counter} / {self.patience}")
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        return False


# ------------------------------ Norm selector ---------------------------------
class NormLayer(nn.Module):
    def __init__(self, norm_type: str, units: int):
        super().__init__()
        nt = (norm_type or "none").lower()
        if nt == "batch":
            self.norm = nn.BatchNorm1d(units)
        elif nt == "layer":
            self.norm = nn.LayerNorm(units)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        return self.norm(x)


# ------------------------------ Models ----------------------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, encoder_layers, dropout_rate, enc_norm: str):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for units in encoder_layers:
            layers += [
                nn.Linear(prev_dim, units),
                nn.LeakyReLU(),
                NormLayer(enc_norm, units),
                nn.Dropout(dropout_rate),
            ]
            prev_dim = units
        self.hidden = nn.Sequential(*layers)
        self.z_mean = nn.Linear(prev_dim, latent_dim)
        self.z_log_var = nn.Linear(prev_dim, latent_dim)

    def forward(self, x):
        h = self.hidden(x)
        z_mean = self.z_mean(h)
        z_log_var = torch.clamp(self.z_log_var(h), min=-10, max=10)
        return z_mean, z_log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, decoder_layers, dropout_rate, enc_norm: str):
        super().__init__()
        layers = []
        prev_dim = latent_dim
        for units in decoder_layers:
            layers += [
                nn.Linear(prev_dim, units),
                nn.LeakyReLU(),
                NormLayer(enc_norm, units),
                nn.Dropout(dropout_rate),
            ]
            prev_dim = units
        layers += [nn.Linear(prev_dim, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, encoder_layers, decoder_layers, dropout_rate, enc_norm: str):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim, encoder_layers, dropout_rate, enc_norm)
        self.decoder = Decoder(latent_dim, input_dim, decoder_layers, dropout_rate, enc_norm)

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        z = z_mean + eps * std
        x_recon = self.decoder(z)
        return x_recon, z_mean, z_log_var


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, mlp_layers, dropout_rate, mlp_norm: str):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for units in mlp_layers:
            layers += [
                nn.Linear(prev_dim, units),
                nn.LeakyReLU(),
                NormLayer(mlp_norm, units),
                nn.Dropout(dropout_rate),
            ]
            prev_dim = units
        layers += [nn.Linear(prev_dim, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ------------------------------ Wrapper: VAE -> MLP ---------------------------
class VAE_MLP(BaseEstimator, ClassifierMixin):
    """
    Pipeline:
    1) Train VAE to reconstruct inputs.
    2) Use reconstructed X as features to train an MLP classifier.
    """

    def __init__(
        self,
        input_dim=30,
        num_classes=2,
        latent_dim=2,
        encoder_layers=(64, 32),
        decoder_layers=(32, 64),
        mlp_layers=(32, 16),
        enc_norm: str = "layer",
        mlp_norm: str = "layer",
        vae_dropout=0.5,
        mlp_dropout=0.5,
        vae_early_stopping_patience=10,
        vae_early_stopping_min_delta=0.0,
        mlp_early_stopping_patience=10,
        mlp_early_stopping_min_delta=0.0,
        mlp_early_stopping_metric: str = "macro_f1",  # 'loss' or 'macro_f1'
        vae_learning_rate=1e-3,
        mlp_learning_rate=1e-3,
        vae_weight_decay=1e-5,
        mlp_weight_decay=1e-5,
        scheduler_factor=0.5,
        scheduler_patience=5,
        scheduler_min_lr=1e-6,
        max_grad_norm=None,  # None, 1.0, 5.0
        epochs=50,
        batch_size=32,
    ):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.encoder_layers = list(encoder_layers)
        self.decoder_layers = list(decoder_layers)
        self.mlp_layers = list(mlp_layers)
        self.enc_norm = enc_norm
        self.mlp_norm = mlp_norm
        self.vae_dropout = vae_dropout
        self.mlp_dropout = mlp_dropout
        self.vae_early_stopping_patience = vae_early_stopping_patience
        self.vae_early_stopping_min_delta = vae_early_stopping_min_delta
        self.mlp_early_stopping_patience = mlp_early_stopping_patience
        self.mlp_early_stopping_min_delta = mlp_early_stopping_min_delta
        self.mlp_early_stopping_metric = mlp_early_stopping_metric
        self.vae_learning_rate = vae_learning_rate
        self.mlp_learning_rate = mlp_learning_rate
        self.vae_weight_decay = vae_weight_decay
        self.mlp_weight_decay = mlp_weight_decay
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.scheduler_min_lr = scheduler_min_lr
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes_ = None
        self.vae = None
        self.vae_optimizer = None
        self.mlp = None
        self.mlp_optimizer = None
        self.vae_scheduler = None
        self.mlp_scheduler = None

    def _create_vae(self):
        self.vae = VAE(
            self.input_dim,
            self.latent_dim,
            self.encoder_layers,
            self.decoder_layers,
            self.vae_dropout,
            self.enc_norm,
        ).to(self.device)
        self.vae_optimizer = optim.Adam(
            self.vae.parameters(), lr=self.vae_learning_rate, weight_decay=self.vae_weight_decay
        )
        self.vae_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.vae_optimizer,
            mode="min",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
            min_lr=self.scheduler_min_lr,
            verbose=False,
        )

    def _create_mlp(self):
        self.mlp = MLPClassifier(
            self.input_dim,
            self.num_classes,
            self.mlp_layers,
            self.mlp_dropout,
            self.mlp_norm,
        ).to(self.device)
        self.mlp_optimizer = optim.Adam(
            self.mlp.parameters(), lr=self.mlp_learning_rate, weight_decay=self.mlp_weight_decay
        )
        mode = "min" if self.mlp_early_stopping_metric == "loss" else "max"
        self.mlp_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.mlp_optimizer,
            mode=mode,
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
            min_lr=self.scheduler_min_lr,
            verbose=False,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        set_seed(42)
        self.classes_ = np.unique(y)

        # ---------------- VAE training with early stopping ----------------
        self._create_vae()

        X_tr_vae, X_val_vae = train_test_split(X, test_size=0.2, random_state=42, shuffle=True)
        X_tr_vae_t = torch.from_numpy(X_tr_vae).float().to(self.device)
        X_val_vae_t = torch.from_numpy(X_val_vae).float().to(self.device)

        dl_tr_vae = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_tr_vae_t),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
        dl_val_vae = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_val_vae_t),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

        es_vae = EarlyStopping(
            patience=self.vae_early_stopping_patience,
            min_delta=self.vae_early_stopping_min_delta,
            mode="min",
            verbose=False,
        )

        self.vae.train()
        for _ in range(self.epochs):
            # train epoch (mean per-sample loss)
            train_sum = 0.0
            train_count = 0
            for (xb,) in dl_tr_vae:
                self.vae_optimizer.zero_grad()
                x_recon, z_mean, z_log_var = self.vae(xb)
                recon = nn.functional.mse_loss(x_recon, xb, reduction="sum")
                kl = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
                loss = recon + kl
                loss.backward()
                if self.max_grad_norm is not None and self.max_grad_norm > 0:
                    clip_grad_norm_(self.vae.parameters(), max_norm=self.max_grad_norm)
                self.vae_optimizer.step()
                train_sum += loss.item()
                train_count += xb.size(0)
            _ = train_sum / max(train_count, 1)

            # validation
            self.vae.eval()
            val_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for (xb,) in dl_val_vae:
                    x_recon, z_mean, z_log_var = self.vae(xb)
                    recon = nn.functional.mse_loss(x_recon, xb, reduction="sum")
                    kl = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
                    loss = recon + kl
                    val_sum += loss.item()
                    val_count += xb.size(0)
            val_mean = val_sum / max(val_count, 1)
            self.vae_scheduler.step(val_mean)
            if es_vae(val_mean):
                break
            self.vae.train()

        # ---------------- Build reconstructed features for the given X ----------------
        self.vae.eval()
        with torch.no_grad():
            ds = torch.utils.data.TensorDataset(torch.from_numpy(X).float().to(self.device))
            dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=False, drop_last=False)
            X_recon_list = []
            for (xb,) in dl:
                x_recon, _, _ = self.vae(xb)
                X_recon_list.append(x_recon.cpu())
            X_recon = torch.cat(X_recon_list, dim=0).numpy()

        # ---------------- MLP training with early stopping ----------------
        self._create_mlp()

        X_tr_mlp, X_val_mlp, y_tr_mlp, y_val_mlp = train_test_split(
            X_recon, y, test_size=0.2, random_state=42, stratify=y
        )
        X_tr_mlp_t = torch.from_numpy(X_tr_mlp).float().to(self.device)
        y_tr_mlp_t = torch.from_numpy(y_tr_mlp).long().to(self.device)
        X_val_mlp_t = torch.from_numpy(X_val_mlp).float().to(self.device)
        y_val_mlp_t = torch.from_numpy(y_val_mlp).long().to(self.device)

        dl_tr_mlp = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_tr_mlp_t, y_tr_mlp_t),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
        dl_val_mlp = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_val_mlp_t, y_val_mlp_t),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

        # class weights aligned to num_classes
        weights_full = np.ones(self.num_classes, dtype=np.float32)
        present = np.unique(y_tr_mlp)
        weights_present = compute_class_weight(class_weight="balanced", classes=present, y=y_tr_mlp)
        for idx, cls in enumerate(present):
            weights_full[int(cls)] = float(weights_present[idx])
        assert len(weights_full) == self.num_classes
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights_full, dtype=torch.float32, device=self.device))

        es_mode = "min" if self.mlp_early_stopping_metric == "loss" else "max"
        es_mlp = EarlyStopping(
            patience=self.mlp_early_stopping_patience,
            min_delta=self.mlp_early_stopping_min_delta,
            mode=es_mode,
            verbose=False,
        )

        self.mlp.train()
        for _ in range(self.epochs):
            # train epoch (mean loss per sample)
            tr_sum = 0.0
            tr_count = 0
            for xb, yb in dl_tr_mlp:
                self.mlp_optimizer.zero_grad()
                logits = self.mlp(xb)
                loss = criterion(logits, yb)
                loss.backward()
                if self.max_grad_norm is not None and self.max_grad_norm > 0:
                    clip_grad_norm_(self.mlp.parameters(), max_norm=self.max_grad_norm)
                self.mlp_optimizer.step()
                tr_sum += loss.item() * yb.size(0)
                tr_count += yb.size(0)
            _ = tr_sum / max(tr_count, 1)

            # validation (loss and optional macro F1)
            self.mlp.eval()
            va_sum = 0.0
            va_count = 0
            all_pred = []
            all_true = []
            with torch.no_grad():
                for xb, yb in dl_val_mlp:
                    logits = self.mlp(xb)
                    loss = criterion(logits, yb)
                    va_sum += loss.item() * yb.size(0)
                    va_count += yb.size(0)
                    if self.mlp_early_stopping_metric == "macro_f1":
                        preds = torch.argmax(logits, dim=1)
                        all_pred.append(preds.cpu().numpy())
                        all_true.append(yb.cpu().numpy())
            va_mean = va_sum / max(va_count, 1)

            if self.mlp_early_stopping_metric == "macro_f1":
                if all_true:
                    y_true_va = np.concatenate(all_true)
                    y_pred_va = np.concatenate(all_pred)
                    monitor = f1_score(
                        y_true_va, y_pred_va, average="macro" if self.num_classes > 2 else "binary"
                    )
                else:
                    monitor = 0.0
                self.mlp_scheduler.step(monitor)
            else:
                monitor = va_mean
                self.mlp_scheduler.step(monitor)

            if es_mlp(monitor):
                break
            self.mlp.train()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.vae.eval()
        self.mlp.eval()
        with torch.no_grad():
            ds = torch.utils.data.TensorDataset(torch.from_numpy(X).float().to(self.device))
            dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=False, drop_last=False)
            X_recon_list = []
            for (xb,) in dl:
                x_recon, _, _ = self.vae(xb)
                X_recon_list.append(x_recon.cpu())
            X_recon = torch.cat(X_recon_list, dim=0).numpy()

            ds2 = torch.utils.data.TensorDataset(torch.from_numpy(X_recon).float().to(self.device))
            dl2 = torch.utils.data.DataLoader(ds2, batch_size=self.batch_size, shuffle=False, drop_last=False)
            preds = []
            for (xb,) in dl2:
                logits = self.mlp(xb)
                preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            return np.concatenate(preds) if preds else np.array([])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.vae.eval()
        self.mlp.eval()
        with torch.no_grad():
            ds = torch.utils.data.TensorDataset(torch.from_numpy(X).float().to(self.device))
            dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=False, drop_last=False)
            X_recon_list = []
            for (xb,) in dl:
                x_recon, _, _ = self.vae(xb)
                X_recon_list.append(x_recon.cpu())
            X_recon = torch.cat(X_recon_list, dim=0).numpy()

            ds2 = torch.utils.data.TensorDataset(torch.from_numpy(X_recon).float().to(self.device))
            dl2 = torch.utils.data.DataLoader(ds2, batch_size=self.batch_size, shuffle=False, drop_last=False)
            probas = []
            for (xb,) in dl2:
                logits = self.mlp(xb)
                probas.append(torch.softmax(logits, dim=1).cpu().numpy())
            return np.vstack(probas) if probas else np.array([])


# ------------------------------ ROC helpers -----------------------------------
def compute_macro_roc(y_true_bin: np.ndarray, y_score: np.ndarray):
    """
    y_true_bin: (n_samples, n_classes) one-hot
    y_score:    (n_samples, n_classes) predicted probabilities
    """
    n_classes = y_true_bin.shape[1]
    fpr_d, tpr_d, auc_d = {}, {}, {}
    for i in range(n_classes):
        yi = y_true_bin[:, i]
        si = y_score[:, i]
        if yi.sum() == 0 or yi.sum() == len(yi):
            fpr_d[i], tpr_d[i] = np.array([0.0, 1.0]), np.array([0.0, 1.0])
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
def vae_script(inp: str, prefix: str):
    # Load
    data = pd.read_csv(inp)
    sample_ids = data["SampleID"].values
    X = data.drop(columns=["SampleID", "Label"]).values
    y_raw = data["Label"].values
    feature_names = data.drop(columns=["SampleID", "Label"]).columns

    # Labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    num_classes = len(np.unique(y))

    # ---------------- Nested CV (only F1/AUC per outer fold line plot) ----------------
    outer_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    per_fold_f1 = []
    per_fold_auc = []
    outer_fold_idx = 0

    for outer_tr_idx, outer_te_idx in outer_skf.split(X, y):
        outer_fold_idx += 1
        X_tr_outer_raw, X_te_outer_raw = X[outer_tr_idx], X[outer_te_idx]
        y_tr_outer, y_te_outer = y[outer_tr_idx], y[outer_te_idx]

        # Inner study: scale INSIDE each inner fold
        def objective(trial: optuna.Trial) -> float:
            set_seed(1234)
            # Hyperparams
            latent_dim = trial.suggest_int("latent_dim", 2, 512, log=True)

            num_encoder_layers = trial.suggest_int("num_encoder_layers", 1, 5)
            encoder_layers = [trial.suggest_int(f"enc_units_l{i}", 16, 512, log=True) for i in range(num_encoder_layers)]

            num_decoder_layers = trial.suggest_int("num_decoder_layers", 1, 5)
            decoder_layers = [trial.suggest_int(f"dec_units_l{i}", 16, 512, log=True) for i in range(num_decoder_layers)]

            num_mlp_layers = trial.suggest_int("num_mlp_layers", 1, 5)
            mlp_layers = [trial.suggest_int(f"mlp_units_l{i}", 16, 512, log=True) for i in range(num_mlp_layers)]

            enc_norm = trial.suggest_categorical("enc_norm", ["batch", "layer", "none"])
            mlp_norm = trial.suggest_categorical("mlp_norm", ["batch", "layer", "none"])

            vae_dropout = trial.suggest_float("vae_dropout", 0.1, 0.6)
            mlp_dropout = trial.suggest_float("mlp_dropout", 0.0, 0.6)

            vae_lr = trial.suggest_float("vae_learning_rate", 1e-5, 5e-2, log=True)
            mlp_lr = trial.suggest_float("mlp_learning_rate", 1e-5, 5e-2, log=True)

            vae_weight_decay = trial.suggest_float("vae_weight_decay", 1e-6, 1e-3, log=True)
            mlp_weight_decay = trial.suggest_float("mlp_weight_decay", 1e-6, 1e-3, log=True)

            scheduler_factor = trial.suggest_float("scheduler_factor", 0.1, 0.7)
            scheduler_patience = trial.suggest_int("scheduler_patience", 2, 10)
            scheduler_min_lr = trial.suggest_float("scheduler_min_lr", 1e-6, 1e-4, log=True)

            max_grad_norm = trial.suggest_categorical("max_grad_norm", [None, 1.0, 5.0])

            epochs = trial.suggest_int("epochs", 10, 200)
            batch_size = trial.suggest_int("batch_size", 2, 256, log=True)

            vae_es_pat = trial.suggest_int("vae_es_patience", 5, 20)
            mlp_es_pat = trial.suggest_int("mlp_es_patience", 5, 20)
            vae_es_delta = trial.suggest_float("vae_es_delta", 0.0, 0.1, step=0.01)
            mlp_es_delta = trial.suggest_float("mlp_es_delta", 0.0, 0.1, step=0.01)

            mlp_es_metric = trial.suggest_categorical("mlp_es_metric", ["loss", "macro_f1"])

            inner_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = []

            for k, (inn_tr_idx, inn_va_idx) in enumerate(inner_skf.split(X_tr_outer_raw, y_tr_outer)):
                set_seed(2020 + k)
                scaler_in = StandardScaler()
                X_tr_in = scaler_in.fit_transform(X_tr_outer_raw[inn_tr_idx])
                X_va_in = scaler_in.transform(X_tr_outer_raw[inn_va_idx])
                y_tr_in = y_tr_outer[inn_tr_idx]
                y_va_in = y_tr_outer[inn_va_idx]

                model = VAE_MLP(
                    input_dim=X.shape[1],
                    num_classes=num_classes,
                    latent_dim=latent_dim,
                    encoder_layers=encoder_layers,
                    decoder_layers=decoder_layers,
                    mlp_layers=mlp_layers,
                    enc_norm=enc_norm,
                    mlp_norm=mlp_norm,
                    vae_dropout=vae_dropout,
                    mlp_dropout=mlp_dropout,
                    vae_early_stopping_patience=vae_es_pat,
                    vae_early_stopping_min_delta=vae_es_delta,
                    mlp_early_stopping_patience=mlp_es_pat,
                    mlp_early_stopping_min_delta=mlp_es_delta,
                    mlp_early_stopping_metric=mlp_es_metric,
                    vae_learning_rate=vae_lr,
                    mlp_learning_rate=mlp_lr,
                    vae_weight_decay=vae_weight_decay,
                    mlp_weight_decay=mlp_weight_decay,
                    scheduler_factor=scheduler_factor,
                    scheduler_patience=scheduler_patience,
                    scheduler_min_lr=scheduler_min_lr,
                    max_grad_norm=max_grad_norm,
                    epochs=epochs,
                    batch_size=batch_size,
                )
                model.fit(X_tr_in, y_tr_in)
                y_pred = model.predict(X_va_in)
                score = f1_score(y_va_in, y_pred, average="macro" if num_classes > 2 else "binary")
                scores.append(score)

            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        with SuppressOutput():
            study.optimize(objective, n_trials=20, n_jobs=1)

        bp = study.best_params

        scaler_outer = StandardScaler()
        X_tr_outer = scaler_outer.fit_transform(X_tr_outer_raw)
        X_te_outer = scaler_outer.transform(X_te_outer_raw)

        best_model = VAE_MLP(
            input_dim=X.shape[1],
            num_classes=num_classes,
            latent_dim=bp["latent_dim"],
            encoder_layers=[bp[f"enc_units_l{i}"] for i in range(bp["num_encoder_layers"])],
            decoder_layers=[bp[f"dec_units_l{i}"] for i in range(bp["num_decoder_layers"])],
            mlp_layers=[bp[f"mlp_units_l{i}"] for i in range(bp["num_mlp_layers"])],
            enc_norm=bp["enc_norm"],
            mlp_norm=bp["mlp_norm"],
            vae_dropout=bp["vae_dropout"],
            mlp_dropout=bp["mlp_dropout"],
            vae_early_stopping_patience=bp["vae_es_patience"],
            vae_early_stopping_min_delta=bp["vae_es_delta"],
            mlp_early_stopping_patience=bp["mlp_es_patience"],
            mlp_early_stopping_min_delta=bp["mlp_es_delta"],
            mlp_early_stopping_metric=bp["mlp_es_metric"],
            vae_learning_rate=bp["vae_learning_rate"],
            mlp_learning_rate=bp["mlp_learning_rate"],
            vae_weight_decay=bp["vae_weight_decay"],
            mlp_weight_decay=bp["mlp_weight_decay"],
            scheduler_factor=bp["scheduler_factor"],
            scheduler_patience=bp["scheduler_patience"],
            scheduler_min_lr=bp["scheduler_min_lr"],
            max_grad_norm=bp["max_grad_norm"],
            epochs=bp["epochs"],
            batch_size=bp["batch_size"],
        )
        best_model.fit(X_tr_outer, y_tr_outer)
        y_pred_outer = best_model.predict(X_te_outer)
        y_proba_outer = best_model.predict_proba(X_te_outer)

        f1_fold = f1_score(y_te_outer, y_pred_outer, average="macro" if num_classes > 2 else "binary")
        if num_classes == 2:
            auc_fold = roc_auc_score(y_te_outer, y_proba_outer[:, 1])
        else:
            auc_fold = roc_auc_score(y_te_outer, y_proba_outer, multi_class="ovr")
        per_fold_f1.append(f1_fold)
        per_fold_auc.append(auc_fold)

    # Plot nested CV F1/AUC per fold
    folds = np.arange(1, len(per_fold_f1) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(folds, per_fold_f1, marker="o", linestyle="-", label="F1 Score")
    plt.plot(folds, per_fold_auc, marker="s", linestyle="-", label="AUC")
    plt.title("F1 and AUC Scores per Outer Fold (Nested CV)", fontsize=TITLE_FONTSIZE, fontweight="bold", pad=15)
    plt.xlabel("Outer Fold Number", fontsize=LABEL_FONTSIZE, labelpad=10)
    plt.ylabel("Score", fontsize=LABEL_FONTSIZE, labelpad=10)
    plt.xticks(folds, fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.ylim(0.0, 1.05)
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{prefix}_vae_nested_cv_f1_auc.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ---------------- Whole-data tuning + final evaluation/saving ----------------
    def full_data_objective(trial: optuna.Trial) -> float:
        set_seed(7777)
        latent_dim = trial.suggest_int("latent_dim", 2, 512, log=True)

        num_encoder_layers = trial.suggest_int("num_encoder_layers", 1, 5)
        encoder_layers = [trial.suggest_int(f"enc_units_l{i}", 16, 512, log=True) for i in range(num_encoder_layers)]

        num_decoder_layers = trial.suggest_int("num_decoder_layers", 1, 5)
        decoder_layers = [trial.suggest_int(f"dec_units_l{i}", 16, 512, log=True) for i in range(num_decoder_layers)]

        num_mlp_layers = trial.suggest_int("num_mlp_layers", 1, 5)
        mlp_layers = [trial.suggest_int(f"mlp_units_l{i}", 16, 512, log=True) for i in range(num_mlp_layers)]

        enc_norm = trial.suggest_categorical("enc_norm", ["batch", "layer", "none"])
        mlp_norm = trial.suggest_categorical("mlp_norm", ["batch", "layer", "none"])

        vae_dropout = trial.suggest_float("vae_dropout", 0.1, 0.6)
        mlp_dropout = trial.suggest_float("mlp_dropout", 0.0, 0.6)

        vae_lr = trial.suggest_float("vae_learning_rate", 1e-5, 5e-2, log=True)
        mlp_lr = trial.suggest_float("mlp_learning_rate", 1e-5, 5e-2, log=True)

        vae_weight_decay = trial.suggest_float("vae_weight_decay", 1e-6, 1e-3, log=True)
        mlp_weight_decay = trial.suggest_float("mlp_weight_decay", 1e-6, 1e-3, log=True)

        scheduler_factor = trial.suggest_float("scheduler_factor", 0.1, 0.7)
        scheduler_patience = trial.suggest_int("scheduler_patience", 2, 10)
        scheduler_min_lr = trial.suggest_float("scheduler_min_lr", 1e-6, 1e-4, log=True)

        max_grad_norm = trial.suggest_categorical("max_grad_norm", [None, 1.0, 5.0])

        epochs = trial.suggest_int("epochs", 10, 200)
        batch_size = trial.suggest_int("batch_size", 2, 256, log=True)

        vae_es_pat = trial.suggest_int("vae_es_patience", 5, 20)
        mlp_es_pat = trial.suggest_int("mlp_es_patience", 5, 20)
        vae_es_delta = trial.suggest_float("vae_es_delta", 0.0, 0.1, step=0.01)
        mlp_es_delta = trial.suggest_float("mlp_es_delta", 0.0, 0.1, step=0.01)

        mlp_es_metric = trial.suggest_categorical("mlp_es_metric", ["loss", "macro_f1"])

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        f1_scores = []

        for k, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
            set_seed(9000 + k)
            scaler_in = StandardScaler()
            X_tr_in = scaler_in.fit_transform(X[tr_idx])
            X_va_in = scaler_in.transform(X[va_idx])
            y_tr_in = y[tr_idx]
            y_va_in = y[va_idx]

            model = VAE_MLP(
                input_dim=X.shape[1],
                num_classes=num_classes,
                latent_dim=latent_dim,
                encoder_layers=encoder_layers,
                decoder_layers=decoder_layers,
                mlp_layers=mlp_layers,
                enc_norm=enc_norm,
                mlp_norm=mlp_norm,
                vae_dropout=vae_dropout,
                mlp_dropout=mlp_dropout,
                vae_early_stopping_patience=vae_es_pat,
                vae_early_stopping_min_delta=vae_es_delta,
                mlp_early_stopping_patience=mlp_es_pat,
                mlp_early_stopping_min_delta=mlp_es_delta,
                mlp_early_stopping_metric=mlp_es_metric,
                vae_learning_rate=vae_lr,
                mlp_learning_rate=mlp_lr,
                vae_weight_decay=vae_weight_decay,
                mlp_weight_decay=mlp_weight_decay,
                scheduler_factor=scheduler_factor,
                scheduler_patience=scheduler_patience,
                scheduler_min_lr=scheduler_min_lr,
                max_grad_norm=max_grad_norm,
                epochs=epochs,
                batch_size=batch_size,
            )
            model.fit(X_tr_in, y_tr_in)
            y_pred = model.predict(X_va_in)
            f1 = f1_score(y_va_in, y_pred, average="macro" if num_classes > 2 else "binary")
            f1_scores.append(f1)

        return float(np.mean(f1_scores))

    study_full = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    with SuppressOutput():
        study_full.optimize(full_data_objective, n_trials=30, n_jobs=1)
    best = study_full.best_params

    # Fit scaler on the entire dataset
    scaler_final = StandardScaler()
    X_scaled_final = scaler_final.fit_transform(X)

    final_model = VAE_MLP(
        input_dim=X.shape[1],
        num_classes=num_classes,
        latent_dim=best["latent_dim"],
        encoder_layers=[best[f"enc_units_l{i}"] for i in range(best["num_encoder_layers"])],
        decoder_layers=[best[f"dec_units_l{i}"] for i in range(best["num_decoder_layers"])],
        mlp_layers=[best[f"mlp_units_l{i}"] for i in range(best["num_mlp_layers"])],
        enc_norm=best["enc_norm"],
        mlp_norm=best["mlp_norm"],
        vae_dropout=best["vae_dropout"],
        mlp_dropout=best["mlp_dropout"],
        vae_early_stopping_patience=best["vae_es_patience"],
        vae_early_stopping_min_delta=best["vae_es_delta"],
        mlp_early_stopping_patience=best["mlp_es_patience"],
        mlp_early_stopping_min_delta=best["mlp_es_delta"],
        mlp_early_stopping_metric=best["mlp_es_metric"],
        vae_learning_rate=best["vae_learning_rate"],
        mlp_learning_rate=best["mlp_learning_rate"],
        vae_weight_decay=best["vae_weight_decay"],
        mlp_weight_decay=best["mlp_weight_decay"],
        scheduler_factor=best["scheduler_factor"],
        scheduler_patience=best["scheduler_patience"],
        scheduler_min_lr=best["scheduler_min_lr"],
        max_grad_norm=best["max_grad_norm"],
        epochs=best["epochs"],
        batch_size=best["batch_size"],
    )
    final_model.fit(X_scaled_final, y)

    # Predictions on full data (per requirement)
    y_pred_final = final_model.predict(X_scaled_final)
    y_proba_final = final_model.predict_proba(X_scaled_final)

    # Confusion matrix with integer annotations (no scientific notation)
    cm = confusion_matrix(y, y_pred_final, labels=np.arange(num_classes))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, include_values=False, colorbar=False)
    ax.set_title("Confusion Matrix", fontsize=TITLE_FONTSIZE, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted label", fontsize=LABEL_FONTSIZE, labelpad=10)
    ax.set_ylabel("True label", fontsize=LABEL_FONTSIZE, labelpad=10)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    # manual integer annotations
    cm_array = np.asarray(cm)
    for i in range(cm_array.shape[0]):
        for j in range(cm_array.shape[1]):
            ax.text(
                j,
                i,
                f"{int(cm_array[i, j])}",
                ha="center",
                va="center",
                fontsize=TICK_FONTSIZE,
                color="black",
            )
    plt.tight_layout()
    plt.savefig(f"{prefix}_vae_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Metrics (no AUC bar in metrics figure)
    accuracy = accuracy_score(y, y_pred_final)
    f1_final = f1_score(y, y_pred_final, average="macro" if num_classes > 2 else "binary")

    if num_classes == 2:
        # For reporting AUC elsewhere, still compute it, but not plotted as a metrics bar
        _ = roc_auc_score(y, y_proba_final[:, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        _ = roc_auc_score(y, y_proba_final, multi_class="ovr")
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

    metrics_df = pd.DataFrame(
        {
            "Metric": ["Accuracy", "F1 Score", "Sensitivity", "Specificity"],
            "Score": [accuracy, f1_final, sensitivity, specificity],
        }
    )

    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics_df["Metric"], metrics_df["Score"])
    ymax = max(metrics_df["Score"]) if len(metrics_df["Score"]) > 0 else 1.0
    plt.ylim(0.0, max(1.05, ymax + 0.08))
    for b, s in zip(bars, metrics_df["Score"]):
        yv = b.get_height()
        plt.text(
            b.get_x() + b.get_width() / 2,
            yv + 0.02,
            f"{s:.3f}",
            ha="center",
            va="bottom",
            fontsize=TICK_FONTSIZE,
        )
    plt.title("Performance Metrics", fontsize=TITLE_FONTSIZE, fontweight="bold", pad=15)
    plt.ylabel("Score", fontsize=LABEL_FONTSIZE, labelpad=10)
    plt.xlabel("Metric", fontsize=LABEL_FONTSIZE, labelpad=10)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.tight_layout()
    plt.savefig(f"{prefix}_vae_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ROC curves: per-class + micro + macro (multi-class); single curve (binary)
    fpr = {}
    tpr = {}
    roc_auc = {}

    if num_classes == 2:
        fpr[0], tpr[0], _ = roc_curve(y, y_proba_final[:, 1])
        roc_auc[0] = auc(fpr[0], tpr[0])
    else:
        y_true_bin = np.eye(num_classes)[y]
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba_final[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_proba_final.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        fpr_macro, tpr_macro, auc_macro, _ = compute_macro_roc(y_true_bin, y_proba_final)
        fpr["macro"], tpr["macro"], roc_auc["macro"] = fpr_macro, tpr_macro, auc_macro

    # Save ROC data (explicit allow_pickle)
    roc_data = {"fpr": fpr, "tpr": tpr, "roc_auc": roc_auc}
    np.save(f"{prefix}_vae_roc_data.npy", roc_data, allow_pickle=True)

    # Plot ROC
    plt.figure(figsize=(10, 8))
    if num_classes == 2:
        plt.plot(fpr[0], tpr[0], label=f"ROC (AUC = {roc_auc[0]:.2f})")
    else:
        for i in range(num_classes):
            plt.plot(fpr[i], tpr[i], label=f'Class {le.inverse_transform([i])[0]} (AUC = {roc_auc[i]:.2f})')
        if "micro" in roc_auc:
            plt.plot(fpr["micro"], tpr["micro"], linestyle="--", label=f"Micro-average (AUC = {roc_auc['micro']:.2f})")
        if "macro" in roc_auc:
            plt.plot(fpr["macro"], tpr["macro"], linestyle="-.", label=f"Macro-average (AUC = {roc_auc['macro']:.2f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=LABEL_FONTSIZE, labelpad=10)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=LABEL_FONTSIZE, labelpad=10)
    plt.title("ROC Curves", fontsize=TITLE_FONTSIZE, fontweight="bold", pad=15)
    plt.legend(loc="lower right", fontsize=LEGEND_FONTSIZE)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    plt.tight_layout()
    plt.savefig(f"{prefix}_vae_roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Predictions CSV (final model on full data)
    results_df = pd.DataFrame(
        {
            "SampleID": sample_ids,
            "Original Label": le.inverse_transform(y),
            "Predicted Label": le.inverse_transform(y_pred_final),
        }
    )
    results_df.to_csv(f"{prefix}_vae_predictions.csv", index=False)

    # Save final artifacts
    joblib.dump(final_model, f"{prefix}_vae_model.pkl")
    joblib.dump(scaler_final, f"{prefix}_vae_scaler.pkl")
    joblib.dump((X, y, le), f"{prefix}_vae_data.pkl")

    # ---------------- SHAP (robust to different return types) ----------------
    try:
        background_size = min(100, X_scaled_final.shape[0])
        bg_idx = np.random.choice(X_scaled_final.shape[0], background_size, replace=False)
        background = X_scaled_final[bg_idx]

        def predict_proba_fn(xinp: np.ndarray) -> np.ndarray:
            return final_model.predict_proba(xinp)

        explainer = shap.PermutationExplainer(
            predict_proba_fn, background, n_repeats=10, random_state=42, n_jobs=-1
        )
        shap_values = explainer.shap_values(X_scaled_final)

        # Normalize to shape: (n_samples, n_features, n_outputs) when possible
        n_features = X.shape[1]
        if isinstance(shap_values, list):
            try:
                sv = np.stack(shap_values, axis=-1)
            except Exception:
                sv = np.stack([np.asarray(s) for s in shap_values], axis=-1)
        else:
            sv = np.asarray(shap_values)
            if sv.ndim == 2:
                sv = sv[:, :, None]

        if num_classes == 2:
            out_idx = 1 if sv.shape[2] >= 2 else 0
            sv_pos = sv[:, :, out_idx]
            mean_shap = np.mean(sv_pos, axis=0)
            shap_df = pd.DataFrame({"Feature": feature_names, "Mean SHAP Value": mean_shap})
            shap_df["Global Mean"] = np.mean(np.abs(sv_pos), axis=0)
            shap_df.to_csv(f"{prefix}_vae_shap_values.csv", index=False)
        else:
            if sv.shape[2] == num_classes:
                per_class = {}
                for c in range(num_classes):
                    cname = le.inverse_transform([c])[0]
                    per_class[cname] = np.mean(sv[:, :, c], axis=0)
                shap_per_class_df = pd.DataFrame(per_class, index=feature_names).reset_index().rename(columns={"index": "Feature"})
                shap_per_class_df["Global Mean"] = np.mean(np.abs(sv), axis=(0, 2))
                shap_per_class_df.to_csv(f"{prefix}_vae_shap_values.csv", index=False)
        
                overall = np.mean(np.abs(sv), axis=(0, 2))
                shap_mean_df = pd.DataFrame({"Feature": feature_names, "Mean SHAP Value": overall})
                shap_mean_df.to_csv(f"{prefix}_vae_shap_values_mean.csv", index=False)
            else:
                sv0 = sv[:, :, 0]
                signed_mean = np.mean(sv0, axis=0)
                shap_df = pd.DataFrame({"Feature": feature_names, "Mean SHAP Value": signed_mean})
                shap_df["Global Mean"] = np.mean(np.abs(sv0), axis=0)
                shap_df.to_csv(f"{prefix}_vae_shap_values.csv", index=False)
        
                overall = np.mean(np.abs(sv0), axis=0)
                shap_mean_df = pd.DataFrame({"Feature": feature_names, "Mean SHAP Value": overall})
                shap_mean_df.to_csv(f"{prefix}_vae_shap_values_mean.csv", index=False)
    except Exception as e:
        print(f"SHAP computation failed: {e}")
        print("Proceeding without SHAP analysis.")


if __name__ == "__main__":
    args = parse_arguments()
    pref = Path(args.csv).stem
    if args.prefix:
        pref = args.prefix
    vae_script(args.csv, pref)
