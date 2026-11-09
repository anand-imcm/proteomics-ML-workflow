import argparse
import random
from pathlib import Path
import warnings
import sys

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_curve,
    auc,
    confusion_matrix,
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


SCRIPT_ROOT = str(Path(__file__).resolve().parents[1])
if SCRIPT_ROOT not in sys.path:
    sys.path.insert(0, SCRIPT_ROOT)
from core.data_contracts import RoleAwarePreprocessor, modeling_frame, resolve_feature_roles, write_missingness_qc, write_reproducibility_manifest, imputation_fallback_reasons
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
    parser = argparse.ArgumentParser(description="VAE -> reconstruct X, then MLP classification")
    parser.add_argument("--input", "-i", "--csv", dest="csv", type=str, help="Input CSV file", required=True)
    parser.add_argument("-p", "--prefix", type=str, help="Output prefix")
    parser.add_argument("--outdir", default=".", help="Base output directory")
    parser.add_argument("--feature_selection", choices=["yes", "no"], default="no", help="Accepted for CLI consistency")
    parser.add_argument("--jobs", type=int, default=1, help="Accepted for CLI consistency")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--imputation", choices=["median", "knn"], default="median")
    parser.add_argument("--covariates", nargs="*", default=None)
    parser.add_argument("--categorical-covariates", nargs="*", default=None)
    return parser.parse_args()


def set_seed(seed: int = DEFAULT_RANDOM_STATE):
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

TITLE_FONTSIZE = 24
LABEL_FONTSIZE = 20
TICK_FONTSIZE = 16
LEGEND_FONTSIZE = 16

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
        set_seed(DEFAULT_RANDOM_STATE)
        self.classes_ = np.unique(y)
        self.training_history_ = []

        self._create_vae()

        X_tr_vae, X_val_vae = train_test_split(X, test_size=0.2, random_state=DEFAULT_RANDOM_STATE, shuffle=True)
        X_tr_vae_t = torch.from_numpy(X_tr_vae).float().to(self.device)
        X_val_vae_t = torch.from_numpy(X_val_vae).float().to(self.device)

        vae_train_batch_size = min(self.batch_size, max(1, len(X_tr_vae)))
        dl_tr_vae = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_tr_vae_t),
            batch_size=vae_train_batch_size,
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
        for epoch in range(self.epochs):
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
            train_mean = train_sum / max(train_count, 1)

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
            stopped = es_vae(val_mean, epoch + 1, checkpoint_modules=(self.vae,))
            self.training_history_.append({
                "Epoch": epoch + 1,
                "TrainingLoss": train_mean,
                "ValidationLoss": val_mean,
                "LearningRate": self.vae_optimizer.param_groups[0]["lr"],
                "EarlyStoppingTriggered": stopped,
                "BestEpoch": es_vae.best_epoch,
            })
            if stopped:
                break
            self.vae.train()

        es_vae.restore_checkpoint(self.vae)
        self.vae.eval()
        with torch.no_grad():
            ds = torch.utils.data.TensorDataset(torch.from_numpy(X).float().to(self.device))
            dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=False, drop_last=False)
            X_recon_list = []
            for (xb,) in dl:
                x_recon, _, _ = self.vae(xb)
                X_recon_list.append(x_recon.cpu())
            X_recon = torch.cat(X_recon_list, dim=0).numpy()

        self._create_mlp()

        X_tr_mlp, X_val_mlp, y_tr_mlp, y_val_mlp = train_test_split(
            X_recon, y, test_size=0.2, random_state=DEFAULT_RANDOM_STATE, stratify=y
        )
        X_tr_mlp_t = torch.from_numpy(X_tr_mlp).float().to(self.device)
        y_tr_mlp_t = torch.from_numpy(y_tr_mlp).long().to(self.device)
        X_val_mlp_t = torch.from_numpy(X_val_mlp).float().to(self.device)
        y_val_mlp_t = torch.from_numpy(y_val_mlp).long().to(self.device)

        mlp_train_batch_size = min(self.batch_size, max(1, len(X_tr_mlp)))
        dl_tr_mlp = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_tr_mlp_t, y_tr_mlp_t),
            batch_size=mlp_train_batch_size,
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
        offset = len(self.training_history_)
        for epoch in range(self.epochs):
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
            train_mean = tr_sum / max(tr_count, 1)

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

            stopped = es_mlp(monitor, epoch + 1, checkpoint_modules=(self.mlp,))
            self.training_history_.append({
                "Epoch": offset + epoch + 1,
                "TrainingLoss": train_mean,
                "ValidationLoss": va_mean,
                "LearningRate": self.mlp_optimizer.param_groups[0]["lr"],
                "EarlyStoppingTriggered": stopped,
                "BestEpoch": offset + es_mlp.best_epoch,
            })
            if stopped:
                break
            self.mlp.train()

        es_mlp.restore_checkpoint(self.mlp)
        return self

    def _inference_logits(self, X: np.ndarray, inference_batch_size: int | None = None) -> torch.Tensor:
        """Run the fitted stochastic reconstruction and classifier in a single batched path."""
        batch_size = int(inference_batch_size or self.batch_size)
        X_tensor = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.device)
        logits = []
        # Preserve the two base-seed draws made by the legacy inference DataLoaders.
        torch.empty((), dtype=torch.int64).random_()
        self.vae.eval()
        self.mlp.eval()
        with torch.inference_mode():
            for start in range(0, X_tensor.shape[0], batch_size):
                x_recon, _, _ = self.vae(X_tensor[start:start + batch_size])
                logits.append(self.mlp(x_recon).cpu())
        torch.empty((), dtype=torch.int64).random_()
        if not logits:
            return torch.empty((0, self.num_classes), dtype=torch.float32)
        return torch.cat(logits, dim=0)

    def predict(self, X: np.ndarray, inference_batch_size: int | None = None) -> np.ndarray:
        logits = self._inference_logits(X, inference_batch_size)
        return torch.argmax(logits, dim=1).numpy() if len(logits) else np.array([])

    def predict_proba(self, X: np.ndarray, inference_batch_size: int | None = None) -> np.ndarray:
        logits = self._inference_logits(X, inference_batch_size)
        return torch.softmax(logits, dim=1).numpy() if len(logits) else np.empty((0, self.num_classes))


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


def _classification_sensitivity_specificity(cm: np.ndarray, num_classes: int) -> tuple[float, float]:
    if np.asarray(cm).shape != (num_classes, num_classes):
        return float("nan"), float("nan")
    return sensitivity_specificity_from_confusion_matrix(cm)


def vae_script(inp: str, prefix: str, imputation="median", covariates=None, categorical_covariates=None):
    # Load
    data = pd.read_csv(inp)
    roles = resolve_feature_roles(data, target="Label", identifier="SampleID", covariates=covariates, categorical_covariates=categorical_covariates)
    sample_ids = data["SampleID"].values if "SampleID" in data.columns else np.asarray([f"Sample_{i+1}" for i in range(len(data))])
    X = modeling_frame(data, roles)
    y_raw = data["Label"].values
    feature_names = list(roles.biomarkers)

    def make_preprocessor():
        return RoleAwarePreprocessor(roles.biomarkers, roles.numeric_covariates, roles.categorical_covariates, imputation=imputation)

    # Labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    num_classes = len(np.unique(y))

    outer_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
    per_fold_f1 = []
    per_fold_auc = []
    fold_rows = []
    oof_true = []
    oof_pred = []
    oof_proba = []
    oof_index = []
    outer_fold_idx = 0
    missingness_folds = []
    tuning_reports = []

    for outer_tr_idx, outer_te_idx in outer_skf.split(X, y):
        outer_fold_idx += 1
        X_tr_outer_raw, X_te_outer_raw = X.iloc[outer_tr_idx], X.iloc[outer_te_idx]
        y_tr_outer, y_te_outer = y[outer_tr_idx], y[outer_te_idx]

        # Inner study: scale INSIDE each inner fold
        def objective(trial: optuna.Trial) -> float:
            set_seed(DEFAULT_RANDOM_STATE)
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

            inner_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
            scores = []

            for k, (inn_tr_idx, inn_va_idx) in enumerate(inner_skf.split(X_tr_outer_raw, y_tr_outer)):
                set_seed(DEFAULT_RANDOM_STATE)
                inner_preprocessor = make_preprocessor()
                y_tr_in = y_tr_outer[inn_tr_idx]
                y_va_in = y_tr_outer[inn_va_idx]
                X_tr_in = inner_preprocessor.fit_transform(X_tr_outer_raw.iloc[inn_tr_idx], y_tr_in)
                X_va_in = inner_preprocessor.transform(X_tr_outer_raw.iloc[inn_va_idx])

                model = VAE_MLP(
                    input_dim=X_tr_in.shape[1],
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
                model.fit(np.asarray(X_tr_in), y_tr_in)
                y_pred = model.predict(np.asarray(X_va_in))
                score = f1_score(y_va_in, y_pred, average="macro" if num_classes > 2 else "binary")
                scores.append(score)

            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=DEFAULT_RANDOM_STATE))
        with SuppressOutput():
            tuning_reports.append(
                run_adaptive_study(
                    study,
                    objective,
                    task="classification",
                    model="VAE",
                    study_scope="outer_fold",
                    objective_name="vae_classification_f1",
                    direction="maximize",
                    tolerance=tolerance_for_objective("vae_classification_f1"),
                    seed=DEFAULT_RANDOM_STATE,
                    study_id=f"classification-vae-outer-fold-{outer_fold_idx}",
                    outer_fold=outer_fold_idx,
                    provenance_path=f"{prefix}_vae_fold{outer_fold_idx}_optuna_study.json",
                )
            )

        bp = study.best_params

        outer_preprocessor = make_preprocessor()
        X_tr_outer = outer_preprocessor.fit_transform(X_tr_outer_raw, y_tr_outer)
        X_te_outer = outer_preprocessor.transform(X_te_outer_raw)
        missingness_folds.append(outer_preprocessor.biomarker_diagnostics(outer_fold_idx))

        best_model = VAE_MLP(
            input_dim=X_tr_outer.shape[1],
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
        best_model.fit(np.asarray(X_tr_outer), y_tr_outer)
        fold_importance, fold_importance_type = estimator_importance_with_source(best_model, X_tr_outer.shape[1])
        fold_biomarkers = outer_preprocessor.selected_biomarker_names_
        export_outer_fold_biomarkers(prefix, "vae", outer_fold_idx, None, fold_biomarkers, selection_method="None", importance_values=np.asarray(fold_importance)[:len(fold_biomarkers)], importance_type=fold_importance_type)
        y_pred_outer = best_model.predict(np.asarray(X_te_outer))
        y_proba_outer = best_model.predict_proba(np.asarray(X_te_outer))

        f1_fold = f1_score(y_te_outer, y_pred_outer, average="macro" if num_classes > 2 else "binary")
        weighted_f1_fold = f1_score(y_te_outer, y_pred_outer, average="weighted", zero_division=0)
        accuracy_fold = accuracy_score(y_te_outer, y_pred_outer)
        if num_classes == 2:
            auc_fold = roc_auc_score(y_te_outer, y_proba_outer[:, 1])
            auc_micro_fold = auc_fold
        else:
            auc_fold = roc_auc_score(y_te_outer, y_proba_outer, multi_class="ovr")
            y_te_bin = np.eye(num_classes)[y_te_outer]
            auc_micro_fold = roc_auc_score(y_te_bin.ravel(), y_proba_outer.ravel())
        cm_fold = confusion_matrix(y_te_outer, y_pred_outer, labels=np.arange(num_classes))
        sensitivity_fold, specificity_fold = _classification_sensitivity_specificity(cm_fold, num_classes)
        fold_rows.append(
            {
                "Model": "VAE",
                "Fold": outer_fold_idx,
                "F1": f1_fold,
                "weighted_F1": weighted_f1_fold,
                "Accuracy": accuracy_fold,
                "Macro_AUC": auc_fold,
                "Micro_AUC": auc_micro_fold,
                "AUC": auc_fold,
                "Sensitivity": sensitivity_fold,
                "Specificity": specificity_fold,
            }
        )
        per_fold_f1.append(f1_fold)
        per_fold_auc.append(auc_fold)
        oof_true.append(y_te_outer)
        oof_pred.append(y_pred_outer)
        oof_proba.append(y_proba_outer)
        oof_index.append(outer_te_idx)

    folds = np.arange(1, len(per_fold_f1) + 1)
    plot_classification_nested_cv(folds, per_fold_f1, per_fold_auc, "VAE Nested CV", f"{prefix}_vae_nested_cv_f1_auc.png", f1_label="F1")
    pd.DataFrame(fold_rows).to_csv(f"{prefix}_vae_outer_fold_metrics.csv", index=False)

    order = np.argsort(np.concatenate(oof_index))
    y_oof = np.concatenate(oof_true)[order]
    y_pred_oof = np.concatenate(oof_pred)[order]
    y_proba_oof = np.vstack(oof_proba)[order]
    oof_fold = np.concatenate(
        [np.full(len(indices), fold, dtype=int) for fold, indices in enumerate(oof_index, start=1)]
    )[order]

    def full_data_objective(trial: optuna.Trial) -> float:
        set_seed(DEFAULT_RANDOM_STATE)
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

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
        f1_scores = []

        for k, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
            set_seed(DEFAULT_RANDOM_STATE)
            inner_preprocessor = make_preprocessor()
            y_tr_in = y[tr_idx]
            y_va_in = y[va_idx]
            X_tr_in = inner_preprocessor.fit_transform(X.iloc[tr_idx], y_tr_in)
            X_va_in = inner_preprocessor.transform(X.iloc[va_idx])

            model = VAE_MLP(
                input_dim=X_tr_in.shape[1],
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
            model.fit(np.asarray(X_tr_in), y_tr_in)
            y_pred = model.predict(np.asarray(X_va_in))
            f1 = f1_score(y_va_in, y_pred, average="macro" if num_classes > 2 else "binary")
            f1_scores.append(f1)

        return float(np.mean(f1_scores))

    study_full = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=DEFAULT_RANDOM_STATE))
    with SuppressOutput():
        tuning_reports.append(
            run_adaptive_study(
                study_full,
                full_data_objective,
                task="classification",
                model="VAE",
                study_scope="final_full_development",
                objective_name="vae_classification_f1",
                direction="maximize",
                tolerance=tolerance_for_objective("vae_classification_f1"),
                seed=DEFAULT_RANDOM_STATE,
                study_id="classification-vae-final",
                provenance_path=f"{prefix}_vae_final_optuna_study.json",
            )
        )
    best = study_full.best_params

    final_preprocessor = make_preprocessor()
    X_scaled_final = final_preprocessor.fit_transform(X, y)
    write_missingness_qc(prefix + "_vae", X, roles, missingness_folds)

    def make_final_model():
        set_seed(DEFAULT_RANDOM_STATE)
        return VAE_MLP(
            input_dim=X_scaled_final.shape[1],
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

    final_model = make_final_model()
    final_model.fit(np.asarray(X_scaled_final), y)
    write_training_history(getattr(final_model, "training_history_", []), prefix, "vae")
    generate_training_size_curve(np.asarray(X_scaled_final), y, make_final_model, prefix, "vae", task="classification")

    cm = confusion_matrix(y_oof, y_pred_oof, labels=np.arange(num_classes))
    plot_confusion_matrix(cm, le.classes_, "VAE Confusion Matrix", f"{prefix}_vae_confusion_matrix.png")

    accuracy = accuracy_score(y_oof, y_pred_oof)
    f1_final = f1_score(y_oof, y_pred_oof, average="macro" if num_classes > 2 else "binary", zero_division=0)
    weighted_f1 = f1_score(y_oof, y_pred_oof, average="weighted", zero_division=0)

    sensitivity, specificity = _classification_sensitivity_specificity(cm, num_classes)
    if num_classes == 2:
        macro_auc = roc_auc_score(y_oof, y_proba_oof[:, 1])
        micro_auc = macro_auc
    else:
        macro_auc = roc_auc_score(y_oof, y_proba_oof, multi_class="ovr")
        y_true_bin = np.eye(num_classes)[y_oof]
        micro_auc = roc_auc_score(y_true_bin.ravel(), y_proba_oof.ravel())

    metrics_df = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Weighted F1", "Sensitivity", "Specificity"],
            "Score": [accuracy, weighted_f1, sensitivity, specificity],
        }
    )

    plot_metric_bars(metrics_df["Metric"], metrics_df["Score"], "VAE Performance", f"{prefix}_vae_metrics.png", ylim=(0.0, 1.12))
    pd.DataFrame(
        [
            {
                "Model": "VAE",
                "Accuracy": accuracy,
                "F1": f1_final,
                "weighted_F1": weighted_f1,
                "OOF_AUC": macro_auc,
                "Macro_AUC": macro_auc,
                "Micro_AUC": micro_auc,
                "MeanFold_AUC": float(np.mean(per_fold_auc)),
                "SDFold_AUC": float(np.std(per_fold_auc, ddof=1)) if len(per_fold_auc) > 1 else 0.0,
                "MetricScope": "Pooled outer-fold out-of-fold predictions",
                "Sensitivity": sensitivity,
                "Specificity": specificity,
            }
        ]
    ).to_csv(f"{prefix}_vae_final_validation_metrics.csv", index=False)

    fpr = {}
    tpr = {}
    roc_auc = {}

    if num_classes == 2:
        fpr[0], tpr[0], _ = roc_curve(y_oof, y_proba_oof[:, 1])
        roc_auc[0] = auc(fpr[0], tpr[0])
    else:
        y_true_bin = np.eye(num_classes)[y_oof]
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba_oof[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_proba_oof.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        fpr_macro, tpr_macro, auc_macro, _ = compute_macro_roc(y_true_bin, y_proba_oof)
        fpr["macro"], tpr["macro"], roc_auc["macro"] = fpr_macro, tpr_macro, auc_macro

    roc_data = {"fpr": fpr, "tpr": tpr, "roc_auc": roc_auc}
    np.save(f"{prefix}_vae_roc_data.npy", roc_data, allow_pickle=True)

    curves = []
    if num_classes == 2:
        curves.append({"fpr": fpr[0], "tpr": tpr[0], "auc": roc_auc[0], "label": "ROC"})
    else:
        for i in range(num_classes):
            curves.append({"fpr": fpr[i], "tpr": tpr[i], "auc": roc_auc[i], "label": str(le.inverse_transform([i])[0])})
        if "micro" in roc_auc:
            curves.append({"fpr": fpr["micro"], "tpr": tpr["micro"], "auc": roc_auc["micro"], "label": "Micro-average", "linestyle": "--"})
        if "macro" in roc_auc:
            curves.append({"fpr": fpr["macro"], "tpr": tpr["macro"], "auc": roc_auc["macro"], "label": "Macro-average", "linestyle": "-."})
    plot_roc_curves(curves, "VAE ROC", f"{prefix}_vae_roc_curve.png")

    results_df = pd.DataFrame(
        {
            "SampleID": sample_ids,
            "Original Label": le.inverse_transform(y_oof),
            "Predicted Label": le.inverse_transform(y_pred_oof),
            "Fold": oof_fold,
        }
    )
    for class_index, class_name in enumerate(le.classes_):
        results_df[f"Probability_{class_name}"] = y_proba_oof[:, class_index]
    results_df.to_csv(f"{prefix}_vae_predictions.csv", index=False)

    joblib.dump(final_model, f"{prefix}_vae_model.pkl")
    joblib.dump(final_preprocessor, f"{prefix}_vae_scaler.pkl")
    joblib.dump({"model": final_model, "preprocessor": final_preprocessor, "feature_roles": roles, "final_hyperparameters": best, "model_feature_order": list(final_preprocessor.get_feature_names_out()), "scope": "final_full_development_model"}, f"{prefix}_vae_final_artifact.pkl")
    joblib.dump((X, y, le, list(final_preprocessor.get_feature_names_out())), f"{prefix}_vae_data.pkl")
    write_reproducibility_manifest(f"{prefix}_vae_manifest.json", input_path=inp, task="classification", model="vae", seed=DEFAULT_RANDOM_STATE, roles=roles, imputation=imputation, final_hyperparameters=best, artifacts=[f"{prefix}_vae_predictions.csv", f"{prefix}_vae_model.pkl"], imputation_fallbacks=imputation_fallback_reasons(missingness_folds), tuning_studies=tuning_reports)


if __name__ == "__main__":
    args = parse_arguments()
    set_seed(args.seed)
    pref = build_output_prefix(args.csv, args.prefix)
    with task_run_context(args.outdir, "classification"):
        vae_script(args.csv, pref, getattr(args, "imputation", "median"), getattr(args, "covariates", None), getattr(args, "categorical_covariates", None))
