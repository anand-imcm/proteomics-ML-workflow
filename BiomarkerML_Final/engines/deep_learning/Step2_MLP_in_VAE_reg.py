import faulthandler; faulthandler.enable()
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import joblib
import matplotlib as mpl
mpl.use("Agg")

import argparse
import random
from pathlib import Path
import warnings
import sys

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim

import optuna

SCRIPT_ROOT = str(Path(__file__).resolve().parents[1])
if SCRIPT_ROOT not in sys.path:
    sys.path.insert(0, SCRIPT_ROOT)
from core.data_contracts import (
    RoleAwarePreprocessor, modeling_frame, resolve_feature_roles,
    write_missingness_qc, write_reproducibility_manifest, imputation_fallback_reasons,
)
from core.adaptive_optuna import (
    reject_legacy_n_trials,
    run_adaptive_study,
    tolerance_for_objective,
)
from core.publication_utils import (
    DEFAULT_RANDOM_STATE,
    build_output_prefix,
    configure_publication_plots,
    SuppressOutput,
    EarlyStopping,
    plot_metric_bars,
    plot_observed_predicted,
    plot_regression_nested_cv,
    plot_residuals,
    task_run_context,
    estimator_importance_with_source,
    export_outer_fold_biomarkers,
    generate_training_size_curve,
    write_training_history,
)

configure_publication_plots()

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass
torch.backends.cudnn.enabled = False  # avoid unpicklable cudnn objects

warnings.filterwarnings('ignore')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Script to run regression with VAE and MLP')
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
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

set_seed(DEFAULT_RANDOM_STATE)


class ConditionalNorm(nn.Module):
    def __init__(self, units, use_batchnorm):
        super(ConditionalNorm, self).__init__()
        if use_batchnorm:
            self.norm = nn.BatchNorm1d(units)
        else:
            self.norm = nn.LayerNorm(units)
    def forward(self, x):
        return self.norm(x)

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, encoder_layers, dropout_rate, use_batchnorm):
        super(Encoder, self).__init__()
        layers = []
        prev_dim = input_dim
        for units in encoder_layers:
            layers.append(nn.Linear(prev_dim, units))
            layers.append(nn.LeakyReLU())
            layers.append(ConditionalNorm(units, use_batchnorm))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = units
        self.hidden_layers = nn.Sequential(*layers)
        self.z_mean = nn.Linear(prev_dim, latent_dim)
        self.z_log_var = nn.Linear(prev_dim, latent_dim)
    def forward(self, x):
        h = self.hidden_layers(x)
        z_mean = self.z_mean(h)
        z_log_var = self.z_log_var(h)
        return z_mean, z_log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, decoder_layers, dropout_rate, use_batchnorm):
        super(Decoder, self).__init__()
        layers = []
        prev_dim = latent_dim
        for units in decoder_layers:
            layers.append(nn.Linear(prev_dim, units))
            layers.append(nn.LeakyReLU())
            layers.append(ConditionalNorm(units, use_batchnorm))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = units
        layers.append(nn.Linear(prev_dim, output_dim))
        self.decoder = nn.Sequential(*layers)
    def forward(self, z):
        return self.decoder(z)

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, encoder_layers, decoder_layers, dropout_rate, use_batchnorm):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim, encoder_layers, dropout_rate, use_batchnorm)
        self.decoder = Decoder(latent_dim, input_dim, decoder_layers, dropout_rate, use_batchnorm)
    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z_log_var = torch.clamp(z_log_var, min=-10, max=10)
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        z = z_mean + eps * std
        x_recon = self.decoder(z)
        return x_recon, z_mean, z_log_var

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, mlp_layers, dropout_rate, use_batchnorm):
        super(MLPRegressor, self).__init__()
        layers = []
        prev_dim = input_dim
        for units in mlp_layers:
            layers.append(nn.Linear(prev_dim, units))
            layers.append(nn.LeakyReLU())
            layers.append(ConditionalNorm(units, use_batchnorm))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = units
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x).squeeze(-1)

class VAE_MLP(BaseEstimator, RegressorMixin):
    def __init__(self,
                 input_dim=30,
                 latent_dim=2,
                 encoder_layers=[64, 32],
                 decoder_layers=[32, 64],
                 mlp_layers=[32, 16],
                 vae_dropout_rate=0.5,
                 mlp_dropout_rate=0.5,
                 use_batchnorm=False,
                 vae_patience=10,
                 vae_min_delta=0.0,
                 mlp_patience=10,
                 mlp_min_delta=0.0,
                 vae_learning_rate=0.001,
                 mlp_learning_rate=0.001,
                 mlp_weight_decay=0.0,
                 beta=1.0,
                 kl_anneal_fraction=0.0,
                 epochs=50,
                 batch_size=32):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.mlp_layers = mlp_layers
        self.vae_dropout_rate = vae_dropout_rate
        self.mlp_dropout_rate = mlp_dropout_rate
        self.use_batchnorm = use_batchnorm
        self.vae_patience = vae_patience
        self.vae_min_delta = vae_min_delta
        self.mlp_patience = mlp_patience
        self.mlp_min_delta = mlp_min_delta
        self.vae_learning_rate = vae_learning_rate
        self.mlp_learning_rate = mlp_learning_rate
        self.mlp_weight_decay = mlp_weight_decay
        self.beta = beta
        self.kl_anneal_fraction = kl_anneal_fraction
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cpu")
        self.vae = None
        self.vae_optimizer = None
        self.vae_scheduler = None
        self.mlp_model = None
        self.mlp_optimizer = None
        self.mlp_scheduler = None

    def create_vae(self):
        self.vae = VAE(self.input_dim, self.latent_dim, self.encoder_layers,
                       self.decoder_layers, self.vae_dropout_rate, self.use_batchnorm).to(self.device)
        self.vae_optimizer = optim.Adam(self.vae.parameters(), lr=self.vae_learning_rate)
        self.vae_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.vae_optimizer, mode='min', factor=0.5, patience=3, verbose=False)

    def create_mlp_model(self):
        self.mlp_model = MLPRegressor(input_dim=self.latent_dim, mlp_layers=self.mlp_layers,
                                      dropout_rate=self.mlp_dropout_rate, use_batchnorm=self.use_batchnorm).to(self.device)
        self.mlp_optimizer = optim.Adam(self.mlp_model.parameters(), lr=self.mlp_learning_rate, weight_decay=self.mlp_weight_decay)
        self.mlp_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.mlp_optimizer, mode='min', factor=0.5, patience=3, verbose=False)

    def _current_beta(self, epoch_idx):
        if self.kl_anneal_fraction <= 0.0:
            return self.beta
        total_anneal_epochs = max(int(self.epochs * self.kl_anneal_fraction), 1)
        scale = min((epoch_idx + 1) / total_anneal_epochs, 1.0)
        return self.beta * scale

    def fit(self, X, y):
        self.create_vae()
        self.training_history_ = []
        X_train_vae, X_val_vae, _, _ = train_test_split(X, y, test_size=0.2, random_state=DEFAULT_RANDOM_STATE)
        X_train_vae_tensor = torch.from_numpy(X_train_vae).float().to(self.device)
        X_val_vae_tensor = torch.from_numpy(X_val_vae).float().to(self.device)
        dataset_train_vae = torch.utils.data.TensorDataset(X_train_vae_tensor)
        vae_train_batch_size = min(self.batch_size, max(1, len(X_train_vae)))
        dataloader_train_vae = torch.utils.data.DataLoader(dataset_train_vae, batch_size=vae_train_batch_size, shuffle=True, drop_last=True)
        dataset_val_vae = torch.utils.data.TensorDataset(X_val_vae_tensor)
        dataloader_val_vae = torch.utils.data.DataLoader(dataset_val_vae, batch_size=self.batch_size, shuffle=False, drop_last=False)

        early_stopping_vae = EarlyStopping(patience=self.vae_patience, min_delta=self.vae_min_delta, verbose=False)

        self.vae.train()
        for epoch in range(self.epochs):
            train_loss_sum = 0.0
            train_n = 0
            for data in dataloader_train_vae:
                x_batch = data[0]
                self.vae_optimizer.zero_grad()
                x_recon, z_mean, z_log_var = self.vae(x_batch)
                recon_loss = nn.functional.mse_loss(x_recon, x_batch, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
                loss = recon_loss + self._current_beta(epoch) * kl_loss
                loss.backward()
                self.vae_optimizer.step()
                train_loss_sum += loss.item()
                train_n += x_batch.size(0)
            train_loss = train_loss_sum / max(train_n, 1)
            self.vae.eval()
            val_loss_sum = 0.0
            val_n = 0
            with torch.no_grad():
                for data in dataloader_val_vae:
                    x_batch = data[0]
                    x_recon, z_mean, z_log_var = self.vae(x_batch)
                    recon_loss = nn.functional.mse_loss(x_recon, x_batch, reduction='sum')
                    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
                    loss = recon_loss + self._current_beta(epoch) * kl_loss
                    val_loss_sum += loss.item()
                    val_n += x_batch.size(0)
            val_loss = val_loss_sum / max(val_n, 1)
            if self.vae_scheduler is not None:
                self.vae_scheduler.step(val_loss)
            stopped = early_stopping_vae(val_loss, epoch + 1, checkpoint_modules=(self.vae,))
            self.training_history_.append({
                "Epoch": epoch + 1,
                "TrainingLoss": train_loss,
                "ValidationLoss": val_loss,
                "LearningRate": self.vae_optimizer.param_groups[0]["lr"],
                "EarlyStoppingTriggered": stopped,
                "BestEpoch": early_stopping_vae.best_epoch,
            })
            if stopped:
                break
            self.vae.train()

        early_stopping_vae.restore_checkpoint(self.vae)
        self.vae.eval()
        with torch.no_grad():
            X_encoded = []
            X_tensor_full = torch.from_numpy(X).float().to(self.device)
            dataset_full = torch.utils.data.TensorDataset(X_tensor_full)
            dataloader_full = torch.utils.data.DataLoader(dataset_full, batch_size=self.batch_size, shuffle=False, drop_last=False)
            for data in dataloader_full:
                x_batch = data[0]
                _, z_mean, _ = self.vae(x_batch)
                X_encoded.append(z_mean.cpu())
            X_encoded = torch.cat(X_encoded, dim=0).numpy()

        self.create_mlp_model()
        X_train_mlp, X_val_mlp, y_train_mlp, y_val_mlp = train_test_split(X_encoded, y, test_size=0.2, random_state=DEFAULT_RANDOM_STATE)
        X_train_mlp_tensor = torch.from_numpy(X_train_mlp).float().to(self.device)
        y_train_mlp_tensor = torch.from_numpy(y_train_mlp).float().to(self.device)
        X_val_mlp_tensor = torch.from_numpy(X_val_mlp).float().to(self.device)
        y_val_mlp_tensor = torch.from_numpy(y_val_mlp).float().to(self.device)
        dataset_train_mlp = torch.utils.data.TensorDataset(X_train_mlp_tensor, y_train_mlp_tensor)
        mlp_train_batch_size = min(self.batch_size, max(1, len(X_train_mlp)))
        dataloader_train_mlp = torch.utils.data.DataLoader(dataset_train_mlp, batch_size=mlp_train_batch_size, shuffle=True, drop_last=True)
        dataset_val_mlp = torch.utils.data.TensorDataset(X_val_mlp_tensor, y_val_mlp_tensor)
        dataloader_val_mlp = torch.utils.data.DataLoader(dataset_val_mlp, batch_size=self.batch_size, shuffle=False, drop_last=False)

        early_stopping_mlp = EarlyStopping(patience=self.mlp_patience, min_delta=self.mlp_min_delta, verbose=False)

        self.mlp_model.train()
        criterion = nn.MSELoss()
        offset = len(self.training_history_)
        for epoch in range(self.epochs):
            train_loss_sum = 0.0
            train_n = 0
            for data in dataloader_train_mlp:
                x_batch, y_batch = data
                self.mlp_optimizer.zero_grad()
                preds = self.mlp_model(x_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                self.mlp_optimizer.step()
                train_loss_sum += loss.item() * x_batch.size(0)
                train_n += x_batch.size(0)
            train_loss = train_loss_sum / max(train_n, 1)
            self.mlp_model.eval()
            val_loss_sum = 0.0
            val_n = 0
            with torch.no_grad():
                for data in dataloader_val_mlp:
                    x_batch, y_batch = data
                    preds = self.mlp_model(x_batch)
                    loss = criterion(preds, y_batch)
                    val_loss_sum += loss.item() * x_batch.size(0)
                    val_n += x_batch.size(0)
            val_loss = val_loss_sum / max(val_n, 1)
            if self.mlp_scheduler is not None:
                self.mlp_scheduler.step(val_loss)
            stopped = early_stopping_mlp(val_loss, epoch + 1, checkpoint_modules=(self.mlp_model,))
            self.training_history_.append({
                "Epoch": offset + epoch + 1,
                "TrainingLoss": train_loss,
                "ValidationLoss": val_loss,
                "LearningRate": self.mlp_optimizer.param_groups[0]["lr"],
                "EarlyStoppingTriggered": stopped,
                "BestEpoch": offset + early_stopping_mlp.best_epoch,
            })
            if stopped:
                break
            self.mlp_model.train()
        early_stopping_mlp.restore_checkpoint(self.mlp_model)
        return self

    def predict(self, X, inference_batch_size: int | None = None):
        batch_size = int(inference_batch_size or self.batch_size)
        X_tensor = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.device)
        predictions = []
        # Preserve the two base-seed draws made by the legacy inference DataLoaders.
        torch.empty((), dtype=torch.int64).random_()
        self.vae.eval()
        self.mlp_model.eval()
        with torch.inference_mode():
            for start in range(0, X_tensor.shape[0], batch_size):
                _, z_mean, _ = self.vae(X_tensor[start:start + batch_size])
                predictions.append(self.mlp_model(z_mean).cpu())
        torch.empty((), dtype=torch.int64).random_()
        if not predictions:
            return np.array([])
        return torch.cat(predictions, dim=0).numpy()

def _make_regression_preprocessor(roles, imputation):
    return RoleAwarePreprocessor(
        biomarker_columns=roles.biomarkers,
        numeric_covariates=roles.numeric_covariates,
        categorical_covariates=roles.categorical_covariates,
        imputation=imputation,
        selector=None,
    )


def _suggest_mlpvae_regression_params(trial):
    params = {
        'latent_dim': trial.suggest_int('latent_dim', 2, 256, log=True),
        'use_batchnorm': trial.suggest_categorical('use_batchnorm', [True, False]),
        'vae_dropout_rate': trial.suggest_float('vae_dropout_rate', 0.0, 0.5, step=0.1),
        'mlp_dropout_rate': trial.suggest_float('mlp_dropout_rate', 0.0, 0.5, step=0.1),
        'vae_patience': trial.suggest_int('vae_patience', 5, 20),
        'vae_min_delta': trial.suggest_float('vae_min_delta', 0.0, 0.1, step=0.01),
        'mlp_patience': trial.suggest_int('mlp_patience', 5, 20),
        'mlp_min_delta': trial.suggest_float('mlp_min_delta', 0.0, 0.1, step=0.01),
        'num_encoder_layers': trial.suggest_int('num_encoder_layers', 1, 5),
        'num_decoder_layers': trial.suggest_int('num_decoder_layers', 1, 5),
        'num_mlp_layers': trial.suggest_int('num_mlp_layers', 1, 5),
        'vae_learning_rate': trial.suggest_float('vae_learning_rate', 1e-5, 0.05, log=True),
        'mlp_learning_rate': trial.suggest_float('mlp_learning_rate', 1e-5, 0.05, log=True),
        'mlp_weight_decay': trial.suggest_float('mlp_weight_decay', 1e-8, 1e-2, log=True),
        'beta': trial.suggest_float('beta', 0.1, 1.0, log=True),
        'kl_anneal_fraction': trial.suggest_float('kl_anneal_fraction', 0.0, 0.5),
        'epochs': trial.suggest_int('epochs', 10, 200),
        'batch_size': trial.suggest_int('batch_size', 2, 256, log=True),
    }
    for i in range(params['num_encoder_layers']):
        params[f'encoder_units_l{i}'] = trial.suggest_int(f'encoder_units_l{i}', 16, 512, log=True)
    for i in range(params['num_decoder_layers']):
        params[f'decoder_units_l{i}'] = trial.suggest_int(f'decoder_units_l{i}', 16, 512, log=True)
    for i in range(params['num_mlp_layers']):
        params[f'mlp_units_l{i}'] = trial.suggest_int(f'mlp_units_l{i}', 16, 512, log=True)
    return params


def _build_mlpvae_regression(params, input_dim):
    return VAE_MLP(
        input_dim=input_dim,
        latent_dim=params['latent_dim'],
        encoder_layers=[params[f'encoder_units_l{i}'] for i in range(params['num_encoder_layers'])],
        decoder_layers=[params[f'decoder_units_l{i}'] for i in range(params['num_decoder_layers'])],
        mlp_layers=[params[f'mlp_units_l{i}'] for i in range(params['num_mlp_layers'])],
        vae_dropout_rate=params['vae_dropout_rate'],
        mlp_dropout_rate=params['mlp_dropout_rate'],
        use_batchnorm=params['use_batchnorm'],
        vae_patience=params['vae_patience'],
        vae_min_delta=params['vae_min_delta'],
        mlp_patience=params['mlp_patience'],
        mlp_min_delta=params['mlp_min_delta'],
        vae_learning_rate=params['vae_learning_rate'],
        mlp_learning_rate=params['mlp_learning_rate'],
        mlp_weight_decay=params['mlp_weight_decay'],
        beta=params['beta'],
        kl_anneal_fraction=params['kl_anneal_fraction'],
        epochs=params['epochs'],
        batch_size=params['batch_size'],
    )


def select_mlpvae_regression_params(
    X_train,
    y_train,
    roles,
    imputation,
    n_trials=None,
    *,
    study_scope="configured_retraining",
    outer_fold=None,
    study_id=None,
    provenance_path=None,
    return_report=False,
):
    reject_legacy_n_trials(n_trials)
    def objective(trial):
        params = _suggest_mlpvae_regression_params(trial)
        inner_kf = KFold(n_splits=min(5, len(X_train)), shuffle=True, random_state=DEFAULT_RANDOM_STATE)
        scores = []
        for inner_train_idx, inner_val_idx in inner_kf.split(X_train):
            X_inner_train = X_train.iloc[inner_train_idx]
            X_inner_val = X_train.iloc[inner_val_idx]
            y_inner_train = y_train[inner_train_idx]
            y_inner_val = y_train[inner_val_idx]
            preprocessor = _make_regression_preprocessor(roles, imputation)
            X_tr = preprocessor.fit_transform(X_inner_train, y_inner_train)
            X_va = preprocessor.transform(X_inner_val)
            model = _build_mlpvae_regression(params, X_tr.shape[1])
            model.fit(np.asarray(X_tr), y_inner_train)
            scores.append(np.sqrt(mean_squared_error(y_inner_val, model.predict(np.asarray(X_va)))))
        return float(np.mean(scores))

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=DEFAULT_RANDOM_STATE))
    with SuppressOutput():
        report = run_adaptive_study(
            study,
            objective,
            task="regression",
            model="MLPVAE_reg",
            study_scope=study_scope,
            objective_name="mlpvae_regression_rmse",
            direction="minimize",
            tolerance=tolerance_for_objective(
                "mlpvae_regression_rmse",
                scale_floor=max(float(np.nanstd(y_train)), np.finfo(float).eps),
            ),
            seed=DEFAULT_RANDOM_STATE,
            study_id=study_id or f"regression-mlpvae-{study_scope}",
            outer_fold=outer_fold,
            provenance_path=provenance_path,
        )
    params = dict(study.best_params)
    return (params, report) if return_report else params


# top-level worker to avoid pickling closures
def evaluate_fold_final(job):
    X, y, train_idx, test_idx, fold_num, roles, imputation, prefix = job
    set_seed(DEFAULT_RANDOM_STATE)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    best_params, tuning_report = select_mlpvae_regression_params(
        X_train,
        y_train,
        roles,
        imputation,
        study_scope="outer_fold",
        outer_fold=fold_num,
        study_id=f"regression-mlpvae-outer-fold-{fold_num}",
        provenance_path=f"{prefix}_MLP_in_VAE_reg_fold{fold_num}_optuna_study.json",
        return_report=True,
    )
    preprocessor = _make_regression_preprocessor(roles, imputation)
    X_train_scaled = preprocessor.fit_transform(X_train, y_train)
    X_test_scaled = preprocessor.transform(X_test)
    model = _build_mlpvae_regression(best_params, X_train_scaled.shape[1])
    model.fit(np.asarray(X_train_scaled), y_train)
    y_pred = model.predict(np.asarray(X_test_scaled))
    fold_importance, fold_importance_type = estimator_importance_with_source(model, X_train_scaled.shape[1])
    biomarker_count = len(preprocessor.selected_biomarker_names_)

    mse = mean_squared_error(y_test, y_pred)
    metrics = {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
    }
    predictions = {
        'X_test': np.asarray(X_test_scaled), 'y_test': y_test, 'y_pred': y_pred,
        'test_idx': test_idx, 'training_history': getattr(model, 'training_history_', []),
        'diagnostic_train_X': np.asarray(X_train_scaled), 'diagnostic_train_y': y_train,
        'diagnostic_val_X': np.asarray(X_test_scaled), 'diagnostic_val_y': y_test,
        'diagnostic_params': best_params, 'fold_num': int(fold_num),
        'feature_importance': np.asarray(fold_importance)[:biomarker_count],
        'importance_type': fold_importance_type,
        'biomarker_names': list(preprocessor.selected_biomarker_names_),
        'missingness_qc': preprocessor.biomarker_diagnostics(fold_num),
        'tuning_report': tuning_report,
    }
    print(f"Fold {fold_num}: MSE={metrics['mse']:.4f}, RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, R2={metrics['r2']:.4f}", flush=True)
    return metrics, predictions

# main driver
def vae_regression(inp, prefix, imputation="median", covariates=None, categorical_covariates=None):
    data = pd.read_csv(inp)
    roles = resolve_feature_roles(data, target="Label", identifier="SampleID", covariates=covariates, categorical_covariates=categorical_covariates)
    sample_ids = data['SampleID'].values if 'SampleID' in data.columns else np.asarray([f"Sample_{i+1}" for i in range(len(data))])
    X = modeling_frame(data, roles)
    y = pd.to_numeric(data['Label'], errors='raise').to_numpy(dtype=np.float32)
    feature_names = list(roles.biomarkers)

    mse_scores, rmse_scores, mae_scores, r2_scores = [], [], [], []
    all_X, all_y_true, all_y_pred = [], [], []
    training_history_rows = []

    outer_kf = KFold(n_splits=5, shuffle=True, random_state=DEFAULT_RANDOM_STATE)

    results = []
    for fold_num, (train_idx, test_idx) in enumerate(outer_kf.split(X), 1):
        metrics, predictions = evaluate_fold_final((X, y, train_idx, test_idx, fold_num, roles, imputation, prefix))
        results.append((metrics, predictions))

    for metrics, predictions in results:
        export_outer_fold_biomarkers(
            prefix,
            "MLP_in_VAE_reg",
            predictions['fold_num'],
            None,
            predictions['biomarker_names'],
            selection_method="None",
            importance_values=predictions['feature_importance'],
            importance_type=predictions['importance_type'],
        )
        mse_scores.append(metrics['mse'])
        rmse_scores.append(metrics['rmse'])
        mae_scores.append(metrics['mae'])
        r2_scores.append(metrics['r2'])
        all_X.append(predictions['X_test'])
        all_y_true.append(predictions['y_test'])
        all_y_pred.append(predictions['y_pred'])
        for row in predictions.get('training_history', []):
            training_history_rows.append(dict(row))

    if all_X:
        X_all = np.vstack(all_X)
        y_all_true = np.hstack(all_y_true)
        y_all_pred = np.hstack(all_y_pred)
        all_test_idx = np.hstack([pred['test_idx'] for _, pred in results])
    else:
        print("No predictions were made. The predictions DataFrame will not be created.", flush=True)
        return

    if X_all.size > 0:
        results_df = pd.DataFrame({
            'SampleID': sample_ids[all_test_idx],
            'Original Label': data['Label'].values[all_test_idx],
            'Predicted Value': y_all_pred,
            'Fold': np.hstack([[pred['fold_num']] * len(pred['test_idx']) for _, pred in results])
        })
        results_df.to_csv(f"{prefix}_vaemlp_reg_predictions.csv", index=False)
    else:
        print("No predictions were made. The predictions DataFrame will not be created.", flush=True)

    mean_mse = float(np.mean(mse_scores)) if mse_scores else 0.0
    mean_rmse = float(np.mean(rmse_scores)) if rmse_scores else 0.0
    mean_mae = float(np.mean(mae_scores)) if mae_scores else 0.0
    mean_r2 = float(np.mean(r2_scores)) if r2_scores else 0.0

    metrics_df = pd.DataFrame({
        'Metric': ['MSE', 'RMSE', 'MAE', 'R2'],
        'Score': [mean_mse, mean_rmse, mean_mae, mean_r2]
    })
    print(metrics_df, flush=True)

    fold_numbers = list(range(1, len(mse_scores) + 1))
    fold_metrics = pd.DataFrame({"Fold": fold_numbers, "MSE": mse_scores, "RMSE": rmse_scores, "MAE": mae_scores, "R2": r2_scores})
    fold_metrics.to_csv(f"{prefix}_vaemlp_reg_metrics_over_folds.csv", index=False)
    fold_metrics.assign(Model="MLP_in_VAE_reg").to_csv(f"{prefix}_MLP_in_VAE_reg_outer_fold_metrics.csv", index=False)
    plot_regression_nested_cv(fold_metrics, "VAE MLP Regression Nested CV", f"{prefix}_vaemlp_reg_metrics_over_folds.png")

    plot_metric_bars(["RMSE", "MAE", "R2"], [mean_rmse, mean_mae, mean_r2], "VAE MLP Regression Evaluation Metrics", f"{prefix}_vaemlp_reg_average_metrics.png")

    plot_observed_predicted(y_all_true, y_all_pred, "VAE MLP Regression: Observed vs Predicted", f"{prefix}_vaemlp_reg_predictions.png")

    plot_residuals(y_all_true, y_all_pred, "VAE MLP Regression: Residuals", f"{prefix}_vaemlp_reg_residuals.png")
    write_training_history(training_history_rows, prefix, "vaemlp_reg")
    write_missingness_qc(prefix + "_vaemlp_reg", X, roles, [pred['missingness_qc'] for _, pred in results])

    # Select final hyperparameters independently on the full development set.
    final_params, final_tuning_report = select_mlpvae_regression_params(
        X,
        y,
        roles,
        imputation,
        study_scope="final_full_development",
        study_id="regression-mlpvae-final",
        provenance_path=f"{prefix}_MLP_in_VAE_reg_final_optuna_study.json",
        return_report=True,
    )
    tuning_reports = [pred['tuning_report'] for _, pred in results] + [final_tuning_report]
    final_preprocessor = _make_regression_preprocessor(roles, imputation)
    X_final = final_preprocessor.fit_transform(X, y)
    final_model = _build_mlpvae_regression(final_params, X_final.shape[1])
    final_model.fit(np.asarray(X_final), y)
    artifact = {
        "model": final_model,
        "preprocessor": final_preprocessor,
        "feature_order": list(X.columns),
        "model_feature_order": list(final_preprocessor.get_feature_names_out()),
        "X_raw": X,
        "y": y,
        "selected_biomarker_features": list(final_preprocessor.selected_biomarker_names_),
        "feature_roles": roles,
        "final_hyperparameters": final_params,
        "architecture": {key: value for key, value in final_params.items() if "layer" in key or "units" in key or key == "latent_dim"},
        "random_seed": DEFAULT_RANDOM_STATE,
        "scope": "final_full_development_model",
    }
    joblib.dump(artifact, f"{prefix}_MLP_in_VAE_reg_final_artifact.pkl")
    joblib.dump(final_model, f"{prefix}_MLP_in_VAE_reg_best_model.pkl")
    with open(f"{prefix}_MLP_in_VAE_reg_best_params.json", "w", encoding="utf-8") as handle:
        import json
        json.dump({"final": final_params, "outer_folds": [pred['diagnostic_params'] for _, pred in results], "provenance": "explicit full-development CV after OOF validation", "adaptive_tuning": tuning_reports}, handle, indent=2)
    pd.DataFrame([{
        "Model": "MLP_in_VAE_reg", "MSE": float(mean_squared_error(y_all_true, y_all_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_all_true, y_all_pred))),
        "MAE": float(mean_absolute_error(y_all_true, y_all_pred)), "R2": float(r2_score(y_all_true, y_all_pred)),
        "MetricScope": "Pooled outer-fold out-of-fold predictions",
    }]).to_csv(f"{prefix}_MLP_in_VAE_reg_final_validation_metrics.csv", index=False)
    write_reproducibility_manifest(f"{prefix}_MLP_in_VAE_reg_manifest.json", input_path=inp, task="regression", model="MLP_in_VAE_reg", seed=DEFAULT_RANDOM_STATE, roles=roles, imputation=imputation, final_hyperparameters=final_params, artifacts=[f"{prefix}_vaemlp_reg_predictions.csv", f"{prefix}_MLP_in_VAE_reg_final_artifact.pkl"], imputation_fallbacks=imputation_fallback_reasons([pred['missingness_qc'] for _, pred in results]), tuning_studies=tuning_reports)

    diagnostic = next((pred for _, pred in results if pred.get('diagnostic_params')), None)
    if diagnostic is not None:
        params = diagnostic['diagnostic_params']

        def make_diagnostic_model():
            set_seed(DEFAULT_RANDOM_STATE)
            return VAE_MLP(
                input_dim=diagnostic['diagnostic_train_X'].shape[1],
                latent_dim=params['latent_dim'],
                encoder_layers=[params[f'encoder_units_l{i}'] for i in range(params['num_encoder_layers'])],
                decoder_layers=[params[f'decoder_units_l{i}'] for i in range(params['num_decoder_layers'])],
                mlp_layers=[params[f'mlp_units_l{i}'] for i in range(params['num_mlp_layers'])],
                vae_dropout_rate=params['vae_dropout_rate'],
                mlp_dropout_rate=params['mlp_dropout_rate'],
                use_batchnorm=params['use_batchnorm'],
                vae_patience=params['vae_patience'],
                vae_min_delta=params['vae_min_delta'],
                mlp_patience=params['mlp_patience'],
                mlp_min_delta=params['mlp_min_delta'],
                vae_learning_rate=params['vae_learning_rate'],
                mlp_learning_rate=params['mlp_learning_rate'],
                mlp_weight_decay=params['mlp_weight_decay'],
                beta=params['beta'],
                kl_anneal_fraction=params['kl_anneal_fraction'],
                epochs=params['epochs'],
                batch_size=params['batch_size']
            )

        generate_training_size_curve(
            diagnostic['diagnostic_train_X'],
            diagnostic['diagnostic_train_y'],
            make_diagnostic_model,
            prefix,
            "vaemlp_reg",
            task="regression",
            X_validation=diagnostic['diagnostic_val_X'],
            y_validation=diagnostic['diagnostic_val_y'],
        )

if __name__ == "__main__":
    args = parse_arguments()
    set_seed(args.seed)
    prefix = build_output_prefix(args.csv, args.prefix)
    with task_run_context(args.outdir, "regression"):
        vae_regression(args.csv, prefix, args.imputation, args.covariates, args.categorical_covariates)
