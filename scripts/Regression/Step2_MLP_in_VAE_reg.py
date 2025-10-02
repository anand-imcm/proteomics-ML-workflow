import faulthandler; faulthandler.enable()
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import matplotlib as mpl
mpl.use("Agg")

import argparse
import random
from pathlib import Path
import warnings
import sys
import contextlib

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler

import shap
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim

import optuna
from joblib import Parallel, delayed

# global torch/thread settings
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass
torch.backends.cudnn.enabled = False  # avoid unpicklable cudnn objects

warnings.filterwarnings('ignore')

# matplotlib sizing
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.labelsize'] = 15
mpl.rcParams['xtick.labelsize'] = 13
mpl.rcParams['ytick.labelsize'] = 13
mpl.rcParams['legend.fontsize'] = 13

# --------------------------------------------------------------------------------------
# utils / infra
# --------------------------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description='Script to run regression with VAE and MLP')
    parser.add_argument('-i', '--csv', type=str, help='Input CSV file', required=True)
    parser.add_argument('-p', '--prefix', type=str, help='Output prefix')
    return parser.parse_args()

def set_seed(seed=42):
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

set_seed(42)

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

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
    def __call__(self, current_loss):
        if self.best_loss is None:
            self.best_loss = current_loss
            return False
        elif current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
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
        X_train_vae, X_val_vae, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_vae_tensor = torch.from_numpy(X_train_vae).float().to(self.device)
        X_val_vae_tensor = torch.from_numpy(X_val_vae).float().to(self.device)
        dataset_train_vae = torch.utils.data.TensorDataset(X_train_vae_tensor)
        dataloader_train_vae = torch.utils.data.DataLoader(dataset_train_vae, batch_size=self.batch_size, shuffle=True, drop_last=True)
        dataset_val_vae = torch.utils.data.TensorDataset(X_val_vae_tensor)
        dataloader_val_vae = torch.utils.data.DataLoader(dataset_val_vae, batch_size=self.batch_size, shuffle=False, drop_last=False)

        early_stopping_vae = EarlyStopping(patience=self.vae_patience, min_delta=self.vae_min_delta, verbose=False)

        self.vae.train()
        for epoch in range(self.epochs):
            for data in dataloader_train_vae:
                x_batch = data[0]
                self.vae_optimizer.zero_grad()
                x_recon, z_mean, z_log_var = self.vae(x_batch)
                recon_loss = nn.functional.mse_loss(x_recon, x_batch, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
                loss = recon_loss + self._current_beta(epoch) * kl_loss
                loss.backward()
                self.vae_optimizer.step()
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
            if early_stopping_vae(val_loss):
                break
            self.vae.train()

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
        X_train_mlp, X_val_mlp, y_train_mlp, y_val_mlp = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
        X_train_mlp_tensor = torch.from_numpy(X_train_mlp).float().to(self.device)
        y_train_mlp_tensor = torch.from_numpy(y_train_mlp).float().to(self.device)
        X_val_mlp_tensor = torch.from_numpy(X_val_mlp).float().to(self.device)
        y_val_mlp_tensor = torch.from_numpy(y_val_mlp).float().to(self.device)
        dataset_train_mlp = torch.utils.data.TensorDataset(X_train_mlp_tensor, y_train_mlp_tensor)
        dataloader_train_mlp = torch.utils.data.DataLoader(dataset_train_mlp, batch_size=self.batch_size, shuffle=True, drop_last=True)
        dataset_val_mlp = torch.utils.data.TensorDataset(X_val_mlp_tensor, y_val_mlp_tensor)
        dataloader_val_mlp = torch.utils.data.DataLoader(dataset_val_mlp, batch_size=self.batch_size, shuffle=False, drop_last=False)

        early_stopping_mlp = EarlyStopping(patience=self.mlp_patience, min_delta=self.mlp_min_delta, verbose=False)

        self.mlp_model.train()
        criterion = nn.MSELoss()
        for epoch in range(self.epochs):
            for data in dataloader_train_mlp:
                x_batch, y_batch = data
                self.mlp_optimizer.zero_grad()
                preds = self.mlp_model(x_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                self.mlp_optimizer.step()
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
            if early_stopping_mlp(val_loss):
                break
            self.mlp_model.train()
        return self

    def predict(self, X):
        self.vae.eval()
        self.mlp_model.eval()
        X_tensor = torch.from_numpy(X).float().to(self.device)
        dataset = torch.utils.data.TensorDataset(X_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        with torch.no_grad():
            X_encoded = []
            for data in dataloader:
                x_batch = data[0]
                _, z_mean, _ = self.vae(x_batch)
                X_encoded.append(z_mean.cpu())
            X_encoded = torch.cat(X_encoded, dim=0).numpy()
            y_pred = []
            dataset_mlp = torch.utils.data.TensorDataset(torch.from_numpy(X_encoded).float().to(self.device))
            dataloader_mlp = torch.utils.data.DataLoader(dataset_mlp, batch_size=self.batch_size, shuffle=False, drop_last=False)
            for data in dataloader_mlp:
                x_batch = data[0].to(self.device)
                preds = self.mlp_model(x_batch)
                y_pred.append(preds.cpu())
            y_pred = torch.cat(y_pred, dim=0).numpy()
        return y_pred

def calculate_shap_feature_importance(shap_values, feature_names):
    mean_shap = np.mean(np.abs(shap_values), axis=0)
    return list(zip(feature_names, mean_shap))

# --------------------------------------------------------------------------------------
# top-level worker to avoid pickling closures
# --------------------------------------------------------------------------------------
def evaluate_fold_final(job):
    X, y, train_idx, test_idx, fold_num = job
    set_seed(42 + int(fold_num))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    def objective(trial):
        latent_dim = trial.suggest_int('latent_dim', 2, 256, log=True)
        use_batchnorm = trial.suggest_categorical('use_batchnorm', [True, False])
        vae_dropout_rate = trial.suggest_float('vae_dropout_rate', 0.0, 0.5, step=0.1)
        mlp_dropout_rate = trial.suggest_float('mlp_dropout_rate', 0.0, 0.5, step=0.1)

        vae_patience = trial.suggest_int('vae_patience', 5, 20)
        vae_min_delta = trial.suggest_float('vae_min_delta', 0.0, 0.1, step=0.01)
        mlp_patience = trial.suggest_int('mlp_patience', 5, 20)
        mlp_min_delta = trial.suggest_float('mlp_min_delta', 0.0, 0.1, step=0.01)

        num_encoder_layers = trial.suggest_int('num_encoder_layers', 1, 5)
        encoder_layers = [trial.suggest_int(f'encoder_units_l{i}', 16, 512, log=True)
                          for i in range(num_encoder_layers)]
        num_decoder_layers = trial.suggest_int('num_decoder_layers', 1, 5)
        decoder_layers = [trial.suggest_int(f'decoder_units_l{i}', 16, 512, log=True)
                          for i in range(num_decoder_layers)]
        num_mlp_layers = trial.suggest_int('num_mlp_layers', 1, 5)
        mlp_layers = [trial.suggest_int(f'mlp_units_l{i}', 16, 512, log=True)
                      for i in range(num_mlp_layers)]

        vae_learning_rate = trial.suggest_float('vae_learning_rate', 1e-5, 0.05, log=True)
        mlp_learning_rate = trial.suggest_float('mlp_learning_rate', 1e-5, 0.05, log=True)
        mlp_weight_decay = trial.suggest_float('mlp_weight_decay', 1e-8, 1e-2, log=True)
        beta = trial.suggest_float('beta', 0.1, 1.0, log=True)
        kl_anneal_fraction = trial.suggest_float('kl_anneal_fraction', 0.0, 0.5)
        epochs = trial.suggest_int('epochs', 10, 200)
        batch_size = trial.suggest_int('batch_size', 2, 256, log=True)

        model = VAE_MLP(
            input_dim=X_train.shape[1],
            latent_dim=latent_dim,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            mlp_layers=mlp_layers,
            vae_dropout_rate=vae_dropout_rate,
            mlp_dropout_rate=mlp_dropout_rate,
            use_batchnorm=use_batchnorm,
            vae_patience=vae_patience,
            vae_min_delta=vae_min_delta,
            mlp_patience=mlp_patience,
            mlp_min_delta=mlp_min_delta,
            vae_learning_rate=vae_learning_rate,
            mlp_learning_rate=mlp_learning_rate,
            mlp_weight_decay=mlp_weight_decay,
            beta=beta,
            kl_anneal_fraction=kl_anneal_fraction,
            epochs=epochs,
            batch_size=batch_size
        )

        inner_kf = KFold(n_splits=5, shuffle=True, random_state=fold_num)
        rmse_scores_inner = []

        for inner_train_idx, inner_val_idx in inner_kf.split(X_train):
            X_inner_train, X_inner_val = X_train[inner_train_idx], X_train[inner_val_idx]
            y_inner_train, y_inner_val = y_train[inner_train_idx], y_train[inner_val_idx]

            scaler_inner = StandardScaler()
            X_inner_train_scaled = scaler_inner.fit_transform(X_inner_train)
            X_inner_val_scaled = scaler_inner.transform(X_inner_val)

            fold_model = VAE_MLP(
                input_dim=model.input_dim,
                latent_dim=model.latent_dim,
                encoder_layers=model.encoder_layers,
                decoder_layers=model.decoder_layers,
                mlp_layers=model.mlp_layers,
                vae_dropout_rate=model.vae_dropout_rate,
                mlp_dropout_rate=model.mlp_dropout_rate,
                use_batchnorm=model.use_batchnorm,
                vae_patience=model.vae_patience,
                vae_min_delta=model.vae_min_delta,
                mlp_patience=model.mlp_patience,
                mlp_min_delta=model.mlp_min_delta,
                vae_learning_rate=model.vae_learning_rate,
                mlp_learning_rate=model.mlp_learning_rate,
                mlp_weight_decay=model.mlp_weight_decay,
                beta=model.beta,
                kl_anneal_fraction=model.kl_anneal_fraction,
                epochs=model.epochs,
                batch_size=model.batch_size
            )
            fold_model.fit(X_inner_train_scaled, y_inner_train)
            y_inner_pred = fold_model.predict(X_inner_val_scaled)
            rmse = np.sqrt(mean_squared_error(y_inner_val, y_inner_pred))
            rmse_scores_inner.append(rmse)

        return np.mean(rmse_scores_inner)

    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    with SuppressOutput():
        study.optimize(objective, n_trials=20, n_jobs=-1)

    best_params = study.best_params

    latent_dim = best_params['latent_dim']
    use_batchnorm = best_params['use_batchnorm']
    vae_dropout_rate = best_params['vae_dropout_rate']
    mlp_dropout_rate = best_params['mlp_dropout_rate']
    vae_patience = best_params['vae_patience']
    vae_min_delta = best_params['vae_min_delta']
    mlp_patience = best_params['mlp_patience']
    mlp_min_delta = best_params['mlp_min_delta']
    encoder_layers = [best_params[f'encoder_units_l{i}'] for i in range(best_params['num_encoder_layers'])]
    decoder_layers = [best_params[f'decoder_units_l{i}'] for i in range(best_params['num_decoder_layers'])]
    mlp_layers = [best_params[f'mlp_units_l{i}'] for i in range(best_params['num_mlp_layers'])]
    vae_learning_rate = best_params['vae_learning_rate']
    mlp_learning_rate = best_params['mlp_learning_rate']
    mlp_weight_decay = best_params['mlp_weight_decay']
    beta = best_params['beta']
    kl_anneal_fraction = best_params['kl_anneal_fraction']
    epochs = best_params['epochs']
    batch_size = best_params['batch_size']

    model = VAE_MLP(
        input_dim=X_train.shape[1],
        latent_dim=latent_dim,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        mlp_layers=mlp_layers,
        vae_dropout_rate=vae_dropout_rate,
        mlp_dropout_rate=mlp_dropout_rate,
        use_batchnorm=use_batchnorm,
        vae_patience=vae_patience,
        vae_min_delta=vae_min_delta,
        mlp_patience=mlp_patience,
        mlp_min_delta=mlp_min_delta,
        vae_learning_rate=vae_learning_rate,
        mlp_learning_rate=mlp_learning_rate,
        mlp_weight_decay=mlp_weight_decay,
        beta=beta,
        kl_anneal_fraction=kl_anneal_fraction,
        epochs=epochs,
        batch_size=batch_size
    )

    scaler_outer = StandardScaler()
    X_train_scaled = scaler_outer.fit_transform(X_train)
    X_test_scaled = scaler_outer.transform(X_test)

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}
    predictions = {
        'X_test': X_test_scaled,
        'y_test': y_test,
        'y_pred': y_pred,
        'test_idx': test_idx
    }
    print(f"Fold {fold_num}: MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}", flush=True)
    return metrics, predictions

# --------------------------------------------------------------------------------------
# main driver
# --------------------------------------------------------------------------------------
def vae_regression(inp, prefix):
    data = pd.read_csv(inp)
    sample_ids = data['SampleID'].values
    X = data.drop(columns=['SampleID', 'Label']).values
    y = data['Label'].values.astype(np.float32)
    feature_names = data.drop(columns=['SampleID', 'Label']).columns.tolist()

    mse_scores, rmse_scores, mae_scores, r2_scores = [], [], [], []
    all_X, all_y_true, all_y_pred = [], [], []

    outer_kf = KFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    for fold_num, (train_idx, test_idx) in enumerate(outer_kf.split(X), 1):
        try:
            metrics, predictions = evaluate_fold_final((X, y, train_idx, test_idx, fold_num))
            results.append((metrics, predictions))
        except Exception as e:
            print(f"[WARN] Fold {fold_num} failed with error: {e}", flush=True)
            continue

    for metrics, predictions in results:
        mse_scores.append(metrics['mse'])
        rmse_scores.append(metrics['rmse'])
        mae_scores.append(metrics['mae'])
        r2_scores.append(metrics['r2'])
        all_X.append(predictions['X_test'])
        all_y_true.append(predictions['y_test'])
        all_y_pred.append(predictions['y_pred'])

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
            'Predicted Value': y_all_pred
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

    metric_names = ['MSE', 'RMSE', 'MAE', 'R2']
    fold_numbers = list(range(1, len(mse_scores) + 1))

    plt.figure(figsize=(12, 8))
    plt.plot(fold_numbers, mse_scores, marker='o', label='MSE')
    plt.plot(fold_numbers, rmse_scores, marker='o', label='RMSE')
    plt.plot(fold_numbers, mae_scores, marker='o', label='MAE')
    plt.plot(fold_numbers, r2_scores, marker='o', label='R2')
    plt.xlabel('Fold Number')
    plt.ylabel('Score')
    plt.title('Evaluation Metrics over Folds')
    plt.legend()
    plt.xticks(fold_numbers)
    plt.tight_layout()
    plt.savefig(f"{prefix}_vaemlp_reg_metrics_over_folds.png", dpi=300)
    plt.close()

    mean_scores = [mean_mse, mean_rmse, mean_mae, mean_r2]
    plt.figure(figsize=(12, 8))
    bars = plt.bar(metric_names, mean_scores)

    max_val = max(mean_scores) if mean_scores else 0.0
    min_val = min(mean_scores) if mean_scores else 0.0
    span = max_val - min(0.0, min_val)
    offset = 0.02 * (span if span > 0 else 1.0)

    for bar, score in zip(bars, mean_scores):
        yval = bar.get_height()
        if yval >= 0:
            plt.text(bar.get_x() + bar.get_width() / 2, yval + offset, round(score, 3), ha='center', va='bottom', fontsize=12)
        else:
            plt.text(bar.get_x() + bar.get_width() / 2, yval - offset, round(score, 3), ha='center', va='top', fontsize=12)

    ylim_low = min(0.0, min_val * 1.2)
    ylim_high = max_val * 1.2 if max_val != 0 else 1.0
    if ylim_high <= ylim_low:
        ylim_high = ylim_low + 1.0
    plt.ylim(ylim_low, ylim_high)

    plt.title('Average Evaluation Metrics for VAE_MLP Regression')
    plt.ylabel('Score')
    plt.tight_layout()
    plt.savefig(f"{prefix}_vaemlp_reg_average_metrics.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.scatter(y_all_true, y_all_pred, alpha=0.6)
    line_min = min(y_all_true.min(), y_all_pred.min())
    line_max = max(y_all_true.max(), y_all_pred.max())
    plt.plot([line_min, line_max], [line_min, line_max], 'k--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual Values')
    plt.tight_layout()
    plt.savefig(f"{prefix}_vaemlp_reg_predictions.png", dpi=300)
    plt.close()

    residuals = y_all_true - y_all_pred
    plt.figure(figsize=(10, 8))
    sns.histplot(residuals, bins=30, alpha=0.7, kde=True, edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residuals Histogram')
    plt.tight_layout()
    plt.savefig(f"{prefix}_vaemlp_reg_residuals.png", dpi=300)
    plt.close()

    try:
        background_size = 100
        rng_bg = np.random.RandomState(42)
        background_indices = rng_bg.choice(X.shape[0], min(background_size, X.shape[0]), replace=False)
        background = X[background_indices]

        def objective_shap(trial):
            latent_dim = trial.suggest_int('latent_dim', 2, 256, log=True)
            use_batchnorm = trial.suggest_categorical('use_batchnorm', [True, False])
            vae_dropout_rate = trial.suggest_float('vae_dropout_rate', 0.0, 0.5, step=0.1)
            mlp_dropout_rate = trial.suggest_float('mlp_dropout_rate', 0.0, 0.5, step=0.1)

            vae_patience = trial.suggest_int('vae_patience', 5, 20)
            vae_min_delta = trial.suggest_float('vae_min_delta', 0.0, 0.1, step=0.01)
            mlp_patience = trial.suggest_int('mlp_patience', 5, 20)
            mlp_min_delta = trial.suggest_float('mlp_min_delta', 0.0, 0.1, step=0.01)

            num_encoder_layers = trial.suggest_int('num_encoder_layers', 1, 5)
            encoder_layers = [trial.suggest_int(f'encoder_units_l{i}', 16, 512, log=True)
                              for i in range(num_encoder_layers)]
            num_decoder_layers = trial.suggest_int('num_decoder_layers', 1, 5)
            decoder_layers = [trial.suggest_int(f'decoder_units_l{i}', 16, 512, log=True)
                              for i in range(num_decoder_layers)]
            num_mlp_layers = trial.suggest_int('num_mlp_layers', 1, 5)
            mlp_layers = [trial.suggest_int(f'mlp_units_l{i}', 16, 512, log=True)
                          for i in range(num_mlp_layers)]

            vae_learning_rate = trial.suggest_float('vae_learning_rate', 1e-5, 0.05, log=True)
            mlp_learning_rate = trial.suggest_float('mlp_learning_rate', 1e-5, 0.05, log=True)
            mlp_weight_decay = trial.suggest_float('mlp_weight_decay', 1e-8, 1e-2, log=True)
            beta = trial.suggest_float('beta', 0.1, 1.0, log=True)
            kl_anneal_fraction = trial.suggest_float('kl_anneal_fraction', 0.0, 0.5)
            epochs = trial.suggest_int('epochs', 10, 200)
            batch_size = trial.suggest_int('batch_size', 2, 256, log=True)

            model = VAE_MLP(
                input_dim=X.shape[1],
                latent_dim=latent_dim,
                encoder_layers=encoder_layers,
                decoder_layers=decoder_layers,
                mlp_layers=mlp_layers,
                vae_dropout_rate=vae_dropout_rate,
                mlp_dropout_rate=mlp_dropout_rate,
                use_batchnorm=use_batchnorm,
                vae_patience=vae_patience,
                vae_min_delta=vae_min_delta,
                mlp_patience=mlp_patience,
                mlp_min_delta=mlp_min_delta,
                vae_learning_rate=vae_learning_rate,
                mlp_learning_rate=mlp_learning_rate,
                mlp_weight_decay=mlp_weight_decay,
                beta=beta,
                kl_anneal_fraction=kl_anneal_fraction,
                epochs=epochs,
                batch_size=batch_size
            )

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            rmse_scores_shap = []

            for inner_train_idx, inner_val_idx in kf.split(X):
                X_inner_train, X_inner_val = X[inner_train_idx], X[inner_val_idx]
                y_inner_train, y_inner_val = y[inner_train_idx], y[inner_val_idx]
                scaler_inner_shap = StandardScaler()
                X_inner_train_scaled = scaler_inner_shap.fit_transform(X_inner_train)
                X_inner_val_scaled = scaler_inner_shap.transform(X_inner_val)

                fold_model_shap = VAE_MLP(
                    input_dim=model.input_dim,
                    latent_dim=model.latent_dim,
                    encoder_layers=model.encoder_layers,
                    decoder_layers=model.decoder_layers,
                    mlp_layers=model.mlp_layers,
                    vae_dropout_rate=model.vae_dropout_rate,
                    mlp_dropout_rate=model.mlp_dropout_rate,
                    use_batchnorm=model.use_batchnorm,
                    vae_patience=model.vae_patience,
                    vae_min_delta=model.vae_min_delta,
                    mlp_patience=model.mlp_patience,
                    mlp_min_delta=model.mlp_min_delta,
                    vae_learning_rate=model.vae_learning_rate,
                    mlp_learning_rate=model.mlp_learning_rate,
                    mlp_weight_decay=model.mlp_weight_decay,
                    beta=model.beta,
                    kl_anneal_fraction=model.kl_anneal_fraction,
                    epochs=model.epochs,
                    batch_size=model.batch_size
                )
                fold_model_shap.fit(X_inner_train_scaled, y_inner_train)
                y_inner_pred_shap = fold_model_shap.predict(X_inner_val_scaled)
                rmse = np.sqrt(mean_squared_error(y_inner_val, y_inner_pred_shap))
                rmse_scores_shap.append(rmse)

            return np.mean(rmse_scores_shap)

        study_shap = optuna.create_study(direction='minimize',
                                         sampler=optuna.samplers.TPESampler(seed=42))
        with SuppressOutput():
            study_shap.optimize(objective_shap, n_trials=20, n_jobs=-1)

        best_params_shap = study_shap.best_params

        latent_dim = best_params_shap['latent_dim']
        use_batchnorm = best_params_shap['use_batchnorm']
        vae_dropout_rate = best_params_shap['vae_dropout_rate']
        mlp_dropout_rate = best_params_shap['mlp_dropout_rate']
        vae_patience = best_params_shap['vae_patience']
        vae_min_delta = best_params_shap['vae_min_delta']
        mlp_patience = best_params_shap['mlp_patience']
        mlp_min_delta = best_params_shap['mlp_min_delta']
        encoder_layers = [best_params_shap[f'encoder_units_l{i}'] for i in range(best_params_shap['num_encoder_layers'])]
        decoder_layers = [best_params_shap[f'decoder_units_l{i}'] for i in range(best_params_shap['num_decoder_layers'])]
        mlp_layers = [best_params_shap[f'mlp_units_l{i}'] for i in range(best_params_shap['num_mlp_layers'])]
        vae_learning_rate = best_params_shap['vae_learning_rate']
        mlp_learning_rate = best_params_shap['mlp_learning_rate']
        mlp_weight_decay = best_params_shap['mlp_weight_decay']
        beta = best_params_shap['beta']
        kl_anneal_fraction = best_params_shap['kl_anneal_fraction']
        epochs = best_params_shap['epochs']
        batch_size = best_params_shap['batch_size']

        final_model = VAE_MLP(
            input_dim=X.shape[1],
            latent_dim=latent_dim,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            mlp_layers=mlp_layers,
            vae_dropout_rate=vae_dropout_rate,
            mlp_dropout_rate=mlp_dropout_rate,
            use_batchnorm=use_batchnorm,
            vae_patience=vae_patience,
            vae_min_delta=vae_min_delta,
            mlp_patience=mlp_patience,
            mlp_min_delta=mlp_min_delta,
            vae_learning_rate=vae_learning_rate,
            mlp_learning_rate=mlp_learning_rate,
            mlp_weight_decay=mlp_weight_decay,
            beta=beta,
            kl_anneal_fraction=kl_anneal_fraction,
            epochs=epochs,
            batch_size=batch_size
        )

        scaler_final = StandardScaler()
        X_scaled_final = scaler_final.fit_transform(X)
        final_model.fit(X_scaled_final, y)

        def model_predict(X_input):
            X_scaled = scaler_final.transform(X_input)
            return final_model.predict(X_scaled)

        explainer = shap.PermutationExplainer(model_predict, background, n_repeats=10, random_state=42, n_jobs=-1)
        shap_values = explainer.shap_values(X)

        mean_shap_values = np.mean(np.abs(shap_values), axis=0)
        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'Mean SHAP Value': mean_shap_values
        })
        shap_df.to_csv(f"{prefix}_vaemlp_reg_shap_values.csv", index=False)

        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        fig = plt.gcf()
        fig.suptitle("SHAP Summary Dot Plot for MLP_VAE_reg", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
        plt.savefig(f"{prefix}_vaemlp_reg_shap_summary.png", dpi=300)
        plt.close()

        shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type='bar', show=False)
        fig = plt.gcf()
        fig.suptitle("SHAP Mean Summary Plot for MLP_VAE_reg", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
        plt.savefig(f"{prefix}_vaemlp_reg_shap_bar.png", dpi=300)
        plt.close()

    except Exception as e:
        print(f"SHAP computation was skipped due to an error: {e}", flush=True)

if __name__ == "__main__":
    args = parse_arguments()
    prefix = Path(args.csv).stem
    if args.prefix:
        prefix = args.prefix
    vae_regression(args.csv, prefix)
