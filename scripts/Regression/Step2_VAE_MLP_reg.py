import argparse
import random
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
import contextlib
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import joblib

# ----------------------- Global Matplotlib Font Setup -----------------------
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})

# ----------------------- Argparse -----------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description='Script to run regressors')
    parser.add_argument('-i', '--csv', type=str, help='Input CSV file', required=True)
    parser.add_argument('-p', '--prefix', type=str, help='Output prefix')
    return parser.parse_args()

# ----------------------- Reproducibility -----------------------
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

# ----------------------- Warnings -----------------------
warnings.filterwarnings('ignore')

# ----------------------- Suppress Output -----------------------
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

# ----------------------- Early Stopping -----------------------
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

# ----------------------- Conditional Norm -----------------------
class ConditionalNorm(nn.Module):
    def __init__(self, units, use_batchnorm):
        super(ConditionalNorm, self).__init__()
        if use_batchnorm:
            self.norm = nn.BatchNorm1d(units)
        else:
            self.norm = nn.LayerNorm(units)
    def forward(self, x):
        return self.norm(x)

# ----------------------- VAE Components -----------------------
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
    def __init__(self, input_dim, latent_dim, encoder_layers, decoder_layers,
                 enc_dropout, dec_dropout, use_batchnorm, beta=1.0, kl_anneal_epochs=0):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim, encoder_layers, enc_dropout, use_batchnorm)
        self.decoder = Decoder(latent_dim, input_dim, decoder_layers, dec_dropout, use_batchnorm)
        self.beta = beta
        self.kl_anneal_epochs = kl_anneal_epochs
    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z_log_var = torch.clamp(z_log_var, min=-10, max=10)
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        z = z_mean + eps * std
        x_recon = self.decoder(z)
        return x_recon, z_mean, z_log_var
    def loss_components(self, x, x_recon, z_mean, z_log_var):
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        return recon_loss, kl_loss

# ----------------------- MLP Regressor (uses x_recon as input) -----------------------
class MLPRegressorTorch(nn.Module):
    def __init__(self, input_dim, output_dim, mlp_layers, dropout_rate, use_batchnorm):
        super(MLPRegressorTorch, self).__init__()
        layers = []
        prev_dim = input_dim
        for units in mlp_layers:
            layers.append(nn.Linear(prev_dim, units))
            layers.append(nn.LeakyReLU())
            layers.append(ConditionalNorm(units, use_batchnorm))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = units
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)

# ----------------------- VAE + MLP Wrapper -----------------------
class VAE_MLP(BaseEstimator, RegressorMixin):
    def __init__(self,
                 input_dim=30, output_dim=1, latent_dim=16,
                 encoder_layers=[64, 32], decoder_layers=[32, 64], mlp_layers=[64, 32],
                 vae_dropout_enc=0.2, vae_dropout_dec=0.2, mlp_dropout=0.2,
                 use_batchnorm=True,
                 vae_patience=15, vae_min_delta=1e-5,
                 mlp_patience=15, mlp_min_delta=1e-5,
                 vae_learning_rate=1e-3, mlp_learning_rate=1e-3,
                 mlp_weight_decay=1e-5,
                 epochs=100, batch_size=32,
                 beta=1.0, kl_anneal_epochs=0):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.mlp_layers = mlp_layers
        self.vae_dropout_enc = vae_dropout_enc
        self.vae_dropout_dec = vae_dropout_dec
        self.mlp_dropout = mlp_dropout
        self.use_batchnorm = use_batchnorm
        self.vae_patience = vae_patience
        self.vae_min_delta = vae_min_delta
        self.mlp_patience = mlp_patience
        self.mlp_min_delta = mlp_min_delta
        self.vae_learning_rate = vae_learning_rate
        self.mlp_learning_rate = mlp_learning_rate
        self.mlp_weight_decay = mlp_weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.beta = beta
        self.kl_anneal_epochs = kl_anneal_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae = None
        self.vae_optimizer = None
        self.vae_scheduler = None
        self.mlp_model = None
        self.mlp_optimizer = None
        self.mlp_scheduler = None

    def _create_vae(self):
        self.vae = VAE(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            encoder_layers=self.encoder_layers,
            decoder_layers=self.decoder_layers,
            enc_dropout=self.vae_dropout_enc,
            dec_dropout=self.vae_dropout_dec,
            use_batchnorm=self.use_batchnorm,
            beta=self.beta,
            kl_anneal_epochs=self.kl_anneal_epochs
        ).to(self.device)
        self.vae_optimizer = optim.Adam(self.vae.parameters(), lr=self.vae_learning_rate, weight_decay=0.0)
        self.vae_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.vae_optimizer, mode='min', factor=0.5,
                                                                  patience=max(1, self.vae_patience // 3),
                                                                  min_lr=1e-6, verbose=False)

    def _create_mlp(self):
        self.mlp_model = MLPRegressorTorch(
            input_dim=self.input_dim,  # x_recon dimension equals original input_dim
            output_dim=self.output_dim,
            mlp_layers=self.mlp_layers,
            dropout_rate=self.mlp_dropout,
            use_batchnorm=self.use_batchnorm
        ).to(self.device)
        self.mlp_optimizer = optim.Adam(self.mlp_model.parameters(),
                                        lr=self.mlp_learning_rate,
                                        weight_decay=self.mlp_weight_decay)
        self.mlp_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.mlp_optimizer, mode='min', factor=0.5,
                                                                  patience=max(1, self.mlp_patience // 3),
                                                                  min_lr=1e-6, verbose=False)

    def fit(self, X, y):
        use_batchnorm = self.use_batchnorm
        self._create_vae()
        # VAE train/val split
        X_train_vae, X_val_vae, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_vae_tensor = torch.from_numpy(X_train_vae).float().to(self.device)
        X_val_vae_tensor = torch.from_numpy(X_val_vae).float().to(self.device)
        train_ds_vae = torch.utils.data.TensorDataset(X_train_vae_tensor)
        val_ds_vae = torch.utils.data.TensorDataset(X_val_vae_tensor)
        train_loader_vae = torch.utils.data.DataLoader(train_ds_vae, batch_size=self.batch_size,
                                                       shuffle=True, drop_last=True)
        val_loader_vae = torch.utils.data.DataLoader(val_ds_vae, batch_size=self.batch_size,
                                                     shuffle=False, drop_last=False)

        es_vae = EarlyStopping(patience=self.vae_patience, min_delta=self.vae_min_delta, verbose=False)

        self.vae.train()
        for epoch in range(self.epochs):
            # Linear KL annealing factor
            if self.kl_anneal_epochs and self.kl_anneal_epochs > 0:
                anneal_factor = min(1.0, (epoch + 1) / float(self.kl_anneal_epochs))
            else:
                anneal_factor = 1.0
            beta_eff = self.beta * anneal_factor

            # Train epoch
            for data in train_loader_vae:
                x_batch = data[0]
                self.vae_optimizer.zero_grad()
                x_recon, z_mean, z_log_var = self.vae(x_batch)
                recon_loss, kl_loss = self.vae.loss_components(x_batch, x_recon, z_mean, z_log_var)
                loss = recon_loss + beta_eff * kl_loss
                loss.backward()
                self.vae_optimizer.step()

            # Validate
            self.vae.eval()
            val_loss_sum = 0.0
            n_val = 0
            with torch.no_grad():
                for data in val_loader_vae:
                    x_batch = data[0]
                    x_recon, z_mean, z_log_var = self.vae(x_batch)
                    recon_loss, kl_loss = self.vae.loss_components(x_batch, x_recon, z_mean, z_log_var)
                    loss = recon_loss + beta_eff * kl_loss
                    bs = x_batch.size(0)
                    val_loss_sum += loss.item()
                    n_val += bs
            val_loss_avg = val_loss_sum / max(1, n_val)
            self.vae_scheduler.step(val_loss_avg)
            stop = es_vae(val_loss_avg)
            self.vae.train()
            if stop:
                break

        # Build reconstructed features for whole X
        self.vae.eval()
        with torch.no_grad():
            X_tensor_full = torch.from_numpy(X).float().to(self.device)
            ds_full = torch.utils.data.TensorDataset(X_tensor_full)
            loader_full = torch.utils.data.DataLoader(ds_full, batch_size=self.batch_size,
                                                      shuffle=False, drop_last=False)
            X_recon_list = []
            for data in loader_full:
                x_batch = data[0]
                x_recon, _, _ = self.vae(x_batch)
                X_recon_list.append(x_recon.cpu())
            X_encoded = torch.cat(X_recon_list, dim=0).numpy()

        # MLP: train/val on reconstructed features
        self._create_mlp()
        X_train_mlp, X_val_mlp, y_train_mlp, y_val_mlp = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42)
        X_train_mlp_tensor = torch.from_numpy(X_train_mlp).float().to(self.device)
        y_train_mlp_tensor = torch.from_numpy(y_train_mlp).float().to(self.device)
        X_val_mlp_tensor = torch.from_numpy(X_val_mlp).float().to(self.device)
        y_val_mlp_tensor = torch.from_numpy(y_val_mlp).float().to(self.device)
        train_ds_mlp = torch.utils.data.TensorDataset(X_train_mlp_tensor, y_train_mlp_tensor)
        val_ds_mlp = torch.utils.data.TensorDataset(X_val_mlp_tensor, y_val_mlp_tensor)
        train_loader_mlp = torch.utils.data.DataLoader(train_ds_mlp, batch_size=self.batch_size,
                                                       shuffle=True, drop_last=True)
        val_loader_mlp = torch.utils.data.DataLoader(val_ds_mlp, batch_size=self.batch_size,
                                                     shuffle=False, drop_last=False)

        es_mlp = EarlyStopping(patience=self.mlp_patience, min_delta=self.mlp_min_delta, verbose=False)
        self.mlp_model.train()
        criterion = nn.MSELoss(reduction='sum')

        for epoch in range(self.epochs):
            # Train epoch
            for xb, yb in train_loader_mlp:
                self.mlp_optimizer.zero_grad()
                preds = self.mlp_model(xb).squeeze(-1)
                loss = criterion(preds, yb)
                loss.backward()
                self.mlp_optimizer.step()
            # Validate
            self.mlp_model.eval()
            val_loss_sum = 0.0
            n_val = 0
            with torch.no_grad():
                for xb, yb in val_loader_mlp:
                    preds = self.mlp_model(xb).squeeze(-1)
                    loss = criterion(preds, yb)
                    bs = xb.size(0)
                    val_loss_sum += loss.item()
                    n_val += bs
            val_loss_avg = val_loss_sum / max(1, n_val)
            self.mlp_scheduler.step(val_loss_avg)
            stop = es_mlp(val_loss_avg)
            self.mlp_model.train()
            if stop:
                break
        return self

    def predict(self, X):
        self.vae.eval()
        self.mlp_model.eval()
        X_tensor = torch.from_numpy(X).float().to(self.device)
        ds = torch.utils.data.TensorDataset(X_tensor)
        loader = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=False, drop_last=False)
        with torch.no_grad():
            X_recon_list = []
            for data in loader:
                x_batch = data[0]
                x_recon, _, _ = self.vae(x_batch)
                X_recon_list.append(x_recon.cpu())
            X_encoded = torch.cat(X_recon_list, dim=0).numpy()
            y_pred_list = []
            ds_mlp = torch.utils.data.TensorDataset(torch.from_numpy(X_encoded).float().to(self.device))
            loader_mlp = torch.utils.data.DataLoader(ds_mlp, batch_size=self.batch_size,
                                                     shuffle=False, drop_last=False)
            for data in loader_mlp:
                xb = data[0].to(self.device)
                preds = self.mlp_model(xb).squeeze(-1)     
                y_pred_list.append(preds.detach().cpu())    

            y_pred = torch.cat([t.reshape(-1) for t in y_pred_list], dim=0).numpy()
            return y_pred

# ----------------------- Main VAE Flow -----------------------
def vae(inp, prefix):
    data = pd.read_csv(inp)
    sample_ids = data['SampleID']
    X_raw = data.drop(columns=['SampleID', 'Label']).values
    y = data['Label'].values.astype(np.float32)
    feature_names = data.drop(columns=['SampleID', 'Label']).columns

    # Stratify bins
    y_binned, _ = pd.qcut(y, q=5, labels=False, retbins=True, duplicates='drop')
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    mses, rmses, maes, r2s, fold_numbers = [], [], [], [], []
    y_tests, y_preds = [], []

    # ---------------- Outer CV (unbiased evaluation) ----------------
    for fold_number, (train_idx, test_idx) in enumerate(outer_cv.split(X_raw, y_binned), 1):
        print(f"Processing fold {fold_number}")
        X_train_outer_raw, X_test_outer_raw = X_raw[train_idx], X_raw[test_idx]
        y_train_outer, y_test_outer = y[train_idx], y[test_idx]
        y_train_binned_outer = y_binned[train_idx]

        # Outer scaler fit only on outer-train
        scaler_outer = StandardScaler()
        X_train_outer_scaled = scaler_outer.fit_transform(X_train_outer_raw)
        X_test_outer_scaled = scaler_outer.transform(X_test_outer_raw)

        # ---------------- Inner CV objective uses RAW outer-train and fits inner scaler inside ----------------
        def objective(trial):
            # Hyperparameters
            latent_dim = trial.suggest_int('latent_dim', 2, 256, log=True)
            vae_dropout_enc = trial.suggest_float('vae_dropout_enc', 0.0, 0.5, step=0.1)
            vae_dropout_dec = trial.suggest_float('vae_dropout_dec', 0.0, 0.5, step=0.1)
            mlp_dropout = trial.suggest_float('mlp_dropout', 0.0, 0.5, step=0.1)
            use_batchnorm = trial.suggest_categorical('use_batchnorm', [True, False])

            vae_patience = trial.suggest_int('vae_patience', 5, 30)
            vae_min_delta = trial.suggest_float('vae_min_delta', 1e-6, 1e-3, log=True)
            mlp_patience = trial.suggest_int('mlp_patience', 5, 30)
            mlp_min_delta = trial.suggest_float('mlp_min_delta', 1e-6, 1e-3, log=True)

            vae_learning_rate = trial.suggest_float('vae_learning_rate', 1e-5, 5e-3, log=True)
            mlp_learning_rate = trial.suggest_float('mlp_learning_rate', 1e-5, 5e-3, log=True)
            mlp_weight_decay = trial.suggest_float('mlp_weight_decay', 1e-6, 1e-2, log=True)

            epochs = trial.suggest_int('epochs', 20, 200)
            batch_size = trial.suggest_int('batch_size', 2, 256, log=True)

            num_encoder_layers = trial.suggest_int('num_encoder_layers', 1, 5)
            encoder_layers = [trial.suggest_int(f'encoder_units_l{i}', 16, 512, log=True)
                              for i in range(num_encoder_layers)]
            num_decoder_layers = trial.suggest_int('num_decoder_layers', 1, 5)
            decoder_layers = [trial.suggest_int(f'decoder_units_l{i}', 16, 512, log=True)
                              for i in range(num_decoder_layers)]
            num_mlp_layers = trial.suggest_int('num_mlp_layers', 1, 5)
            mlp_layers = [trial.suggest_int(f'mlp_units_l{i}', 16, 512, log=True)
                          for i in range(num_mlp_layers)]

            beta = trial.suggest_float('beta', 0.1, 1.0)
            kl_anneal_epochs = trial.suggest_int('kl_anneal_epochs', 0, epochs)

            inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            mses_inner = []

            for inner_train_idx, inner_val_idx in inner_cv.split(X_train_outer_raw, y_train_binned_outer):
                X_inner_train_raw = X_train_outer_raw[inner_train_idx]
                X_inner_val_raw = X_train_outer_raw[inner_val_idx]
                y_inner_train = y_train_outer[inner_train_idx]
                y_inner_val = y_train_outer[inner_val_idx]

                # Inner scaler fit only on inner-train
                scaler_inner = StandardScaler()
                X_inner_train = scaler_inner.fit_transform(X_inner_train_raw)
                X_inner_val = scaler_inner.transform(X_inner_val_raw)

                model = VAE_MLP(
                    input_dim=X_inner_train.shape[1],
                    output_dim=1,
                    latent_dim=latent_dim,
                    encoder_layers=encoder_layers,
                    decoder_layers=decoder_layers,
                    mlp_layers=mlp_layers,
                    vae_dropout_enc=vae_dropout_enc,
                    vae_dropout_dec=vae_dropout_dec,
                    mlp_dropout=mlp_dropout,
                    use_batchnorm=use_batchnorm,
                    vae_patience=vae_patience,
                    vae_min_delta=vae_min_delta,
                    mlp_patience=mlp_patience,
                    mlp_min_delta=mlp_min_delta,
                    vae_learning_rate=vae_learning_rate,
                    mlp_learning_rate=mlp_learning_rate,
                    mlp_weight_decay=mlp_weight_decay,
                    epochs=epochs,
                    batch_size=batch_size,
                    beta=beta,
                    kl_anneal_epochs=kl_anneal_epochs
                )

                model.fit(X_inner_train, y_inner_train)
                y_pred_inner = model.predict(X_inner_val)
                mses_inner.append(mean_squared_error(y_inner_val, y_pred_inner))

            return float(np.mean(mses_inner))

        study = optuna.create_study(direction='minimize',
                                    sampler=optuna.samplers.TPESampler(seed=42))
        with SuppressOutput():
            study.optimize(objective, n_trials=50, n_jobs=-1)

        best = study.best_params
        # Train fold-final model on outer-train (scaled by outer scaler), evaluate on outer-test
        final_model = VAE_MLP(
            input_dim=X_train_outer_scaled.shape[1],
            output_dim=1,
            latent_dim=best['latent_dim'],
            encoder_layers=[best[f'encoder_units_l{i}'] for i in range(best['num_encoder_layers'])],
            decoder_layers=[best[f'decoder_units_l{i}'] for i in range(best['num_decoder_layers'])],
            mlp_layers=[best[f'mlp_units_l{i}'] for i in range(best['num_mlp_layers'])],
            vae_dropout_enc=best['vae_dropout_enc'],
            vae_dropout_dec=best['vae_dropout_dec'],
            mlp_dropout=best['mlp_dropout'],
            use_batchnorm=best['use_batchnorm'],
            vae_patience=best['vae_patience'],
            vae_min_delta=best['vae_min_delta'],
            mlp_patience=best['mlp_patience'],
            mlp_min_delta=best['mlp_min_delta'],
            vae_learning_rate=best['vae_learning_rate'],
            mlp_learning_rate=best['mlp_learning_rate'],
            mlp_weight_decay=best['mlp_weight_decay'],
            epochs=best['epochs'],
            batch_size=best['batch_size'],
            beta=best['beta'],
            kl_anneal_epochs=best['kl_anneal_epochs']
        )
        final_model.fit(X_train_outer_scaled, y_train_outer)
        y_pred_outer = final_model.predict(X_test_outer_scaled)

        mse = mean_squared_error(y_test_outer, y_pred_outer)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_outer, y_pred_outer)
        r2 = r2_score(y_test_outer, y_pred_outer)

        mses.append(mse); rmses.append(rmse); maes.append(mae); r2s.append(r2); fold_numbers.append(fold_number)
        y_tests.extend(y_test_outer); y_preds.extend(y_pred_outer)

    # ---------------- Save CV metrics ----------------
    metrics_df = pd.DataFrame({'Fold': fold_numbers, 'MSE': mses, 'RMSE': rmses, 'MAE': maes, 'R2': r2s})
    metrics_df.to_csv(f"{prefix}_vae_reg_metrics_over_folds.csv", index=False)
    print(metrics_df)

    # Line plot over folds
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(fold_numbers, mses, marker='o', label='MSE')
    ax.plot(fold_numbers, rmses, marker='o', label='RMSE')
    ax.plot(fold_numbers, maes, marker='o', label='MAE')
    ax.plot(fold_numbers, r2s, marker='o', label='R2')
    ax.set_xlabel('Fold'); ax.set_ylabel('Metric Score'); ax.set_title('Metrics over Folds')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{prefix}_vae_reg_metrics_over_folds.png", dpi=300)
    plt.close(fig)

    # Average metrics bar chart
    mean_mse, mean_rmse, mean_mae, mean_r2 = np.mean(mses), np.mean(rmses), np.mean(maes), np.mean(r2s)
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ['MSE', 'RMSE', 'MAE', 'R2']
    values = [mean_mse, mean_rmse, mean_mae, mean_r2]
    bars = ax.bar(labels, values)
    y_max = max(values) if len(values) > 0 else 1.0
    ax.set_ylim(0, y_max * 1.15)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.01, f"{val:.4f}", ha='center', va='bottom')
    ax.set_xlabel('Metrics'); ax.set_ylabel('Average Score'); ax.set_title('Average Metrics over 5 Folds')
    fig.tight_layout()
    fig.savefig(f"{prefix}_vae_reg_average_metrics.png", dpi=300)
    plt.close(fig)

    # Actual vs Predicted (outer test predictions)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_tests, y_preds, alpha=0.6)
    lo, hi = min(y_tests + y_preds), max(y_tests + y_preds)
    ax.plot([lo, hi], [lo, hi], 'r--')
    ax.set_xlabel('Actual Values'); ax.set_ylabel('Predicted Values'); ax.set_title('Actual vs Predicted')
    fig.tight_layout()
    fig.savefig(f"{prefix}_vae_reg_predictions.png", dpi=300)
    plt.close(fig)

    # Residuals
    residuals = np.array(y_tests) - np.array(y_preds)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(residuals, bins=30, kde=True, edgecolor='black', alpha=0.7, ax=ax)
    ax.set_xlabel('Residuals'); ax.set_ylabel('Frequency'); ax.set_title('Residuals Histogram')
    fig.tight_layout()
    fig.savefig(f"{prefix}_vae_reg_residuals.png", dpi=300)
    plt.close(fig)

    # ---------------- Final model selection on full data (separate CV) ----------------
    def full_objective(trial):
        latent_dim = trial.suggest_int('latent_dim', 2, 256, log=True)
        vae_dropout_enc = trial.suggest_float('vae_dropout_enc', 0.0, 0.5, step=0.1)
        vae_dropout_dec = trial.suggest_float('vae_dropout_dec', 0.0, 0.5, step=0.1)
        mlp_dropout = trial.suggest_float('mlp_dropout', 0.0, 0.5, step=0.1)
        use_batchnorm = trial.suggest_categorical('use_batchnorm', [True, False])

        vae_patience = trial.suggest_int('vae_patience', 5, 30)
        vae_min_delta = trial.suggest_float('vae_min_delta', 1e-6, 1e-3, log=True)
        mlp_patience = trial.suggest_int('mlp_patience', 5, 30)
        mlp_min_delta = trial.suggest_float('mlp_min_delta', 1e-6, 1e-3, log=True)

        vae_learning_rate = trial.suggest_float('vae_learning_rate', 1e-5, 5e-3, log=True)
        mlp_learning_rate = trial.suggest_float('mlp_learning_rate', 1e-5, 5e-3, log=True)
        mlp_weight_decay = trial.suggest_float('mlp_weight_decay', 1e-6, 1e-2, log=True)

        epochs = trial.suggest_int('epochs', 20, 200)
        batch_size = trial.suggest_int('batch_size', 2, 256, log=True)

        num_encoder_layers = trial.suggest_int('num_encoder_layers', 1, 5)
        encoder_layers = [trial.suggest_int(f'encoder_units_l{i}', 16, 512, log=True)
                          for i in range(num_encoder_layers)]
        num_decoder_layers = trial.suggest_int('num_decoder_layers', 1, 5)
        decoder_layers = [trial.suggest_int(f'decoder_units_l{i}', 16, 512, log=True)
                          for i in range(num_decoder_layers)]
        num_mlp_layers = trial.suggest_int('num_mlp_layers', 1, 5)
        mlp_layers = [trial.suggest_int(f'mlp_units_l{i}', 16, 512, log=True)
                      for i in range(num_mlp_layers)]

        beta = trial.suggest_float('beta', 0.1, 1.0)
        kl_anneal_epochs = trial.suggest_int('kl_anneal_epochs', 0, epochs)

        # 5-fold CV on full data with per-fold scaler
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
        mses_cv = []
        for tr_idx, va_idx in inner_cv.split(X_raw):
            X_tr_raw, X_va_raw = X_raw[tr_idx], X_raw[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr_raw)
            X_va = scaler.transform(X_va_raw)

            model = VAE_MLP(
                input_dim=X_tr.shape[1],
                output_dim=1,
                latent_dim=latent_dim,
                encoder_layers=encoder_layers,
                decoder_layers=decoder_layers,
                mlp_layers=mlp_layers,
                vae_dropout_enc=vae_dropout_enc,
                vae_dropout_dec=vae_dropout_dec,
                mlp_dropout=mlp_dropout,
                use_batchnorm=use_batchnorm,
                vae_patience=vae_patience,
                vae_min_delta=vae_min_delta,
                mlp_patience=mlp_patience,
                mlp_min_delta=mlp_min_delta,
                vae_learning_rate=vae_learning_rate,
                mlp_learning_rate=mlp_learning_rate,
                mlp_weight_decay=mlp_weight_decay,
                epochs=epochs,
                batch_size=batch_size,
                beta=beta,
                kl_anneal_epochs=kl_anneal_epochs
            )
            model.fit(X_tr, y_tr)
            preds = model.predict(X_va)
            mses_cv.append(mean_squared_error(y_va, preds))
        return float(np.mean(mses_cv))

    study_full = optuna.create_study(direction='minimize',
                                     sampler=optuna.samplers.TPESampler(seed=42))
    with SuppressOutput():
        study_full.optimize(full_objective, n_trials=50, n_jobs=-1)

    best_full = study_full.best_params
    scaler_final = StandardScaler()
    X_scaled_final = scaler_final.fit_transform(X_raw)

    final_model = VAE_MLP(
        input_dim=X_scaled_final.shape[1],
        output_dim=1,
        latent_dim=best_full['latent_dim'],
        encoder_layers=[best_full[f'encoder_units_l{i}'] for i in range(best_full['num_encoder_layers'])],
        decoder_layers=[best_full[f'decoder_units_l{i}'] for i in range(best_full['num_decoder_layers'])],
        mlp_layers=[best_full[f'mlp_units_l{i}'] for i in range(best_full['num_mlp_layers'])],
        vae_dropout_enc=best_full['vae_dropout_enc'],
        vae_dropout_dec=best_full['vae_dropout_dec'],
        mlp_dropout=best_full['mlp_dropout'],
        use_batchnorm=best_full['use_batchnorm'],
        vae_patience=best_full['vae_patience'],
        vae_min_delta=best_full['vae_min_delta'],
        mlp_patience=best_full['mlp_patience'],
        mlp_min_delta=best_full['mlp_min_delta'],
        vae_learning_rate=best_full['vae_learning_rate'],
        mlp_learning_rate=best_full['mlp_learning_rate'],
        mlp_weight_decay=best_full['mlp_weight_decay'],
        epochs=best_full['epochs'],
        batch_size=best_full['batch_size'],
        beta=best_full['beta'],
        kl_anneal_epochs=best_full['kl_anneal_epochs']
    )
    final_model.fit(X_scaled_final, y)
    y_pred_final = final_model.predict(X_scaled_final)

    # Save predictions on full data
    results_df = pd.DataFrame({'SampleID': sample_ids, 'Actual Value': y, 'Predicted Value': y_pred_final})
    results_df.to_csv(f"{prefix}_vae_reg_predictions.csv", index=False)

    # Save model and scaler
    joblib.dump(final_model, f"{prefix}_vae_reg_model.pkl")
    joblib.dump(scaler_final, f"{prefix}_vae_reg_scaler.pkl")
    joblib.dump((X_raw, y), f"{prefix}_vae_reg_data.pkl")

    # ---------------- SHAP (with consistent scaling) ----------------
    try:
        background_size = 100
        if X_raw.shape[0] > background_size:
            idx = np.random.choice(X_raw.shape[0], background_size, replace=False)
            background_raw = X_raw[idx]
        else:
            background_raw = X_raw
        background = scaler_final.transform(background_raw)
        X_for_shap = scaler_final.transform(X_raw)

        def model_predict_scaled(x_scaled):
            return final_model.predict(x_scaled)

        explainer = shap.Explainer(model_predict_scaled, background)
        shap_values = explainer(X_for_shap)

        # Mean |SHAP|
        mean_shap_values = np.abs(shap_values.values).mean(axis=0)
        shap_df = pd.DataFrame({'Feature': feature_names, 'Mean SHAP Value': mean_shap_values})
        shap_df.to_csv(f"{prefix}_vae_reg_shap_values.csv", index=False)
        print(f"SHAP values have been saved to {prefix}_vae_reg_shap_values.csv")

        # SHAP summary plot (beeswarm)
        shap.summary_plot(shap_values, X_for_shap, feature_names=feature_names, show=False)
        plt.title("SHAP Summary Dot Plot for VAE_reg")
        plt.tight_layout()
        plt.savefig(f"{prefix}_vae_reg_shap_summary.png", dpi=300)
        plt.close()
        
        # SHAP mean bar plot
        shap.summary_plot(shap_values, X_for_shap, feature_names=feature_names, plot_type='bar', show=False)
        plt.title("SHAP Mean Summary Plot for VAE_reg")
        plt.tight_layout()
        plt.savefig(f"{prefix}_vae_reg_shap_bar.png", dpi=300)
        plt.close()

    except Exception as e:
        print(f"SHAP computation failed: {e}")
        print("Proceeding without SHAP analysis.")

if __name__ == "__main__":
    args = parse_arguments()
    prefix = Path(args.csv).stem
    if args.prefix:
        prefix = args.prefix
    vae(args.csv, prefix)
