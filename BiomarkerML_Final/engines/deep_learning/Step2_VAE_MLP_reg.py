import argparse
import random
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
import warnings
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import joblib

SCRIPT_ROOT = str(Path(__file__).resolve().parents[1])
if SCRIPT_ROOT not in sys.path:
    sys.path.insert(0, SCRIPT_ROOT)
from core.data_contracts import RoleAwarePreprocessor, modeling_frame, resolve_feature_roles, write_missingness_qc, write_reproducibility_manifest, imputation_fallback_reasons
from core.adaptive_optuna import run_adaptive_study, tolerance_for_objective
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
    export_outer_fold_biomarkers,
    estimator_importance_with_source,
    generate_training_size_curve,
    write_training_history,
)

configure_publication_plots()


def parse_arguments():
    parser = argparse.ArgumentParser(description='Script to run regressors')
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
        self.training_history_ = []
        # VAE train/val split
        X_train_vae, X_val_vae, _, _ = train_test_split(X, y, test_size=0.2, random_state=DEFAULT_RANDOM_STATE)
        X_train_vae_tensor = torch.from_numpy(X_train_vae).float().to(self.device)
        X_val_vae_tensor = torch.from_numpy(X_val_vae).float().to(self.device)
        train_ds_vae = torch.utils.data.TensorDataset(X_train_vae_tensor)
        val_ds_vae = torch.utils.data.TensorDataset(X_val_vae_tensor)
        vae_train_batch_size = min(self.batch_size, max(1, len(X_train_vae)))
        train_loader_vae = torch.utils.data.DataLoader(train_ds_vae, batch_size=vae_train_batch_size,
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

            train_loss_sum = 0.0
            n_train = 0
            for data in train_loader_vae:
                x_batch = data[0]
                self.vae_optimizer.zero_grad()
                x_recon, z_mean, z_log_var = self.vae(x_batch)
                recon_loss, kl_loss = self.vae.loss_components(x_batch, x_recon, z_mean, z_log_var)
                loss = recon_loss + beta_eff * kl_loss
                loss.backward()
                self.vae_optimizer.step()
                train_loss_sum += loss.item()
                n_train += x_batch.size(0)
            train_loss_avg = train_loss_sum / max(1, n_train)

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
            stop = es_vae(val_loss_avg, epoch + 1, checkpoint_modules=(self.vae,))
            self.training_history_.append({
                "Epoch": epoch + 1,
                "TrainingLoss": train_loss_avg,
                "ValidationLoss": val_loss_avg,
                "LearningRate": self.vae_optimizer.param_groups[0]["lr"],
                "EarlyStoppingTriggered": stop,
                "BestEpoch": es_vae.best_epoch,
            })
            self.vae.train()
            if stop:
                break

        es_vae.restore_checkpoint(self.vae)

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
            X_encoded, y, test_size=0.2, random_state=DEFAULT_RANDOM_STATE)
        X_train_mlp_tensor = torch.from_numpy(X_train_mlp).float().to(self.device)
        y_train_mlp_tensor = torch.from_numpy(y_train_mlp).float().to(self.device)
        X_val_mlp_tensor = torch.from_numpy(X_val_mlp).float().to(self.device)
        y_val_mlp_tensor = torch.from_numpy(y_val_mlp).float().to(self.device)
        train_ds_mlp = torch.utils.data.TensorDataset(X_train_mlp_tensor, y_train_mlp_tensor)
        val_ds_mlp = torch.utils.data.TensorDataset(X_val_mlp_tensor, y_val_mlp_tensor)
        mlp_train_batch_size = min(self.batch_size, max(1, len(X_train_mlp)))
        train_loader_mlp = torch.utils.data.DataLoader(train_ds_mlp, batch_size=mlp_train_batch_size,
                                                       shuffle=True, drop_last=True)
        val_loader_mlp = torch.utils.data.DataLoader(val_ds_mlp, batch_size=self.batch_size,
                                                     shuffle=False, drop_last=False)

        es_mlp = EarlyStopping(patience=self.mlp_patience, min_delta=self.mlp_min_delta, verbose=False)
        self.mlp_model.train()
        criterion = nn.MSELoss(reduction='sum')

        offset = len(self.training_history_)
        for epoch in range(self.epochs):
            train_loss_sum = 0.0
            n_train = 0
            for xb, yb in train_loader_mlp:
                self.mlp_optimizer.zero_grad()
                preds = self.mlp_model(xb).squeeze(-1)
                loss = criterion(preds, yb)
                loss.backward()
                self.mlp_optimizer.step()
                train_loss_sum += loss.item()
                n_train += xb.size(0)
            train_loss_avg = train_loss_sum / max(1, n_train)
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
            stop = es_mlp(val_loss_avg, epoch + 1, checkpoint_modules=(self.mlp_model,))
            self.training_history_.append({
                "Epoch": offset + epoch + 1,
                "TrainingLoss": train_loss_avg,
                "ValidationLoss": val_loss_avg,
                "LearningRate": self.mlp_optimizer.param_groups[0]["lr"],
                "EarlyStoppingTriggered": stop,
                "BestEpoch": offset + es_mlp.best_epoch,
            })
            self.mlp_model.train()
            if stop:
                break
        es_mlp.restore_checkpoint(self.mlp_model)
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
                x_recon, _, _ = self.vae(X_tensor[start:start + batch_size])
                predictions.append(self.mlp_model(x_recon).squeeze(-1).cpu())
        torch.empty((), dtype=torch.int64).random_()
        if not predictions:
            return np.array([])
        return torch.cat(predictions, dim=0).numpy()

def vae(inp, prefix, imputation="median", covariates=None, categorical_covariates=None):
    data = pd.read_csv(inp)
    roles = resolve_feature_roles(data, target="Label", identifier="SampleID", covariates=covariates, categorical_covariates=categorical_covariates)
    sample_ids = data['SampleID'] if 'SampleID' in data.columns else pd.Series([f"Sample_{i+1}" for i in range(len(data))])
    X_raw = modeling_frame(data, roles)
    y = pd.to_numeric(data['Label'], errors='raise').to_numpy(dtype=np.float32)
    feature_names = list(roles.biomarkers)

    def make_preprocessor():
        return RoleAwarePreprocessor(roles.biomarkers, roles.numeric_covariates, roles.categorical_covariates, imputation=imputation)

    # Stratify bins
    y_binned, _ = pd.qcut(y, q=5, labels=False, retbins=True, duplicates='drop')
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=DEFAULT_RANDOM_STATE)

    mses, rmses, maes, r2s, fold_numbers = [], [], [], [], []
    y_tests, y_preds, y_test_indices, y_test_folds = [], [], [], []
    missingness_folds = []
    tuning_reports = []

    for fold_number, (train_idx, test_idx) in enumerate(outer_cv.split(X_raw, y_binned), 1):
        print(f"Processing fold {fold_number}")
        X_train_outer_raw, X_test_outer_raw = X_raw.iloc[train_idx], X_raw.iloc[test_idx]
        y_train_outer, y_test_outer = y[train_idx], y[test_idx]
        y_train_binned_outer = y_binned[train_idx]

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

            inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
            mses_inner = []

            for inner_train_idx, inner_val_idx in inner_cv.split(X_train_outer_raw, y_train_binned_outer):
                X_inner_train_raw = X_train_outer_raw.iloc[inner_train_idx]
                X_inner_val_raw = X_train_outer_raw.iloc[inner_val_idx]
                y_inner_train = y_train_outer[inner_train_idx]
                y_inner_val = y_train_outer[inner_val_idx]
                inner_preprocessor = make_preprocessor()
                X_inner_train = inner_preprocessor.fit_transform(X_inner_train_raw, y_inner_train)
                X_inner_val = inner_preprocessor.transform(X_inner_val_raw)

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

                model.fit(np.asarray(X_inner_train), y_inner_train)
                y_pred_inner = model.predict(np.asarray(X_inner_val))
                mses_inner.append(mean_squared_error(y_inner_val, y_pred_inner))

            return float(np.mean(mses_inner))

        study = optuna.create_study(direction='minimize',
                                    sampler=optuna.samplers.TPESampler(seed=DEFAULT_RANDOM_STATE))
        with SuppressOutput():
            tuning_reports.append(
                run_adaptive_study(
                    study,
                    objective,
                    task="regression",
                    model="VAE_reg",
                    study_scope="outer_fold",
                    objective_name="vae_regression_mse",
                    direction="minimize",
                    tolerance=tolerance_for_objective(
                        "vae_regression_mse",
                        scale_floor=max(float(np.nanvar(y_train_outer)), np.finfo(float).eps),
                    ),
                    seed=DEFAULT_RANDOM_STATE,
                    study_id=f"regression-vae-outer-fold-{fold_number}",
                    outer_fold=fold_number,
                    provenance_path=f"{prefix}_vae_reg_fold{fold_number}_optuna_study.json",
                )
            )

        best = study.best_params
        outer_preprocessor = make_preprocessor()
        X_train_outer_scaled = outer_preprocessor.fit_transform(X_train_outer_raw, y_train_outer)
        X_test_outer_scaled = outer_preprocessor.transform(X_test_outer_raw)
        missingness_folds.append(outer_preprocessor.biomarker_diagnostics(fold_number))
        # Train fold-final model on outer-train, evaluate only on untouched outer-test.
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
        final_model.fit(np.asarray(X_train_outer_scaled), y_train_outer)
        fold_importance, fold_importance_type = estimator_importance_with_source(final_model, X_train_outer_scaled.shape[1])
        fold_biomarkers = outer_preprocessor.selected_biomarker_names_
        export_outer_fold_biomarkers(prefix, "VAE_MLP_reg", fold_number, None, fold_biomarkers, selection_method="None", importance_values=np.asarray(fold_importance)[:len(fold_biomarkers)], importance_type=fold_importance_type)
        y_pred_outer = final_model.predict(np.asarray(X_test_outer_scaled))

        mse = mean_squared_error(y_test_outer, y_pred_outer)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_outer, y_pred_outer)
        r2 = r2_score(y_test_outer, y_pred_outer)

        mses.append(mse); rmses.append(rmse); maes.append(mae); r2s.append(r2); fold_numbers.append(fold_number)
        y_tests.extend(y_test_outer)
        y_preds.extend(y_pred_outer)
        y_test_indices.extend(test_idx)
        y_test_folds.extend([fold_number] * len(test_idx))

    metrics_df = pd.DataFrame({'Fold': fold_numbers, 'MSE': mses, 'RMSE': rmses, 'MAE': maes, 'R2': r2s})
    metrics_df.to_csv(f"{prefix}_vae_reg_metrics_over_folds.csv", index=False)
    metrics_df.assign(Model="VAE_MLP_reg").to_csv(f"{prefix}_VAE_MLP_reg_outer_fold_metrics.csv", index=False)
    print(metrics_df)

    order = np.argsort(np.asarray(y_test_indices))
    y_oof = np.asarray(y_tests, dtype=float)[order]
    y_pred_oof = np.asarray(y_preds, dtype=float)[order]
    folds_oof = np.asarray(y_test_folds, dtype=int)[order]
    oof_mse = mean_squared_error(y_oof, y_pred_oof)
    oof_rmse = np.sqrt(oof_mse)
    oof_mae = mean_absolute_error(y_oof, y_pred_oof)
    oof_r2 = r2_score(y_oof, y_pred_oof)
    pd.DataFrame(
        [{"Model": "VAE_reg", "MSE": oof_mse, "RMSE": oof_rmse, "MAE": oof_mae, "R2": oof_r2, "MetricScope": "Pooled outer-fold out-of-fold predictions"}]
    ).to_csv(f"{prefix}_vae_reg_final_validation_metrics.csv", index=False)
    pd.DataFrame(
        {
            "SampleID": sample_ids,
            "Original Label": y_oof,
            "Predicted Label": y_pred_oof,
            "Fold": folds_oof,
        }
    ).to_csv(f"{prefix}_vae_reg_predictions.csv", index=False)

    plot_regression_nested_cv(metrics_df, "VAE Regression Nested CV", f"{prefix}_vae_reg_metrics_over_folds.png")

    # Average metrics bar chart
    mean_mse, mean_rmse, mean_mae, mean_r2 = np.mean(mses), np.mean(rmses), np.mean(maes), np.mean(r2s)
    plot_metric_bars(["RMSE", "MAE", "R2"], [mean_rmse, mean_mae, mean_r2], "VAE Regression Evaluation Metrics", f"{prefix}_vae_reg_average_metrics.png")

    # Actual vs Predicted (outer test predictions)
    plot_observed_predicted(y_oof, y_pred_oof, "VAE Regression: Observed vs Predicted", f"{prefix}_vae_reg_predictions.png")

    # Residuals
    plot_residuals(y_oof, y_pred_oof, "VAE Regression: Residuals", f"{prefix}_vae_reg_residuals.png")

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
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
        mses_cv = []
        for tr_idx, va_idx in inner_cv.split(X_raw):
            X_tr_raw, X_va_raw = X_raw.iloc[tr_idx], X_raw.iloc[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]
            inner_preprocessor = make_preprocessor()
            X_tr = inner_preprocessor.fit_transform(X_tr_raw, y_tr)
            X_va = inner_preprocessor.transform(X_va_raw)

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
            model.fit(np.asarray(X_tr), y_tr)
            preds = model.predict(np.asarray(X_va))
            mses_cv.append(mean_squared_error(y_va, preds))
        return float(np.mean(mses_cv))

    study_full = optuna.create_study(direction='minimize',
                                     sampler=optuna.samplers.TPESampler(seed=DEFAULT_RANDOM_STATE))
    with SuppressOutput():
        tuning_reports.append(
            run_adaptive_study(
                study_full,
                full_objective,
                task="regression",
                model="VAE_reg",
                study_scope="final_full_development",
                objective_name="vae_regression_mse",
                direction="minimize",
                tolerance=tolerance_for_objective(
                    "vae_regression_mse",
                    scale_floor=max(float(np.nanvar(y)), np.finfo(float).eps),
                ),
                seed=DEFAULT_RANDOM_STATE,
                study_id="regression-vae-final",
                provenance_path=f"{prefix}_vae_reg_final_optuna_study.json",
            )
        )

    best_full = study_full.best_params
    final_preprocessor = make_preprocessor()
    X_scaled_final = final_preprocessor.fit_transform(X_raw, y)
    write_missingness_qc(prefix + "_vae_reg", X_raw, roles, missingness_folds)

    def make_final_model():
        set_seed(DEFAULT_RANDOM_STATE)
        return VAE_MLP(
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

    final_model = make_final_model()
    final_model.fit(np.asarray(X_scaled_final), y)
    write_training_history(getattr(final_model, "training_history_", []), prefix, "vae_reg")
    generate_training_size_curve(np.asarray(X_scaled_final), y, make_final_model, prefix, "vae_reg", task="regression")
    y_pred_final = final_model.predict(np.asarray(X_scaled_final))

    results_df = pd.DataFrame({'SampleID': sample_ids, 'Actual Value': y, 'Predicted Value': y_pred_final})
    results_df.to_csv(f"{prefix}_vae_reg_full_data_predictions.csv", index=False)

    joblib.dump(final_model, f"{prefix}_vae_reg_model.pkl")
    joblib.dump(final_preprocessor, f"{prefix}_vae_reg_scaler.pkl")
    joblib.dump({"model": final_model, "preprocessor": final_preprocessor, "feature_roles": roles, "final_hyperparameters": best_full, "model_feature_order": list(final_preprocessor.get_feature_names_out()), "scope": "final_full_development_model"}, f"{prefix}_VAE_MLP_reg_final_artifact.pkl")
    joblib.dump((X_raw, y, list(final_preprocessor.get_feature_names_out())), f"{prefix}_vae_reg_data.pkl")
    write_reproducibility_manifest(f"{prefix}_vae_reg_manifest.json", input_path=inp, task="regression", model="VAE_MLP_reg", seed=DEFAULT_RANDOM_STATE, roles=roles, imputation=imputation, final_hyperparameters=best_full, artifacts=[f"{prefix}_vae_reg_predictions.csv", f"{prefix}_vae_reg_model.pkl"], imputation_fallbacks=imputation_fallback_reasons(missingness_folds), tuning_studies=tuning_reports)

if __name__ == "__main__":
    args = parse_arguments()
    set_seed(args.seed)
    prefix = build_output_prefix(args.csv, args.prefix)
    with task_run_context(args.outdir, "regression"):
        vae(args.csv, prefix, args.imputation, args.covariates, args.categorical_covariates)
