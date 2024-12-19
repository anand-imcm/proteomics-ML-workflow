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
import warnings
import sys
import contextlib
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import joblib
from joblib import Parallel, delayed

def parse_arguments():
    parser = argparse.ArgumentParser(description='Script to run regressors')
    parser.add_argument('-i', '--csv', type=str, help='Input CSV file', required=True)
    parser.add_argument('-p', '--prefix', type=str, help='Output prefix')
    return parser.parse_args()

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Call set_seed function
set_seed()

# Suppress all warnings
warnings.filterwarnings('ignore')

# Suppress output
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

# Early Stopping Class
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
            self.counter +=1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False

# Custom Conditional Normalization Layer
class ConditionalNorm(nn.Module):
    def __init__(self, units, use_batchnorm):
        super(ConditionalNorm, self).__init__()
        if use_batchnorm:
            self.norm = nn.BatchNorm1d(units)
        else:
            self.norm = nn.LayerNorm(units)
    
    def forward(self, x):
        return self.norm(x)

# Define the Encoder network for VAE
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

# Define the Decoder network for VAE
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

# Define the VAE model combining Encoder and Decoder
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, encoder_layers, decoder_layers, dropout_rate, use_batchnorm):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim, encoder_layers, dropout_rate, use_batchnorm)
        self.decoder = Decoder(latent_dim, input_dim, decoder_layers, dropout_rate, use_batchnorm)

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        # Clamp log_var for numerical stability
        z_log_var = torch.clamp(z_log_var, min=-10, max=10)
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        z = z_mean + eps * std
        x_recon = self.decoder(z)
        return x_recon, z_mean, z_log_var

# Define the MLP regressor
class MLPRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, mlp_layers, dropout_rate, use_batchnorm):
        super(MLPRegressor, self).__init__()
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

# Define the VAE_MLP regressor combining VAE and MLPRegressor
class VAE_MLP(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim=30, output_dim=1, latent_dim=2, encoder_layers=[64, 32],
                 decoder_layers=[32, 64], mlp_layers=[32, 16], dropout_rate=0.5,
                 early_stopping_patience=10, early_stopping_min_delta=0.0,
                 vae_learning_rate=0.001, mlp_learning_rate=0.001, epochs=50, batch_size=32):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.mlp_layers = mlp_layers
        self.dropout_rate = dropout_rate
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.vae_learning_rate = vae_learning_rate  # Separate learning rate for VAE
        self.mlp_learning_rate = mlp_learning_rate  # Separate learning rate for MLP
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae = None
        self.vae_optimizer = None
        self.mlp_model = None
        self.mlp_optimizer = None

    # Create the VAE model
    def create_vae(self, use_batchnorm):
        self.vae = VAE(self.input_dim, self.latent_dim, self.encoder_layers,
                      self.decoder_layers, self.dropout_rate, use_batchnorm).to(self.device)
        self.vae_optimizer = optim.Adam(self.vae.parameters(), lr=self.vae_learning_rate)

    # Create the MLP regressor
    def create_mlp_model(self, use_batchnorm):
        self.mlp_model = MLPRegressor(self.input_dim, self.output_dim, self.mlp_layers,
                                       self.dropout_rate, use_batchnorm).to(self.device)
        self.mlp_optimizer = optim.Adam(self.mlp_model.parameters(), lr=self.mlp_learning_rate)

    # Fit the model to data with Early Stopping
    def fit(self, X, y):
        # Determine normalization based on batch_size
        use_batchnorm = self.batch_size > 1
        self.create_vae(use_batchnorm)
        self.create_mlp_model(use_batchnorm)

        # Split training data into training and validation for VAE
        X_train_vae, X_val_vae, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_vae_tensor = torch.from_numpy(X_train_vae).float().to(self.device)
        X_val_vae_tensor = torch.from_numpy(X_val_vae).float().to(self.device)
        dataset_train_vae = torch.utils.data.TensorDataset(X_train_vae_tensor)
        dataloader_train_vae = torch.utils.data.DataLoader(dataset_train_vae, batch_size=self.batch_size, shuffle=True, drop_last=True)
        dataset_val_vae = torch.utils.data.TensorDataset(X_val_vae_tensor)
        dataloader_val_vae = torch.utils.data.DataLoader(dataset_val_vae, batch_size=self.batch_size, shuffle=False, drop_last=True)
        
        # Initialize Early Stopping for VAE
        early_stopping_vae = EarlyStopping(patience=self.early_stopping_patience,
                                           min_delta=self.early_stopping_min_delta,
                                           verbose=False)
        
        # Train VAE with Early Stopping
        self.vae.train()
        for epoch in range(self.epochs):
            for data in dataloader_train_vae:
                x_batch = data[0]
                self.vae_optimizer.zero_grad()
                x_recon, z_mean, z_log_var = self.vae(x_batch)
                # Compute VAE loss
                recon_loss = nn.functional.mse_loss(x_recon, x_batch, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
                loss = recon_loss + kl_loss
                loss.backward()
                self.vae_optimizer.step()
            # Validation loss
            self.vae.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data in dataloader_val_vae:
                    x_batch = data[0]
                    x_recon, z_mean, z_log_var = self.vae(x_batch)
                    recon_loss = nn.functional.mse_loss(x_recon, x_batch, reduction='sum')
                    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
                    loss = recon_loss + kl_loss
                    val_loss += loss.item()
            val_loss /= len(X_val_vae)
            # Check Early Stopping
            if early_stopping_vae(val_loss):
                if self.early_stopping_patience > 0 and self.early_stopping_min_delta >= 0:
                    print(f"Early stopping VAE at epoch {epoch+1}")
                break
            self.vae.train()
        
        # Obtain the reconstructed data from the entire training set
        self.vae.eval()
        with torch.no_grad():
            X_encoded = []
            X_tensor_full = torch.from_numpy(X).float().to(self.device)
            dataset_full = torch.utils.data.TensorDataset(X_tensor_full)
            dataloader_full = torch.utils.data.DataLoader(dataset_full, batch_size=self.batch_size, shuffle=False, drop_last=False)
            for data in dataloader_full:
                x_batch = data[0]
                x_recon, _, _ = self.vae(x_batch)
                X_encoded.append(x_recon.cpu())
            X_encoded = torch.cat(X_encoded, dim=0).numpy()
        
        # Split training data into training and validation for MLP
        X_train_mlp, X_val_mlp, y_train_mlp, y_val_mlp = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
        X_train_mlp_tensor = torch.from_numpy(X_train_mlp).float().to(self.device)
        y_train_mlp_tensor = torch.from_numpy(y_train_mlp).float().to(self.device)
        X_val_mlp_tensor = torch.from_numpy(X_val_mlp).float().to(self.device)
        y_val_mlp_tensor = torch.from_numpy(y_val_mlp).float().to(self.device)
        dataset_train_mlp = torch.utils.data.TensorDataset(X_train_mlp_tensor, y_train_mlp_tensor)
        dataloader_train_mlp = torch.utils.data.DataLoader(dataset_train_mlp, batch_size=self.batch_size, shuffle=True, drop_last=True)
        dataset_val_mlp = torch.utils.data.TensorDataset(X_val_mlp_tensor, y_val_mlp_tensor)
        dataloader_val_mlp = torch.utils.data.DataLoader(dataset_val_mlp, batch_size=self.batch_size, shuffle=False, drop_last=True)
        
        # Initialize Early Stopping for MLP
        early_stopping_mlp = EarlyStopping(patience=self.early_stopping_patience,
                                           min_delta=self.early_stopping_min_delta,
                                           verbose=False)
        
        # Train MLP with Early Stopping
        self.mlp_model.train()
        criterion = nn.MSELoss()
        for epoch in range(self.epochs):
            for data in dataloader_train_mlp:
                x_batch, y_batch = data
                self.mlp_optimizer.zero_grad()
                outputs = self.mlp_model(x_batch)
                outputs = outputs.squeeze()
                loss = criterion(outputs, y_batch)
                loss.backward()
                self.mlp_optimizer.step()
            # Validation loss
            self.mlp_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data in dataloader_val_mlp:
                    x_batch, y_batch = data
                    outputs = self.mlp_model(x_batch)
                    outputs = outputs.squeeze()
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
            val_loss /= len(X_val_mlp)
            # Check Early Stopping
            if early_stopping_mlp(val_loss):
                if self.early_stopping_patience > 0 and self.early_stopping_min_delta >= 0:
                    print(f"Early stopping MLP at epoch {epoch+1}")
                break
            self.mlp_model.train()
        return self

    # Predict values
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
                x_recon, _, _ = self.vae(x_batch)
                X_encoded.append(x_recon.cpu())
            X_encoded = torch.cat(X_encoded, dim=0).numpy()
            y_pred = []
            dataset_mlp = torch.utils.data.TensorDataset(torch.from_numpy(X_encoded).float().to(self.device))
            dataloader_mlp = torch.utils.data.DataLoader(dataset_mlp, batch_size=self.batch_size, shuffle=False, drop_last=False)
            for data in dataloader_mlp:
                x_batch = data[0].to(self.device)
                outputs = self.mlp_model(x_batch)
                outputs = outputs.view(-1)  # Ensure outputs is a one-dimensional tensor
                y_pred.append(outputs.cpu())
            y_pred = torch.cat(y_pred, dim=0).numpy()
        return y_pred

def vae(inp, prefix):
    # Load and preprocess data
    data = pd.read_csv(inp)
    sample_ids = data['SampleID']
    X = data.drop(columns=['SampleID', 'Label']).values
    y = data['Label'].values.astype(np.float32)
    feature_names = data.drop(columns=['SampleID', 'Label']).columns  # Use original feature names

    # Bin y into 5 bins for stratification
    y_binned, bins = pd.qcut(y, q=5, labels=False, retbins=True, duplicates='drop')

    # Outer StratifiedKFold
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize lists to hold metrics for each fold
    mses = []
    rmses = []
    maes = []
    r2s = []
    fold_numbers = []

    y_tests = []
    y_preds = []

    # Iterate over each outer fold
    for fold_number, (train_indices, test_indices) in enumerate(outer_cv.split(X, y_binned), 1):
        print(f"Processing fold {fold_number}")
        X_train_outer, X_test_outer = X[train_indices], X[test_indices]
        y_train_outer, y_test_outer = y[train_indices], y[test_indices]
        y_train_binned_outer = y_binned[train_indices]

        # Standardize the data within the outer fold
        scaler_outer = StandardScaler()
        X_train_outer_scaled = scaler_outer.fit_transform(X_train_outer)
        X_test_outer_scaled = scaler_outer.transform(X_test_outer)

        # Perform hyperparameter tuning using Optuna with nested cross-validation
        def objective(trial):
            # Suggest hyperparameters
            latent_dim = trial.suggest_int('latent_dim', 2, 512, log=True)
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1)
            early_stopping_patience = trial.suggest_int('early_stopping_patience', 5, 20)
            early_stopping_min_delta = trial.suggest_float('early_stopping_min_delta', 0.0, 0.1, step=0.01)
            # Suggest number of encoder layers and units
            num_encoder_layers = trial.suggest_int('num_encoder_layers', 1, 5)
            encoder_layers = []
            for i in range(num_encoder_layers):
                units = trial.suggest_int(f'encoder_units_l{i}', 16, 512, log=True)
                encoder_layers.append(units)

            # Suggest number of decoder layers and units
            num_decoder_layers = trial.suggest_int('num_decoder_layers', 1, 5)
            decoder_layers = []
            for i in range(num_decoder_layers):
                units = trial.suggest_int(f'decoder_units_l{i}', 16, 512, log=True)
                decoder_layers.append(units)

            # Suggest number of MLP layers and units
            num_mlp_layers = trial.suggest_int('num_mlp_layers', 1, 5)
            mlp_layers = []
            for i in range(num_mlp_layers):
                units = trial.suggest_int(f'mlp_units_l{i}', 16, 512, log=True)
                mlp_layers.append(units)

            # Suggest learning rates, epochs, and batch size
            vae_learning_rate = trial.suggest_float('vae_learning_rate', 1e-5, 0.05, log=True)
            mlp_learning_rate = trial.suggest_float('mlp_learning_rate', 1e-5, 0.05, log=True)
            epochs = trial.suggest_int('epochs', 10, 200)
            batch_size = trial.suggest_int('batch_size', 2, 256, log=True)  # Minimum batch_size=2 to avoid batch_size=1

            # Inner cross-validation on training data
            inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            y_train_inner_binned = y_train_binned_outer
            mses_inner = []

            for inner_train_indices, inner_val_indices in inner_cv.split(X_train_outer_scaled, y_train_inner_binned):
                X_train_inner, X_val_inner = X_train_outer_scaled[inner_train_indices], X_train_outer_scaled[inner_val_indices]
                y_train_inner, y_val_inner = y_train_outer[inner_train_indices], y_train_outer[inner_val_indices]

                # Standardize the data within the inner fold
                scaler_inner = StandardScaler()
                X_train_inner_scaled = scaler_inner.fit_transform(X_train_inner)
                X_val_inner_scaled = scaler_inner.transform(X_val_inner)

                # Create a model with the suggested hyperparameters
                model = VAE_MLP(
                    input_dim=X.shape[1],
                    output_dim=1,
                    latent_dim=latent_dim,
                    encoder_layers=encoder_layers,
                    decoder_layers=decoder_layers,
                    mlp_layers=mlp_layers,
                    dropout_rate=dropout_rate,
                    early_stopping_patience=early_stopping_patience,
                    early_stopping_min_delta=early_stopping_min_delta,
                    vae_learning_rate=vae_learning_rate,
                    mlp_learning_rate=mlp_learning_rate,
                    epochs=epochs,
                    batch_size=batch_size
                )
                # Fit the model on inner training data
                model.fit(X_train_inner_scaled, y_train_inner)
                # Predict on inner validation data
                y_pred_inner = model.predict(X_val_inner_scaled)
                mse_inner = mean_squared_error(y_val_inner, y_pred_inner)
                mses_inner.append(mse_inner)

            # Return mean MSE over inner folds
            return np.mean(mses_inner)

        # Use Optuna to find the best hyperparameters on this fold
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
        with SuppressOutput():
            study.optimize(objective, n_trials=20)

        best_params = study.best_params

        # Extract hyperparameters from best_params
        latent_dim = best_params['latent_dim']
        dropout_rate = best_params['dropout_rate']
        early_stopping_patience = best_params['early_stopping_patience']
        early_stopping_min_delta = best_params['early_stopping_min_delta']
        encoder_layers = [best_params[f'encoder_units_l{i}'] for i in range(best_params['num_encoder_layers'])]
        decoder_layers = [best_params[f'decoder_units_l{i}'] for i in range(best_params['num_decoder_layers'])]
        mlp_layers = [best_params[f'mlp_units_l{i}'] for i in range(best_params['num_mlp_layers'])]
        vae_learning_rate = best_params['vae_learning_rate']
        mlp_learning_rate = best_params['mlp_learning_rate']
        epochs = best_params['epochs']
        batch_size = best_params['batch_size']

        print(f"Best hyperparameters for fold {fold_number}:")
        for key, value in best_params.items():
            print(f"{key}: {value}")

        # Train the final model on the entire outer training set with best hyperparameters
        final_model = VAE_MLP(
            input_dim=X.shape[1],
            output_dim=1,
            latent_dim=latent_dim,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            mlp_layers=mlp_layers,
            dropout_rate=dropout_rate,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            vae_learning_rate=vae_learning_rate,
            mlp_learning_rate=mlp_learning_rate,
            epochs=epochs,
            batch_size=batch_size
        )
        final_model.fit(X_train_outer_scaled, y_train_outer)
        # Predict on the outer test set
        y_pred_outer = final_model.predict(X_test_outer_scaled)
        # Compute metrics
        mse = mean_squared_error(y_test_outer, y_pred_outer)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_outer, y_pred_outer)
        r2 = r2_score(y_test_outer, y_pred_outer)

        # Append metrics
        mses.append(mse)
        rmses.append(rmse)
        maes.append(mae)
        r2s.append(r2)
        fold_numbers.append(fold_number)

        y_tests.extend(y_test_outer)
        y_preds.extend(y_pred_outer)

    # After all folds are processed, compute mean metrics
    mean_mse = np.mean(mses)
    mean_rmse = np.mean(rmses)
    mean_mae = np.mean(maes)
    mean_r2 = np.mean(r2s)

    metrics_df = pd.DataFrame({
        'Fold': fold_numbers,
        'MSE': mses,
        'RMSE': rmses,
        'MAE': maes,
        'R2': r2s
    })
    metrics_df.to_csv(f"{prefix}_metrics_over_folds.csv", index=False)
    print(metrics_df)

    # Plot line charts showing the metrics over the folds
    plt.figure(figsize=(10, 6))
    plt.plot(fold_numbers, mses, marker='o', label='MSE')
    plt.plot(fold_numbers, rmses, marker='o', label='RMSE')
    plt.plot(fold_numbers, maes, marker='o', label='MAE')
    plt.plot(fold_numbers, r2s, marker='o', label='R2')
    plt.xlabel('Fold')
    plt.ylabel('Metric Score')
    plt.title('Metrics over Folds')
    plt.legend()
    plt.savefig(f"{prefix}_vae_metrics_over_folds.png", dpi=300)
    plt.close()

    # After cross-validation, train the final model on the entire dataset with best hyperparameters
    # Use the hyperparameters from the last fold's best_params
    # Standardize the entire dataset
    scaler_final = StandardScaler()
    X_scaled_final = scaler_final.fit_transform(X)

    final_model = VAE_MLP(
        input_dim=X.shape[1],
        output_dim=1,
        latent_dim=latent_dim,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        mlp_layers=mlp_layers,
        dropout_rate=dropout_rate,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        vae_learning_rate=vae_learning_rate,
        mlp_learning_rate=mlp_learning_rate,
        epochs=epochs,
        batch_size=batch_size
    )
    final_model.fit(X_scaled_final, y)

    # Predict using the final model
    y_pred_final = final_model.predict(X_scaled_final)

    # Create DataFrame with SampleID, original labels, and predicted values
    results_df = pd.DataFrame({
        'SampleID': sample_ids,
        'Actual Value': y,
        'Predicted Value': y_pred_final
    })
    results_df.to_csv(f"{prefix}_vae_predictions.csv", index=False)

    # Plot Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_tests, y_preds, alpha=0.5)
    plt.plot([min(y_tests), max(y_tests)], [min(y_tests), max(y_tests)], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted')
    plt.savefig(f"{prefix}_vae_predictions.png", dpi=300)
    plt.close()

    # Plot Residuals
    residuals = np.array(y_tests) - np.array(y_preds)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_preds, residuals, alpha=0.5)
    plt.hlines(y=0, xmin=min(y_preds), xmax=max(y_preds), linestyles='dashed')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.savefig(f"{prefix}_vae_residuals.png", dpi=300)
    plt.close()

    # Save the final model and scaler
    joblib.dump(final_model, f"{prefix}_vae_model.pkl")
    joblib.dump(scaler_final, f"{prefix}_vae_scaler.pkl")
    joblib.dump((X, y), f"{prefix}_vae_data.pkl")

    # Calculate SHAP feature importance using SHAP Explainer
    try:
        # Set a reasonable background data size to speed up computation
        background_size = 100
        if X.shape[0] > background_size:
            background = X[np.random.choice(X.shape[0], background_size, replace=False)]
        else:
            background = X

        # Initialize SHAP Explainer with scaled background data
        explainer = shap.Explainer(final_model.predict, scaler_final.transform(background))
        shap_values = explainer(scaler_final.transform(X))

        # Mean absolute SHAP values for features
        mean_shap_values = np.abs(shap_values.values).mean(axis=0)
        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'Mean SHAP Value': mean_shap_values
        })
        shap_df.to_csv(f"{prefix}_vae_shap_values.csv", index=False)
        print(f"SHAP values have been saved to {prefix}_vae_shap_values.csv")

        # Plot SHAP summary plot
        shap.summary_plot(shap_values, scaler_final.transform(X), feature_names=feature_names, show=False)
        plt.savefig(f"{prefix}_vae_shap_summary.png", dpi=300)
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
