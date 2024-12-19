import argparse
import random
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
import matplotlib.ticker as mticker

def parse_arguments():
    parser = argparse.ArgumentParser(description='Script to run regression with VAE and MLP')
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

# Call the seed setting function
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
        layers.append(nn.Linear(prev_dim, 1))  # Output layer for regression
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)  # Output shape (batch_size,)

# Define the VAE_MLP regressor combining VAE and MLPRegressor
class VAE_MLP(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim=30, latent_dim=2, encoder_layers=[64, 32],
                 decoder_layers=[32, 64], mlp_layers=[32, 16], dropout_rate=0.5,
                 early_stopping_patience=10, early_stopping_min_delta=0.0,
                 vae_learning_rate=0.001, mlp_learning_rate=0.001, epochs=50, batch_size=32):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.mlp_layers = mlp_layers
        self.dropout_rate = dropout_rate
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.vae_learning_rate = vae_learning_rate
        self.mlp_learning_rate = mlp_learning_rate
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

    # Create the MLP regression model
    def create_mlp_model(self, use_batchnorm):
        self.mlp_model = MLPRegressor(input_dim=self.latent_dim, mlp_layers=self.mlp_layers,
                                       dropout_rate=self.dropout_rate, use_batchnorm=use_batchnorm).to(self.device)
        self.mlp_optimizer = optim.Adam(self.mlp_model.parameters(), lr=self.mlp_learning_rate)

    # Fit the model to data with Early Stopping
    def fit(self, X, y):
        # Determine normalization based on batch_size
        use_batchnorm = self.batch_size > 1
        self.create_vae(use_batchnorm)
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
            # Check Early Stopping for VAE
            if early_stopping_vae(val_loss):
                if self.early_stopping_patience > 0 and self.early_stopping_min_delta >= 0:
                    print(f"Early stopping for VAE at epoch {epoch+1}")
                break
            self.vae.train()
        
        # Obtain the latent representations from the entire training set
        self.vae.eval()
        with torch.no_grad():
            X_encoded = []
            X_tensor_full = torch.from_numpy(X).float().to(self.device)
            dataset_full = torch.utils.data.TensorDataset(X_tensor_full)
            dataloader_full = torch.utils.data.DataLoader(dataset_full, batch_size=self.batch_size, shuffle=False, drop_last=False)
            for data in dataloader_full:
                x_batch = data[0]
                _, z_mean, _ = self.vae(x_batch)  # Use z_mean as the encoded latent representation
                X_encoded.append(z_mean.cpu())
            X_encoded = torch.cat(X_encoded, dim=0).numpy()
        
        # Train MLP regressor with Early Stopping
        self.create_mlp_model(use_batchnorm)
        y_tensor = torch.from_numpy(y).float().to(self.device)
        dataset_mlp = torch.utils.data.TensorDataset(torch.from_numpy(X_encoded).float().to(self.device), y_tensor)
        # Split into training and validation
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
            # Validation loss
            self.mlp_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data in dataloader_val_mlp:
                    x_batch, y_batch = data
                    preds = self.mlp_model(x_batch)
                    loss = criterion(preds, y_batch)
                    val_loss += loss.item()
            val_loss /= len(X_val_mlp)
            # Check Early Stopping for MLP
            if early_stopping_mlp(val_loss):
                if self.early_stopping_patience > 0 and self.early_stopping_min_delta >= 0:
                    print(f"Early stopping for MLP at epoch {epoch+1}")
                break
            self.mlp_model.train()
        return self

    # Predict continuous values
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
                _, z_mean, _ = self.vae(x_batch)  # Use z_mean as the encoded latent representation
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
    # Calculate mean absolute SHAP values for each feature
    mean_shap = np.mean(np.abs(shap_values), axis=0)
    return list(zip(feature_names, mean_shap))

def vae_regression(inp, prefix):
    # Load and preprocess data
    data = pd.read_csv(inp)
    sample_ids = data['SampleID'].values
    X = data.drop(columns=['SampleID', 'Label']).values
    y = data['Label'].values.astype(np.float32)
    feature_names = data.drop(columns=['SampleID', 'Label']).columns.tolist()

    # Initialize lists to collect metrics and predictions
    mse_scores = []
    rmse_scores = []
    mae_scores = []
    r2_scores = []

    # Lists to collect predictions and true values
    all_X = []
    all_y_true = []
    all_y_pred = []

    # Perform 5-fold outer cross-validation
    outer_kf = KFold(n_splits=5, shuffle=True, random_state=42)

    def evaluate_fold_final(train_idx, test_idx, fold_num):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Define Optuna's objective function for this fold
        def objective(trial):
            # Suggest hyperparameters
            latent_dim = trial.suggest_int('latent_dim', 2, 512, log=True)
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1)
            early_stopping_patience = trial.suggest_int('early_stopping_patience', 5, 20)
            early_stopping_min_delta = trial.suggest_float('early_stopping_min_delta', 0.0, 0.1, step=0.01)
            
            # Suggest encoder layers
            num_encoder_layers = trial.suggest_int('num_encoder_layers', 1, 5)
            encoder_layers = []
            for i in range(num_encoder_layers):
                units = trial.suggest_int(f'encoder_units_l{i}', 16, 512, log=True)
                encoder_layers.append(units)
            
            # Suggest decoder layers
            num_decoder_layers = trial.suggest_int('num_decoder_layers', 1, 5)
            decoder_layers = []
            for i in range(num_decoder_layers):
                units = trial.suggest_int(f'decoder_units_l{i}', 16, 512, log=True)
                decoder_layers.append(units)
            
            # Suggest MLP layers
            num_mlp_layers = trial.suggest_int('num_mlp_layers', 1, 5)
            mlp_layers = []
            for i in range(num_mlp_layers):
                units = trial.suggest_int(f'mlp_units_l{i}', 16, 512, log=True)
                mlp_layers.append(units)
            
            # Suggest learning rates, epochs, and batch size
            vae_learning_rate = trial.suggest_float('vae_learning_rate', 1e-5, 0.05, log=True)
            mlp_learning_rate = trial.suggest_float('mlp_learning_rate', 1e-5, 0.05, log=True)
            epochs = trial.suggest_int('epochs', 10, 200)
            batch_size = trial.suggest_int('batch_size', 2, 256, log=True)  # Set minimum batch_size=2 to avoid batch_size=1
            
            # Create the model
            model = VAE_MLP(
                input_dim=X_train.shape[1],
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
            
            # Perform inner cross-validation on training data
            inner_kf = KFold(n_splits=5, shuffle=True, random_state=fold_num)  # Use fold_num as seed
            rmse_scores = []

            for inner_train_idx, inner_val_idx in inner_kf.split(X_train):
                X_inner_train, X_inner_val = X_train[inner_train_idx], X_train[inner_val_idx]
                y_inner_train, y_inner_val = y_train[inner_train_idx], y_train[inner_val_idx]
                # Each fold uses a separate instance to prevent shared state issues
                fold_model = VAE_MLP(
                    input_dim=model.input_dim,
                    latent_dim=model.latent_dim,
                    encoder_layers=model.encoder_layers,
                    decoder_layers=model.decoder_layers,
                    mlp_layers=model.mlp_layers,
                    dropout_rate=model.dropout_rate,
                    early_stopping_patience=model.early_stopping_patience,
                    early_stopping_min_delta=model.early_stopping_min_delta,
                    vae_learning_rate=model.vae_learning_rate,
                    mlp_learning_rate=model.mlp_learning_rate,
                    epochs=model.epochs,
                    batch_size=model.batch_size
                )
                fold_model.fit(X_inner_train, y_inner_train)
                y_pred = fold_model.predict(X_inner_val)
                rmse = np.sqrt(mean_squared_error(y_inner_val, y_pred))
                rmse_scores.append(rmse)

            return np.mean(rmse_scores)

        # Create Optuna study
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
        with SuppressOutput():
            study.optimize(objective, n_trials=20, n_jobs=1)  # n_jobs=1 to avoid nested parallelism issues

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

        # Create the model with best hyperparameters
        model = VAE_MLP(
            input_dim=X_train.shape[1],
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

        # Fit the model on training data
        model.fit(X_train, y_train)

        # Predict on test data
        y_pred = model.predict(X_test)

        # Compute metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Collect metrics
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

        # Collect predictions
        predictions = {
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'test_idx': test_idx
        }

        print(f"Fold {fold_num}: MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

        return metrics, predictions

    # Parallelize the evaluation of each fold
    results = Parallel(n_jobs=-1)(
        delayed(evaluate_fold_final)(train_idx, test_idx, fold_num)
        for fold_num, (train_idx, test_idx) in enumerate(outer_kf.split(X), 1)
    )

    # Aggregate results
    for metrics, predictions in results:
        mse_scores.append(metrics['mse'])
        rmse_scores.append(metrics['rmse'])
        mae_scores.append(metrics['mae'])
        r2_scores.append(metrics['r2'])
        
        all_X.append(predictions['X_test'])
        all_y_true.append(predictions['y_test'])
        all_y_pred.append(predictions['y_pred'])

    # Aggregate all predictions and true labels
    if all_X:
        X_all = np.vstack(all_X)
        y_all_true = np.hstack(all_y_true)
        y_all_pred = np.hstack(all_y_pred)
        all_test_idx = np.hstack([predictions['test_idx'] for _, predictions in results])
    else:
        print("No predictions were made. The predictions DataFrame will not be created.")
        return

    if X_all.size > 0:
        # Create a DataFrame with SampleID, Original Label, and Predicted Value
        results_df = pd.DataFrame({
            'SampleID': sample_ids[all_test_idx],
            'Original Label': data['Label'].values[all_test_idx],
            'Predicted Value': y_all_pred
        })
        results_df.to_csv(f"{prefix}_vaemlp_predictions.csv", index=False)
    else:
        print("No predictions were made. The predictions DataFrame will not be created.")

    # Calculate average evaluation metrics
    mean_mse = np.mean(mse_scores) if mse_scores else 0
    mean_rmse = np.mean(rmse_scores) if rmse_scores else 0
    mean_mae = np.mean(mae_scores) if mae_scores else 0
    mean_r2 = np.mean(r2_scores) if r2_scores else 0

    metrics_df = pd.DataFrame({
        'Metric': ['MSE', 'RMSE', 'MAE', 'R2'],
        'Score': [mean_mse, mean_rmse, mean_mae, mean_r2]
    })
    print(metrics_df)

    # Plot the evaluation metrics over the folds
    metrics = ['MSE', 'RMSE', 'MAE', 'R2']
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
    plt.savefig(f"{prefix}_vaemlp_metrics_over_folds.png", dpi=300)
    plt.close()

    # Plot average evaluation metrics bar chart
    mean_scores = [mean_mse, mean_rmse, mean_mae, mean_r2]

    plt.figure(figsize=(12, 8))
    bars = plt.bar(metrics, mean_scores)

    # Add value labels on top of each bar
    for bar, score in zip(bars, mean_scores):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, round(score, 3), ha='center', va='bottom', fontsize=12)

    plt.title('Average Evaluation Metrics for VAE_MLP Regression')
    plt.ylabel('Score')
    plt.savefig(f"{prefix}_vaemlp_average_metrics.png", dpi=300)
    plt.close()

    # Plot predictions vs actual values
    plt.figure(figsize=(10, 8))
    plt.scatter(y_all_true, y_all_pred, alpha=0.6)
    plt.plot([y_all_true.min(), y_all_true.max()], [y_all_true.min(), y_all_true.max()], 'k--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual Values')
    plt.savefig(f"{prefix}_vaemlp_predictions_vs_actual.png", dpi=300)
    plt.close()

    # Plot residuals
    residuals = y_all_true - y_all_pred
    plt.figure(figsize=(10, 8))
    plt.scatter(y_all_pred, residuals, alpha=0.6)
    plt.hlines(y=0, xmin=y_all_pred.min(), xmax=y_all_pred.max(), colors='r', linestyles='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.savefig(f"{prefix}_vaemlp_residuals.png", dpi=300)
    plt.close()

    # Calculate SHAP feature importance using PermutationExplainer with parallel processing
    try:
        # Set a reasonable background data size to speed up computation
        background_size = 100
        background_indices = np.random.choice(X.shape[0], min(background_size, X.shape[0]), replace=False)
        background = X[background_indices]

        # Create and train a final model on the entire dataset for SHAP
        # For SHAP, we can perform hyperparameter optimization on the entire dataset
        def objective(trial):
            # Suggest hyperparameters
            latent_dim = trial.suggest_int('latent_dim', 2, 512, log=True)
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1)
            early_stopping_patience = trial.suggest_int('early_stopping_patience', 5, 20)
            early_stopping_min_delta = trial.suggest_float('early_stopping_min_delta', 0.0, 0.1, step=0.01)
            
            # Suggest encoder layers
            num_encoder_layers = trial.suggest_int('num_encoder_layers', 1, 5)
            encoder_layers = []
            for i in range(num_encoder_layers):
                units = trial.suggest_int(f'encoder_units_l{i}', 16, 512, log=True)
                encoder_layers.append(units)
            
            # Suggest decoder layers
            num_decoder_layers = trial.suggest_int('num_decoder_layers', 1, 5)
            decoder_layers = []
            for i in range(num_decoder_layers):
                units = trial.suggest_int(f'decoder_units_l{i}', 16, 512, log=True)
                decoder_layers.append(units)
            
            # Suggest MLP layers
            num_mlp_layers = trial.suggest_int('num_mlp_layers', 1, 5)
            mlp_layers = []
            for i in range(num_mlp_layers):
                units = trial.suggest_int(f'mlp_units_l{i}', 16, 512, log=True)
                mlp_layers.append(units)
            
            # Suggest learning rates, epochs, and batch size
            vae_learning_rate = trial.suggest_float('vae_learning_rate', 1e-5, 0.05, log=True)
            mlp_learning_rate = trial.suggest_float('mlp_learning_rate', 1e-5, 0.05, log=True)
            epochs = trial.suggest_int('epochs', 10, 200)
            batch_size = trial.suggest_int('batch_size', 2, 256, log=True)  # Set minimum batch_size=2 to avoid batch_size=1
            
            # Create the model
            model = VAE_MLP(
                input_dim=X.shape[1],
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
            
            # Perform 5-fold cross-validation
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            rmse_scores = []

            for train_idx, val_idx in kf.split(X):
                X_train_cv, X_val_cv = X[train_idx], X[val_idx]
                y_train_cv, y_val_cv = y[train_idx], y[val_idx]
                # Each fold uses a separate instance to prevent shared state issues
                fold_model = VAE_MLP(
                    input_dim=model.input_dim,
                    latent_dim=model.latent_dim,
                    encoder_layers=model.encoder_layers,
                    decoder_layers=model.decoder_layers,
                    mlp_layers=model.mlp_layers,
                    dropout_rate=model.dropout_rate,
                    early_stopping_patience=model.early_stopping_patience,
                    early_stopping_min_delta=model.early_stopping_min_delta,
                    vae_learning_rate=model.vae_learning_rate,
                    mlp_learning_rate=model.mlp_learning_rate,
                    epochs=model.epochs,
                    batch_size=model.batch_size
                )
                fold_model.fit(X_train_cv, y_train_cv)
                y_pred_cv = fold_model.predict(X_val_cv)
                rmse = np.sqrt(mean_squared_error(y_val_cv, y_pred_cv))
                rmse_scores.append(rmse)

            return np.mean(rmse_scores)

        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
        with SuppressOutput():
            study.optimize(objective, n_trials=20, n_jobs=-1)  # n_jobs=-1 uses all available cores

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

        # Create a model with best hyperparameters
        final_model = VAE_MLP(
            input_dim=X.shape[1],
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

        # Fit the final model on the entire dataset
        final_model.fit(X, y)

        # Define a prediction function based on original features
        def model_predict(X_input):
            return final_model.predict(X_input)

        # Initialize PermutationExplainer with n_jobs=-1 to use all available CPU cores
        explainer = shap.PermutationExplainer(model_predict, background, n_repeats=10, random_state=42, n_jobs=-1)

        # Calculate SHAP values
        shap_values = explainer.shap_values(X)  # Shape: (samples, features)

        # Calculate mean absolute SHAP values for each feature
        mean_shap_values = np.mean(np.abs(shap_values), axis=0)
        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'Mean SHAP Value': mean_shap_values
        })
        shap_df.to_csv(f"{prefix}_vaemlp_shap_values.csv", index=False)

        print(f"SHAP values have been saved to {prefix}_vaemlp_shap_values.csv")

        # Plot SHAP summary plot
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        plt.savefig(f"{prefix}_vaemlp_shap_summary.png", dpi=300)
        plt.close()

        # Plot SHAP bar chart
        shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type='bar', show=False)
        plt.savefig(f"{prefix}_vaemlp_shap_bar.png", dpi=300)
        plt.close()

    except Exception as e:
        print(f"SHAP computation was skipped due to an error: {e}")
        # Continue execution to ensure other results are still output

if __name__ == "__main__":
    args = parse_arguments()
    prefix = Path(args.csv).stem
    if args.prefix:
        prefix = args.prefix
    vae_regression(args.csv, prefix)