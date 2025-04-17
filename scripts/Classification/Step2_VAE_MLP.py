import argparse
import random
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
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
from joblib import Parallel, delayed
from sklearn.utils.class_weight import compute_class_weight

def parse_arguments():
    parser = argparse.ArgumentParser(description='Script to run classifiers')
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

# Define the MLP classifier
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, mlp_layers, dropout_rate, use_batchnorm):
        super(MLPClassifier, self).__init__()
        layers = []
        prev_dim = input_dim
        for units in mlp_layers:
            layers.append(nn.Linear(prev_dim, units))
            layers.append(nn.LeakyReLU())
            layers.append(ConditionalNorm(units, use_batchnorm))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = units
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Define the VAE_MLP classifier combining VAE and MLPClassifier
class VAE_MLP(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim=30, num_classes=2, latent_dim=2, encoder_layers=[64, 32],
                 decoder_layers=[32, 64], mlp_layers=[32, 16], dropout_rate=0.5,
                 early_stopping_patience=10, early_stopping_min_delta=0.0,
                 vae_learning_rate=0.001, mlp_learning_rate=0.001, epochs=50, batch_size=32):
        self.input_dim = input_dim
        self.num_classes = num_classes
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
        self.classes_ = None
        self.vae = None
        self.vae_optimizer = None
        self.mlp_model = None
        self.mlp_optimizer = None

    # Create the VAE model
    def create_vae(self, use_batchnorm):
        self.vae = VAE(self.input_dim, self.latent_dim, self.encoder_layers,
                      self.decoder_layers, self.dropout_rate, use_batchnorm).to(self.device)
        self.vae_optimizer = optim.Adam(self.vae.parameters(), lr=self.vae_learning_rate)

    # Create the MLP classifier
    def create_mlp_model(self, use_batchnorm):
        self.mlp_model = MLPClassifier(self.input_dim, self.num_classes, self.mlp_layers,
                                       self.dropout_rate, use_batchnorm).to(self.device)
        self.mlp_optimizer = optim.Adam(self.mlp_model.parameters(), lr=self.mlp_learning_rate)

    # Fit the model to data with Early Stopping
    def fit(self, X, y):
        self.classes_ = np.unique(y)
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
        y_train_mlp_tensor = torch.from_numpy(y_train_mlp).long().to(self.device)
        X_val_mlp_tensor = torch.from_numpy(X_val_mlp).float().to(self.device)
        y_val_mlp_tensor = torch.from_numpy(y_val_mlp).long().to(self.device)
        dataset_train_mlp = torch.utils.data.TensorDataset(X_train_mlp_tensor, y_train_mlp_tensor)
        dataloader_train_mlp = torch.utils.data.DataLoader(dataset_train_mlp, batch_size=self.batch_size, shuffle=True, drop_last=True)
        dataset_val_mlp = torch.utils.data.TensorDataset(X_val_mlp_tensor, y_val_mlp_tensor)
        dataloader_val_mlp = torch.utils.data.DataLoader(dataset_val_mlp, batch_size=self.batch_size, shuffle=False, drop_last=True)
        
        # Initialize Early Stopping for MLP
        early_stopping_mlp = EarlyStopping(patience=self.early_stopping_patience,
                                           min_delta=self.early_stopping_min_delta,
                                           verbose=False)
        
        # Train MLP with Early Stopping
        classes = np.unique(y_train_mlp)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_mlp)
        weights_tensor = torch.tensor(weights, dtype=torch.float).to(self.device)
        
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        
        self.mlp_model.train()
        for epoch in range(self.epochs):
            for data in dataloader_train_mlp:
                x_batch, y_batch = data
                self.mlp_optimizer.zero_grad()
                logits = self.mlp_model(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                self.mlp_optimizer.step()
            # Validation loss
            self.mlp_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data in dataloader_val_mlp:
                    x_batch, y_batch = data
                    logits = self.mlp_model(x_batch)
                    loss = criterion(logits, y_batch)
                    val_loss += loss.item()
            val_loss /= len(X_val_mlp)
            # Check Early Stopping
            if early_stopping_mlp(val_loss):
                if self.early_stopping_patience > 0 and self.early_stopping_min_delta >= 0:
                    print(f"Early stopping MLP at epoch {epoch+1}")
                break
            self.mlp_model.train()
        return self

    # Predict class labels
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
                logits = self.mlp_model(x_batch)
                preds = torch.argmax(logits, dim=1)
                y_pred.append(preds.cpu())
            y_pred = torch.cat(y_pred, dim=0).numpy()
        return y_pred

    # Predict class probabilities
    def predict_proba(self, X):
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
            y_proba = []
            dataset_mlp = torch.utils.data.TensorDataset(torch.from_numpy(X_encoded).float().to(self.device))
            dataloader_mlp = torch.utils.data.DataLoader(dataset_mlp, batch_size=self.batch_size, shuffle=False, drop_last=False)
            for data in dataloader_mlp:
                x_batch = data[0].to(self.device)
                logits = self.mlp_model(x_batch)
                proba = torch.softmax(logits, dim=1).cpu()
                y_proba.append(proba)
            y_proba = torch.cat(y_proba, dim=0).numpy()
        return y_proba

def calculate_shap_for_class_feature(shap_values, class_idx, feature_idx, num_classes, feature_names):
    if num_classes == 2:
        # Binary classification, use positive class SHAP values
        shap_value = shap_values[:, feature_idx, 1]  # Shape: (samples,)
    else:
        # Multi-class classification, use specific class SHAP values
        shap_value = shap_values[:, feature_idx, class_idx]  # Shape: (samples,)
    mean_shap = np.mean(np.abs(shap_value))
    return (feature_names[feature_idx], mean_shap)

def vae(inp, prefix):
    # Load and preprocess data
    data = pd.read_csv(inp)
    sample_ids = data['SampleID']
    X = data.drop(columns=['SampleID', 'Label']).values
    y = data['Label'].values
    feature_names = data.drop(columns=['SampleID', 'Label']).columns  # Use original feature names

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    num_classes = len(np.unique(y))

    # One-hot encode labels
    y_binarized = pd.get_dummies(y).values

    # Define outer cross-validation
    outer_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize lists to store metrics
    accuracies_final = []
    f1_scores_final = []
    auc_scores_final = []
    sensitivities_final = []
    specificities_final = []
    cm_total_final = np.zeros((num_classes, num_classes), dtype=int)

    # Initialize lists to store ROC data
    all_y_test = []
    all_y_pred_proba = []

    # Initialize StandardScaler for final model
    scaler_final = StandardScaler()

    # Iterate over each outer fold
    fold_number = 1
    for train_index, test_index in outer_skf.split(X, y):
        print(f"Starting outer fold {fold_number}...")
        X_train_outer, X_test_outer = X[train_index], X[test_index]
        y_train_outer, y_test_outer = y[train_index], y[test_index]

        # Standardize the data within the outer fold
        scaler_outer = StandardScaler()
        X_train_outer_scaled = scaler_outer.fit_transform(X_train_outer)
        X_test_outer_scaled = scaler_outer.transform(X_test_outer)

        # Define Optuna's objective function for inner CV
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

            # Perform inner cross-validation
            inner_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            accuracies = []

            # Define a function to train and evaluate on a fold
            def train_evaluate_inner_fold(train_idx_inner, val_idx_inner):
                X_train_inner, X_val_inner = X_train_outer_scaled[train_idx_inner], X_train_outer_scaled[val_idx_inner]
                y_train_inner, y_val_inner = y_train_outer[train_idx_inner], y_train_outer[val_idx_inner]
                
                # Standardize the data within the inner fold
                scaler_inner = StandardScaler()
                X_train_inner_scaled = scaler_inner.fit_transform(X_train_inner)
                X_val_inner_scaled = scaler_inner.transform(X_val_inner)
                
                # Create a new model instance for each fold
                fold_model = VAE_MLP(
                    input_dim=X.shape[1],
                    num_classes=num_classes,
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
                fold_model.fit(X_train_inner_scaled, y_train_inner)
                y_pred = fold_model.predict(X_val_inner_scaled)
                accuracy = accuracy_score(y_val_inner, y_pred)
                return accuracy

            # Parallelize the fold training and evaluation
            results = Parallel(n_jobs=-1)(
                delayed(train_evaluate_inner_fold)(train_idx, val_idx) for train_idx, val_idx in inner_skf.split(X_train_outer_scaled, y_train_outer)
            )

            for accuracy in results:
                accuracies.append(accuracy)

            # Return the mean accuracy as the objective to maximize
            return np.mean(accuracies)

        # Use Optuna for hyperparameter optimization with fixed random seed
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
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

        print(f"Best hyperparameters for outer fold {fold_number}:")
        for key, value in best_params.items():
            print(f"{key}: {value}")

        # Train the model on the outer training set with the best hyperparameters
        outer_model = VAE_MLP(
            input_dim=X.shape[1],
            num_classes=num_classes,
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
        outer_model.fit(X_train_outer_scaled, y_train_outer)
        y_pred_outer = outer_model.predict(X_test_outer_scaled)
        y_pred_proba_outer = outer_model.predict_proba(X_test_outer_scaled)

        # Calculate metrics
        accuracy = accuracy_score(y_test_outer, y_pred_outer)
        f1 = f1_score(y_test_outer, y_pred_outer, average='macro' if num_classes > 2 else 'binary')
        
        if num_classes == 2:
            auc_score = roc_auc_score(y_test_outer, y_pred_proba_outer[:, 1])
        else:
            auc_score = roc_auc_score(y_test_outer, y_pred_proba_outer, multi_class='ovr')

        cm = confusion_matrix(y_test_outer, y_pred_outer, labels=np.arange(num_classes))

        accuracies_final.append(accuracy)
        f1_scores_final.append(f1)
        auc_scores_final.append(auc_score)
        cm_total_final += cm

        if num_classes == 2:
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                sensitivities_final.append(sensitivity)
                specificities_final.append(specificity)
            else:
                # Handle cases where some classes might not appear
                sensitivities_final.append(0)
                specificities_final.append(0)
        else:
            sensitivities_per_class = []
            specificities_per_class = []
            for i in range(num_classes):
                tp = cm[i, i]
                fn = np.sum(cm[i, :]) - tp
                fp = np.sum(cm[:, i]) - tp
                tn = np.sum(cm) - tp - fn - fp
                sensitivities_per_class.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
                specificities_per_class.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
            sensitivities_final.append(np.mean(sensitivities_per_class))
            specificities_final.append(np.mean(specificities_per_class))

        # Store ROC data
        all_y_test.extend(y_test_outer)
        all_y_pred_proba.append(y_pred_proba_outer)

        print(f"Completed outer fold {fold_number}")
        fold_number +=1

    # Average confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_total_final, display_labels=le.classes_)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')  # Set values_format='d' to avoid scientific notation
    plt.title('Confusion Matrix for VAE')
    plt.savefig(f"{prefix}_vae_confusion_matrix.png", dpi=300)
    plt.close()

    # Aggregate ROC data
    all_y_test = np.array(all_y_test)
    all_y_pred_proba = np.vstack(all_y_pred_proba)

    fpr = {}
    tpr = {}
    roc_auc = {}
    if num_classes == 2:
        # Binary classification
        fpr[0], tpr[0], _ = roc_curve(all_y_test, all_y_pred_proba[:, 1])
        roc_auc[0] = auc(fpr[0], tpr[0])
    else:
        # Multi-class classification
        for i in range(y_binarized.shape[1]):
            fpr[i], tpr[i], _ = roc_curve(y_binarized[:, i], all_y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr["micro"], tpr["micro"], _ = roc_curve(y_binarized.ravel(), all_y_pred_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    roc_data = {
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc
    }
    np.save(f"{prefix}_vae_roc_data.npy", roc_data)

    # Create DataFrame with SampleID, original labels, and predicted labels
    # Since predictions are across folds, collect them
    # To align SampleIDs with predictions, sort indices
    sorted_indices = np.argsort(np.concatenate([test_index for _, test_index in outer_skf.split(X, y)]))
    sorted_sample_ids = sample_ids.iloc[sorted_indices].values
    sorted_y_test = all_y_test[sorted_indices]
    sorted_y_pred_proba = all_y_pred_proba[sorted_indices]
    sorted_y_pred = sorted_y_pred_proba.argmax(axis=1)
    sorted_y_pred_labels = le.inverse_transform(sorted_y_pred)

    results_df = pd.DataFrame({
        'SampleID': sorted_sample_ids,
        'Original Label': le.inverse_transform(sorted_y_test),
        'Predicted Label': sorted_y_pred_labels
    })
    results_df.to_csv(f"{prefix}_vae_predictions.csv", index=False)

    # Calculate average evaluation metrics
    mean_accuracy = np.mean(accuracies_final)
    mean_f1 = np.mean(f1_scores_final)
    mean_auc = np.mean(auc_scores_final)
    mean_sensitivity = np.mean(sensitivities_final)
    mean_specificity = np.mean(specificities_final)

    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'F1 Score', 'AUC', 'Sensitivity', 'Specificity'],
        'Score': [mean_accuracy, mean_f1, mean_auc, mean_sensitivity, mean_specificity]
    })
    print(metrics_df)
    
    sns.set(style="whitegrid")

    plt.figure(figsize=(10, 6))
    barplot = sns.barplot(x='Metric', y='Score', data=metrics_df)
    
    for p in barplot.patches:
        height = p.get_height()
        barplot.annotate(f'{height:.2f}',
                         (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='bottom',
                         fontsize=11, color='black', xytext=(0, 5),
                         textcoords='offset points')
    
    plt.title('Average Evaluation Metrics for VAE')
    plt.ylabel('Score')
    plt.ylim(0, 1.05)  
    plt.savefig(f"{prefix}_vae_metrics.png", dpi=300)
    plt.close()

    # Plot evaluation metrics line chart
    # folds = np.arange(1, len(f1_scores_final) + 1)
    # plt.figure(figsize=(12, 8))
    # plt.plot(folds, f1_scores_final, marker='o', label='F1 Score')
    # plt.plot(folds, auc_scores_final, marker='s', label='AUC')
    # plt.title('F1 Score and AUC Across Folds for VAE')
    # plt.xlabel('Fold Number')
    # plt.ylabel('Score')
    # plt.xticks(folds)
    # plt.ylim(0, 1.05)
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(f"{prefix}_vae_nested_cv_f1_auc.png", dpi=300)
    # plt.close()

    # Plot overall ROC curves
    plt.figure(figsize=(10, 8))
    if num_classes == 2:
        plt.plot(fpr[0], tpr[0], label=f'ROC curve (AUC = {roc_auc[0]:.2f})')
    else:
        for i in range(y_binarized.shape[1]):
            plt.plot(fpr[i], tpr[i], label=f'Class {le.inverse_transform([i])[0]} (AUC = {roc_auc[i]:.2f})')
        plt.plot(fpr["micro"], tpr["micro"], label=f'Overall (AUC = {roc_auc["micro"]:.2f})', linestyle='--')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for VAE')
    plt.legend(loc="lower right")
    plt.savefig(f"{prefix}_vae_roc_curve.png", dpi=300)
    plt.close()

    # Train final model on the entire dataset with the best hyperparameters from the last outer fold
    # Fit scaler_final on the entire dataset
    scaler_final = StandardScaler()
    X_scaled_final = scaler_final.fit_transform(X)

    final_model = VAE_MLP(
        input_dim=X.shape[1],
        num_classes=num_classes,
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

    # Predict probabilities using the final model
    y_pred_prob_final = final_model.predict_proba(X_scaled_final)

    # Compute ROC curves and AUC
    fpr_final = {}
    tpr_final = {}
    roc_auc_final = {}
    if num_classes == 2:
        # Binary classification
        fpr_final[0], tpr_final[0], _ = roc_curve(y_binarized[:,1], y_pred_prob_final[:, 1])
        roc_auc_final[0] = auc(fpr_final[0], tpr_final[0])
    else:
        # Multi-class classification
        for i in range(y_binarized.shape[1]):
            fpr_final[i], tpr_final[i], _ = roc_curve(y_binarized[:, i], y_pred_prob_final[:, i])
            roc_auc_final[i] = auc(fpr_final[i], tpr_final[i])

        fpr_final["micro"], tpr_final["micro"], _ = roc_curve(y_binarized.ravel(), y_pred_prob_final.ravel())
        roc_auc_final["micro"] = auc(fpr_final["micro"], tpr_final["micro"])

    roc_data_final = {
        'fpr': fpr_final,
        'tpr': tpr_final,
        'roc_auc': roc_auc_final
    }
    np.save(f"{prefix}_vae_roc_data.npy", roc_data_final)

    # Save the final model, scaler, and data
    joblib.dump(final_model, f"{prefix}_vae_model.pkl")
    joblib.dump(scaler_final, f"{prefix}_vae_scaler.pkl")
    joblib.dump((X, y, le), f"{prefix}_vae_data.pkl")

    # Create DataFrame with SampleID, original labels, and predicted labels
    y_pred_final = final_model.predict(X_scaled_final)
    results_final_df = pd.DataFrame({
        'SampleID': sample_ids,
        'Original Label': le.inverse_transform(y),
        'Predicted Label': le.inverse_transform(y_pred_final)
    })
    results_final_df.to_csv(f"{prefix}_vae_predictions.csv", index=False)

    # Calculate average evaluation metrics (already calculated above)
    mean_accuracy_final = mean_accuracy
    mean_f1_final = mean_f1
    mean_auc_final = mean_auc
    mean_sensitivity_final = mean_sensitivity
    mean_specificity_final = mean_specificity

    metrics_final_df = pd.DataFrame({
        'Metric': ['Accuracy', 'F1 Score', 'AUC', 'Sensitivity', 'Specificity'],
        'Score': [mean_accuracy_final, mean_f1_final, mean_auc_final, mean_sensitivity_final, mean_specificity_final]
    })
    print(metrics_final_df)

    # Plot evaluation metrics line chart
    folds = np.arange(1, len(f1_scores_final) + 1)
    plt.figure(figsize=(12, 8))
    plt.plot(folds, f1_scores_final, marker='o', label='F1 Score')
    plt.plot(folds, auc_scores_final, marker='s', label='AUC')
    plt.title('F1 Score and AUC Across Folds for VAE')
    plt.xlabel('Fold Number')
    plt.ylabel('Score')
    plt.xticks(folds)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{prefix}_vae_nested_cv_f1_auc.png", dpi=300)
    plt.close()

    # Plot overall ROC curves
    plt.figure(figsize=(10, 8))
    if num_classes == 2:
        plt.plot(fpr_final[0], tpr_final[0], label=f'ROC curve (AUC = {roc_auc_final[0]:.2f})')
    else:
        for i in range(y_binarized.shape[1]):
            plt.plot(fpr_final[i], tpr_final[i], label=f'Class {le.inverse_transform([i])[0]} (AUC = {roc_auc_final[i]:.2f})')
        plt.plot(fpr_final["micro"], tpr_final["micro"], label=f'Overall (AUC = {roc_auc_final["micro"]:.2f})', linestyle='--')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for VAE')
    plt.legend(loc="lower right")
    plt.savefig(f"{prefix}_vae_roc_curve.png", dpi=300)
    plt.close()

    # Calculate SHAP feature importance using PermutationExplainer with parallelization
    try:
        # Set a reasonable background data size to speed up computation
        background_size = 100
        if X.shape[0] > background_size:
            background = X[np.random.choice(X.shape[0], background_size, replace=False)]
        else:
            background = X

        # Initialize PermutationExplainer with all available CPU cores
        explainer = shap.PermutationExplainer(final_model.predict_proba, scaler_final.transform(background), n_repeats=10, random_state=42, n_jobs=-1)
        shap_values = explainer.shap_values(scaler_final.transform(X))  # Shape: (classes, samples, features) or (samples, features) depending on SHAP version

        # Ensure shap_values is in (samples, features, classes) format
        if isinstance(shap_values, list):
            # For multi-class, shap_values is a list of arrays, one per class
            shap_values = np.stack(shap_values, axis=-1)  # Now (samples, features, classes)

        # Now shap_values should be (samples, features, num_classes)

        if num_classes == 2:
            # Binary classification
            # Use the SHAP values for the positive class
            shap_class = 1
            shap_values_binary = shap_values[:, :, shap_class]
            # Compute mean absolute SHAP values per feature
            mean_shap_across_classes = np.mean(np.abs(shap_values_binary), axis=0)
            shap_df = pd.DataFrame({
                'Feature': feature_names,
                'Mean SHAP Value': mean_shap_across_classes
            })
            shap_df.to_csv(f"{prefix}_vae_shap_values.csv", index=False)
            print(f"SHAP values have been saved to {prefix}_vae_shap_values.csv")
        else:
            # Multi-class classification
            # Compute mean absolute SHAP values per feature per class
            mean_shap_values_per_class = {}
            for class_idx in range(num_classes):
                class_name = le.inverse_transform([class_idx])[0]
                # shap_values[:, :, class_idx] is (samples, features)
                mean_shap = np.mean(np.abs(shap_values[:, :, class_idx]), axis=0)
                mean_shap_values_per_class[class_name] = mean_shap

            # Create DataFrame for per-class SHAP values
            shap_per_class_df = pd.DataFrame(mean_shap_values_per_class, index=feature_names)
            shap_per_class_df.index.name = 'Feature'
            shap_per_class_df.reset_index(inplace=True)
            shap_per_class_df.to_csv(f"{prefix}_vae_shap_values.csv", index=False)

            # Also, compute overall mean SHAP values across classes
            mean_shap_across_classes = np.mean([mean_shap_values_per_class[class_name] for class_name in mean_shap_values_per_class], axis=0)
            shap_df = pd.DataFrame({
                'Feature': feature_names,
                'Mean SHAP Value': mean_shap_across_classes
            })
            shap_df.to_csv(f"{prefix}_vae_shap_values_mean.csv", index=False)

            print(f"SHAP values have been saved to {prefix}_vae_shap_values_mean.csv and {prefix}_vae_shap_values.csv")

    except Exception as e:
        print(f"SHAP computation failed: {e}")
        print("Proceeding without SHAP analysis.")

if __name__ == "__main__":
    args = parse_arguments()
    prefix = Path(args.csv).stem
    if args.prefix:
        prefix = args.prefix
    vae(args.csv, prefix)
