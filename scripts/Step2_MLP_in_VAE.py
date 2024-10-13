import argparse
import random
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
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
        self.vae_learning_rate = vae_learning_rate
        self.mlp_learning_rate = mlp_learning_rate
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
        self.mlp_model = MLPClassifier(self.latent_dim, self.num_classes, self.mlp_layers,
                                       self.dropout_rate, use_batchnorm).to(self.device)
        self.mlp_optimizer = optim.Adam(self.mlp_model.parameters(), lr=self.mlp_learning_rate)

    # Fit the model to data with Early Stopping
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        # Determine normalization based on batch_size
        use_batchnorm = self.batch_size > 1
        self.create_vae(use_batchnorm)
        # Split training data into training and validation for VAE
        X_train_vae, X_val_vae, y_train_vae, y_val_vae = train_test_split(X, y, test_size=0.2, random_state=42)
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
        
        # Train MLP classifier with Early Stopping
        self.create_mlp_model(use_batchnorm)
        y_tensor = torch.from_numpy(y).long().to(self.device)
        dataset_mlp = torch.utils.data.TensorDataset(torch.from_numpy(X_encoded).float().to(self.device), y_tensor)
        # Split into training and validation
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
        
        self.mlp_model.train()
        criterion = nn.CrossEntropyLoss()
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
            # Check Early Stopping for MLP
            if early_stopping_mlp(val_loss):
                if self.early_stopping_patience > 0 and self.early_stopping_min_delta >= 0:
                    print(f"Early stopping for MLP at epoch {epoch+1}")
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
                _, z_mean, _ = self.vae(x_batch)  # Use z_mean as the encoded latent representation
                X_encoded.append(z_mean.cpu())
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
                _, z_mean, _ = self.vae(x_batch)  # Use z_mean as the encoded latent representation
                X_encoded.append(z_mean.cpu())
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
        # Multiclass classification, use specific class SHAP values
        shap_value = shap_values[:, feature_idx, class_idx]  # Shape: (samples,)
    mean_shap = np.mean(np.abs(shap_value))
    return (feature_names[feature_idx], mean_shap)

def vae(inp, prefix):
    # Load and preprocess data
    data = pd.read_csv(inp)
    sample_ids = data['SampleID'].values
    X = data.drop(columns=['SampleID', 'Label']).values
    y = data['Label'].values
    feature_names = data.drop(columns=['SampleID', 'Label']).columns.tolist()

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    num_classes = len(np.unique(y))

    # One-hot encode labels
    y_binarized = pd.get_dummies(y).values

    # Define Optuna's objective function
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
        
        # Perform 5-fold cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        f1_scores = []

        def evaluate_fold(train_idx, val_idx):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            # Each fold uses a separate instance to prevent shared state issues
            fold_model = VAE_MLP(
                input_dim=model.input_dim,
                num_classes=model.num_classes,
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
            fold_model.fit(X_train, y_train)
            y_pred = fold_model.predict(X_val)
            if num_classes > 2:
                f1 = f1_score(y_val, y_pred, average='macro')
            else:
                f1 = f1_score(y_val, y_pred, average='binary')
            return f1

        # Parallelize the evaluation of each fold
        f1_scores = Parallel(n_jobs=-1)(
            delayed(evaluate_fold)(train_idx, val_idx) for train_idx, val_idx in skf.split(X, y)
        )

        return np.mean(f1_scores)

    # Use Optuna for hyperparameter optimization with parallelization
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
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

    # Print best hyperparameters
    print("Best hyperparameters found by Optuna:")
    for key, value in best_params.items():
        print(f"{key}: {value}")

    # Define a function to create a new model instance with best hyperparameters
    def create_best_model():
        return VAE_MLP(
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

    # Perform 5-fold cross-validation with separate model instances
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    f1_scores_final = []
    sensitivities = []
    specificities = []
    cm_total = np.zeros((num_classes, num_classes), dtype=float)  # Initialize as float to avoid casting issues

    # Lists to collect predictions and probabilities
    all_X = []
    all_y_true = []
    all_y_pred = []
    all_y_pred_proba = []

    def evaluate_fold_final(train_idx, test_idx, fold_num):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Create a new model instance for each fold
        fold_model = create_best_model()
        fold_model.fit(X_train, y_train)
        y_pred = fold_model.predict(X_test)
        y_pred_proba = fold_model.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        if num_classes > 2:
            f1 = f1_score(y_test, y_pred, average='macro')
        else:
            f1 = f1_score(y_test, y_pred, average='binary')

        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=np.arange(num_classes))
        # Ensure cm_sum shape consistency
        cm_sum = np.zeros((num_classes, num_classes), dtype=float)
        cm_sum[:cm.shape[0], :cm.shape[1]] = cm

        # Compute sensitivity and specificity
        if num_classes == 2:
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            else:
                # Handle cases where one class is missing in predictions
                sensitivity = 0
                specificity = 0
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
            sensitivity = np.mean(sensitivities_per_class)
            specificity = np.mean(specificities_per_class)

        # Collect metrics
        metrics = {
            'accuracy': accuracy,
            'f1': f1,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'confusion_matrix': cm_sum
        }

        # Collect predictions
        predictions = {
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

        print(f"Fold {fold_num}: Accuracy={accuracy:.4f}, F1 Score={f1:.4f}, Sensitivity={sensitivity:.4f}, Specificity={specificity:.4f}")

        return metrics, predictions

    # Parallelize the evaluation of each fold
    results = Parallel(n_jobs=-1)(
        delayed(evaluate_fold_final)(train_idx, test_idx, fold_num)
        for fold_num, (train_idx, test_idx) in enumerate(skf.split(X, y), 1)
    )

    # Aggregate results
    for metrics, predictions in results:
        accuracies.append(metrics['accuracy'])
        f1_scores_final.append(metrics['f1'])
        sensitivities.append(metrics['sensitivity'])
        specificities.append(metrics['specificity'])
        cm_total += metrics['confusion_matrix']
        
        all_X.append(predictions['X_test'])
        all_y_true.append(predictions['y_test'])
        all_y_pred.append(predictions['y_pred'])
        all_y_pred_proba.append(predictions['y_pred_proba'])

    # Plot the average confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_total, display_labels=le.classes_)
    disp.plot(cmap=plt.cm.Blues)
    # Manually set ScalarFormatter to avoid scientific notation
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())  # Use ScalarFormatter for x-axis
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())  # Use ScalarFormatter for y-axis
    # Ensure tick labels are not in scientific notation
    ax.ticklabel_format(style='plain', axis='both')
    plt.title('Confusion Matrix for VAE_MLP')
    plt.savefig(f"{prefix}_vaemlp_confusion_matrix.png",dpi=300)
    plt.close()

    # Aggregate all predictions and true labels
    if all_X:
        X_all = np.vstack(all_X)
        y_all_true = np.hstack(all_y_true)
        y_all_pred = np.hstack(all_y_pred)
        y_all_pred_proba = np.vstack(all_y_pred_proba)
    else:
        X_all = np.array([])
        y_all_true = np.array([])
        y_all_pred = np.array([])
        y_all_pred_proba = np.array([])

    if X_all.size > 0:
        # Create a DataFrame with SampleID, Original Label, and Predicted Label
        # To ensure correct mapping, we map predictions back to the original SampleIDs
        # Here, we assume that each sample appears exactly once across all folds
        # If this is not the case, additional handling is required
        # Create a mask for the test indices to map predictions back
        test_indices = []
        for _, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            test_indices.extend(test_idx)
        # Sort test_indices to align with sample_ids
        sorted_indices = np.argsort(test_indices)
        sorted_test_indices = np.array(test_indices)[sorted_indices]
        sorted_pred = y_all_pred[sorted_indices]
        results_df = pd.DataFrame({
            'SampleID': sample_ids[sorted_test_indices],
            'Original Label': data['Label'].values[sorted_test_indices],
            'Predicted Label': le.inverse_transform(sorted_pred)
        })
        results_df.to_csv(f"{prefix}_vaemlp_predictions.csv", index=False)
    else:
        print("No predictions were made. The predictions DataFrame will not be created.")

    # Calculate ROC curves and AUC
    fpr = {}
    tpr = {}
    roc_auc = {}
    if num_classes == 2:
        # Binary classification
        if y_all_pred_proba.shape[1] >= 2:
            fpr[0], tpr[0], _ = roc_curve(y_binarized[:,1], y_all_pred_proba[:, 1])
            roc_auc[0] = auc(fpr[0], tpr[0])
    else:
        # Multiclass classification
        for i in range(y_binarized.shape[1]):
            if y_all_pred_proba.shape[1] > i:
                fpr[i], tpr[i], _ = roc_curve(y_binarized[:, i], y_all_pred_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
        
        if y_all_pred_proba.size > 0:
            fpr["micro"], tpr["micro"], _ = roc_curve(y_binarized.ravel(), y_all_pred_proba.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    roc_data = {
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc
    }
    np.save(f"{prefix}_vaemlp_roc_data.npy", roc_data)
    joblib.dump(study, f"{prefix}_vaemlp_study.pkl")  # Save Optuna study for future reference
    joblib.dump((X, y, le), f"{prefix}_vaemlp_data.pkl", compress=True)

    # Calculate average evaluation metrics
    mean_accuracy = np.mean(accuracies) if accuracies else 0
    mean_f1 = np.mean(f1_scores_final) if f1_scores_final else 0
    mean_sensitivity = np.mean(sensitivities) if sensitivities else 0
    mean_specificity = np.mean(specificities) if specificities else 0

    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'F1 Score', 'Sensitivity', 'Specificity'],
        'Score': [mean_accuracy, mean_f1, mean_sensitivity, mean_specificity]
    })
    print(metrics_df)

    # Plot the evaluation metrics bar chart
    metrics = ['Accuracy', 'F1 Score', 'Sensitivity', 'Specificity']
    mean_scores = [mean_accuracy, mean_f1, mean_sensitivity, mean_specificity]

    plt.figure(figsize=(12, 8))
    bars = plt.bar(metrics, mean_scores)

    # Add value labels on top of each bar
    for bar, score in zip(bars, mean_scores):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, round(score, 3), ha='center', va='bottom', fontsize=12)

    plt.title('Evaluation Metrics for VAE_MLP')
    plt.ylabel('Score')
    plt.savefig(f"{prefix}_vaemlp_metrics.png",dpi=300)
    plt.close()

    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    if num_classes == 2 and 0 in roc_auc:
        plt.plot(fpr[0], tpr[0], label=f'ROC curve (AUC = {roc_auc[0]:.2f})')
    else:
        for i in range(y_binarized.shape[1]):
            if i in roc_auc:
                plt.plot(fpr[i], tpr[i], label=f'Class {le.inverse_transform([i])[0]} (AUC = {roc_auc[i]:.2f})')
        if "micro" in roc_auc:
            plt.plot(fpr["micro"], tpr["micro"], label=f'Overall (AUC = {roc_auc["micro"]:.2f})', linestyle='--')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for VAE_MLP')
    plt.legend(loc="lower right")

    # Ensure ScalarFormatter is used to avoid AttributeError
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())  # Ensure x-axis uses ScalarFormatter
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())  # Ensure y-axis uses ScalarFormatter
    plt.ticklabel_format(style='plain', axis='both')  # Ensure tick labels are in plain format

    plt.savefig(f"{prefix}_vaemlp_roc_curve.png",dpi=300)
    plt.close()

    # Calculate SHAP feature importance using PermutationExplainer with parallel processing
    try:
        # Set a reasonable background data size to speed up computation
        background_size = 100
        background_indices = np.random.choice(X.shape[0], min(background_size, X.shape[0]), replace=False)
        background = X[background_indices]

        # Create and train a final model on the entire dataset for SHAP
        final_model = create_best_model()
        final_model.fit(X, y)

        # Define a prediction probability function based on original features
        def model_predict_proba(X_input):
            return final_model.predict_proba(X_input)

        # Initialize PermutationExplainer with n_jobs=-1 to use all available CPU cores
        explainer = shap.PermutationExplainer(model_predict_proba, background, n_repeats=10, random_state=42, n_jobs=-1)

        # Calculate SHAP values
        shap_values = explainer.shap_values(X)  # Shape: (samples, features, classes) or (classes, samples, features) depending on SHAP version

        # Ensure shap_values is in (samples, features, classes) format
        if isinstance(shap_values, list):
            shap_values = np.stack(shap_values, axis=-1)  # Convert list to numpy array

        # Prepare all (class, feature) combinations
        class_feature_tuples = [(class_idx, feature_idx) for class_idx in range(num_classes) for feature_idx in range(len(feature_names))]

        # Define function to compute mean SHAP value for each (class, feature) combination
        def compute_mean_shap(shap_values, class_idx, feature_idx):
            if num_classes == 2:
                shap_val = shap_values[:, feature_idx, 1]  # Shape: (samples,)
            else:
                shap_val = shap_values[:, feature_idx, class_idx]  # Shape: (samples,)
            mean_shap = np.mean(np.abs(shap_val))
            return (feature_names[feature_idx], mean_shap)

        # Parallelize the computation of mean SHAP values for each (class, feature) combination
        mean_shap_results = Parallel(n_jobs=-1)(
            delayed(compute_mean_shap)(shap_values, class_idx, feature_idx)
            for class_idx, feature_idx in class_feature_tuples
        )

        # Aggregate SHAP values for each feature across all classes
        shap_dict = {}
        for feature, mean_shap in mean_shap_results:
            if feature in shap_dict:
                shap_dict[feature].append(mean_shap)
            else:
                shap_dict[feature] = [mean_shap]

        # Compute the final average SHAP value for each feature across all classes
        final_mean_shap = {feature: np.mean(shaps) for feature, shaps in shap_dict.items()}

        # Convert to DataFrame and save
        shap_df = pd.DataFrame({
            'Feature': list(final_mean_shap.keys()),
            'Mean SHAP Value': list(final_mean_shap.values())
        })
        shap_df.to_csv(f"{prefix}_vaemlp_shap_values.csv", index=False)

        print(f"SHAP values have been saved to {prefix}_vaemlp_shap_values.csv")

    except Exception as e:
        print(f"SHAP computation was skipped due to an error: {e}")
        # Continue execution to ensure other results are still output

if __name__ == "__main__":
    args = parse_arguments()
    prefix = Path(args.csv).stem
    if args.prefix:
        prefix = args.prefix
    vae(args.csv, prefix)