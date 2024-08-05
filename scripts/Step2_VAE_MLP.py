import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy, sparse_categorical_crossentropy
from tensorflow.keras.optimizers.legacy import Adam
import shap
import matplotlib.pyplot as plt
import warnings
import sys
import contextlib
import tensorflow as tf
import joblib

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

def vae(inp,prefix):
    # Load and preprocess data
    data = pd.read_csv(inp)
    X = data.drop(columns=['Label']).values
    y = data['Label'].values
    feature_names = data.drop(columns=['Label']).columns  # Use original variable names
    
    le = LabelEncoder()
    y = le.fit_transform(y)
    num_classes = len(np.unique(y))
    
    # Convert y to binary form
    if num_classes == 2:
        y_binarized = y.reshape(-1, 1)
    else:
        y_binarized = pd.get_dummies(y).values
    
    # Define VAE model
    input_dim = X.shape[1]
    latent_dim = 2
    
    class VAE_MLP(BaseEstimator, ClassifierMixin):
        def __init__(self, latent_dim=2, encoder_layers=[64, 32], decoder_layers=[32, 64], mlp_layers=[32, 16], learning_rate=0.001, epochs=50, batch_size=32):
            self.latent_dim = latent_dim
            self.encoder_layers = encoder_layers
            self.decoder_layers = decoder_layers
            self.mlp_layers = mlp_layers
            self.learning_rate = learning_rate
            self.epochs = epochs
            self.batch_size = batch_size
            self.vae = None
            self.encoder = None
            self.decoder = None
            self.mlp_model = None
            self.classes_ = None
    
        def sampling(self, args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], K.int_shape(z_mean)[1]))
            return z_mean + K.exp(0.5 * z_log_var) * epsilon
    
        def create_vae(self):
            inputs = Input(shape=(input_dim,))
            h = inputs
            for units in self.encoder_layers:
                h = Dense(units)(h)
                h = LeakyReLU()(h)
            z_mean = Dense(self.latent_dim)(h)
            z_log_var = Dense(self.latent_dim)(h)
            z = Lambda(self.sampling, output_shape=(self.latent_dim,))([z_mean, z_log_var])
    
            self.encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    
            latent_inputs = Input(shape=(self.latent_dim,))
            h_decoded = latent_inputs
            for units in self.decoder_layers:
                h_decoded = Dense(units)(h_decoded)
                h_decoded = LeakyReLU()(h_decoded)
            outputs = Dense(input_dim)(h_decoded)
    
            self.decoder = Model(latent_inputs, outputs, name='decoder')
    
            outputs = self.decoder(self.encoder(inputs)[2])
            self.vae = Model(inputs, outputs, name='vae_mlp')
    
            # VAE loss
            reconstruction_loss = binary_crossentropy(inputs, outputs) * input_dim
            kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            vae_loss = K.mean(reconstruction_loss + kl_loss)
            self.vae.add_loss(vae_loss)
            self.vae.compile(optimizer=Adam(learning_rate=self.learning_rate))
    
        def create_mlp_model(self):
            mlp_input = Input(shape=(input_dim,))
            x = mlp_input
            for units in self.mlp_layers:
                x = Dense(units)(x)
                x = LeakyReLU()(x)
            if num_classes == 2:
                mlp_output = Dense(1, activation='sigmoid')(x)
                self.mlp_model = Model(mlp_input, mlp_output)
                self.mlp_model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
            else:
                mlp_output = Dense(num_classes, activation='softmax')(x)
                self.mlp_model = Model(mlp_input, mlp_output)
                self.mlp_model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
        def fit(self, X, y):
            self.classes_ = np.unique(y)
            self.create_vae()
            self.vae.fit(X, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
            X_decoded = self.decoder.predict(self.encoder.predict(X)[2])
            self.create_mlp_model()
            self.mlp_model.fit(X_decoded, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
            return self
    
        def predict(self, X):
            X_decoded = self.decoder.predict(self.encoder.predict(X)[2])
            y_pred = self.mlp_model(X_decoded, training=False)
            if num_classes == 2:
                return (y_pred > 0.5).numpy().astype(int).flatten()
            else:
                return np.argmax(y_pred, axis=1)
    
        def predict_proba(self, X):
            X_decoded = self.decoder.predict(self.encoder.predict(X)[2])
            return self.mlp_model(X_decoded, training=False).numpy()
    
    # Define parameter search space
    param_distributions = {
        'latent_dim': [2, 4, 8, 16, 32, 64, 128, 256, 512], 
        'encoder_layers': [[64, 32], [128, 64], [64, 64, 32], [256, 128], [128, 128, 64]],  
        'decoder_layers': [[32, 64], [64, 128], [32, 32, 64], [128, 256], [64, 64, 128]],  
        'mlp_layers': [[32, 16], [64, 32], [128, 64], [256, 128], [64, 64, 32]],  
        'learning_rate': [0.001, 0.01, 0.0001, 0.005, 0.0005],  
        'epochs': [10, 50, 100, 200], 
        'batch_size': [16, 32, 64, 128, 256] 
    }
    
    # Use RandomizedSearchCV for parameter search
    vae_mlp = VAE_MLP()
    random_search = RandomizedSearchCV(vae_mlp, param_distributions, n_iter=10, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), verbose=1, n_jobs=-1)
    with SuppressOutput():
        random_search.fit(X, y)
    
    # Use the best model for five-fold cross-validation
    best_model = random_search.best_estimator_
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    f1_scores = []
    sensitivities = []
    specificities = []
    cm_total = np.zeros((num_classes, num_classes))
    
    # Save data for plotting ROC curves
    roc_data = {
        'fpr': {},
        'tpr': {},
        'roc_auc': {}
    }
    
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro' if num_classes > 2 else 'binary')
    
        # Compute sensitivity and specificity
        cm = confusion_matrix(y_test, y_pred)
        cm_total += cm
        if num_classes == 2:
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            sensitivities.append(sensitivity)
            specificities.append(specificity)
        else:
            sensitivities_per_class = []
            specificities_per_class = []
            for i in range(num_classes):
                tp = cm[i, i]
                fn = np.sum(cm[i, :]) - tp
                fp = np.sum(cm[:, i]) - tp
                tn = np.sum(cm) - tp - fn - fp
                sensitivities_per_class.append(tp / (tp + fn))
                specificities_per_class.append(tn / (tn + fp))
            sensitivities.append(np.mean(sensitivities_per_class))
            specificities.append(np.mean(specificities_per_class))
    
        accuracies.append(accuracy)
        f1_scores.append(f1)
    
    # Average confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_total, display_labels=le.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for VAE')
    plt.savefig(f"{prefix}_vae_confusion_matrix.png")
    plt.close()
    
    # Predict and calculate ROC data
    y_pred_prob = cross_val_predict(best_model, X, y, cv=skf, method='predict_proba')
    y_pred_class = np.argmax(y_pred_prob, axis=1)
    
    fpr = {}
    tpr = {}
    roc_auc = {}
    if num_classes == 2:
        fpr[0], tpr[0], _ = roc_curve(y, y_pred_prob[:, 0])
        roc_auc[0] = auc(fpr[0], tpr[0])
    else:
        for i in range(y_binarized.shape[1]):
            fpr[i], tpr[i], _ = roc_curve(y_binarized[:, i], y_pred_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
    
        fpr["micro"], tpr["micro"], _ = roc_curve(y_binarized.ravel(), y_pred_prob.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    roc_data = {
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc
    }
    np.save(f"{prefix}_vae_roc_data.npy", roc_data)
    joblib.dump(best_model, f"{prefix}_vae_model.pkl")
    joblib.dump((X, y, le), f"{prefix}_vae_data.pkl")
    
    # Calculate average evaluation metrics
    mean_accuracy = np.mean(accuracies)
    mean_f1 = np.mean(f1_scores)
    mean_sensitivity = np.mean(sensitivities)
    mean_specificity = np.mean(specificities)
    
    results_df = pd.DataFrame({
        'Metric': ['Accuracy', 'F1 Score', 'Sensitivity', 'Specificity'],
        'Score': [mean_accuracy, mean_f1, mean_sensitivity, mean_specificity]
    })
    print(results_df)
    
    # Plot evaluation metrics
    metrics = ['Accuracy', 'F1 Score', 'Sensitivity', 'Specificity']
    mean_scores = [mean_accuracy, mean_f1, mean_sensitivity, mean_specificity]
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(metrics, mean_scores)  # Set to blue color
    
    # Add value labels on top of bars
    for bar, score in zip(bars, mean_scores):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, round(score, 3), ha='center', va='bottom', fontsize=12)
    
    plt.title('Evaluation Metrics for VAE')
    plt.ylabel('Score')
    plt.savefig(f"{prefix}_vae_evaluation_metrics.png")
    plt.close()
    
    # Plot ROC curves
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
    plt.savefig(f"{prefix}_vae_roc_curve.png")
    plt.close()
    
    # Calculate SHAP feature importance
    explainer = shap.KernelExplainer(best_model.predict_proba, X)
    shap_values = explainer.shap_values(X)
    
    # Output SHAP values to CSV file
    if num_classes == 2:
        shap_values = np.array(shap_values).reshape(-1, X.shape[1])
        mean_shap_values = np.mean(np.abs(shap_values), axis=0)
    else:
        shap_values = np.array(shap_values).reshape(num_classes, -1, X.shape[1])
        mean_shap_values = np.mean(np.abs(shap_values), axis=1).mean(axis=0)
    
    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean SHAP Value': mean_shap_values
    })
    shap_df.to_csv(f"{prefix}_vae_shap_values.csv", index=False)
    
    print("SHAP values saved to vae_shap_values.csv")
    
    # Plot top 5% SHAP value distribution (box plot)
    top_5_percent = int(0.05 * len(feature_names))
    top_features = shap_df.nlargest(top_5_percent, 'Mean SHAP Value')['Feature']
    top_shap_indices = [list(feature_names).index(feature) for feature in top_features]
    
    # For binary classification, shap_values shape is (n_samples, n_features)
    # For multi-class classification, shap_values shape is (n_classes, n_samples, n_features)
    if num_classes == 2:
        top_shap_values = shap_values[:, top_shap_indices]
    else:
        top_shap_values = shap_values[:, :, top_shap_indices].reshape(-1, len(top_shap_indices))
    
    plt.figure(figsize=(10, 6))
    plt.title('Top 5% SHAP Value Distribution for VAE')
    plt.boxplot(top_shap_values, vert=False, labels=top_features)
    plt.xlabel('SHAP Value')
    plt.savefig(f"{prefix}_vae_shap_value_distribution.png")
    plt.close()
