import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import warnings
import sys
import contextlib
import joblib
import optuna
from optuna.samplers import TPESampler
from sklearn.base import BaseEstimator, ClassifierMixin

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

# Define custom PLSDA classifier
class PLSDAClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.pls = PLSRegression(n_components=self.n_components)
        self.classes_ = None

    def fit(self, X, y):
        # Save class labels
        self.classes_ = np.unique(y)
        y_onehot = label_binarize(y, classes=self.classes_)
        if y_onehot.ndim == 1:
            y_onehot = np.vstack([1 - y_onehot, y_onehot]).T
        self.pls.fit(X, y_onehot)
        return self

    def predict(self, X):
        y_pred_continuous = self.pls.predict(X)
        if y_pred_continuous.ndim == 2 and y_pred_continuous.shape[1] > 1:
            y_pred = self.classes_[np.argmax(y_pred_continuous, axis=1)]
        else:
            y_pred = (y_pred_continuous >= 0.5).astype(int).ravel()
        return y_pred

    def predict_proba(self, X):
        y_pred_continuous = self.pls.predict(X)
        y_pred_proba = np.maximum(y_pred_continuous, 0)
        if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] > 1:
            y_pred_proba_sum = y_pred_proba.sum(axis=1, keepdims=True)
            y_pred_proba = y_pred_proba / y_pred_proba_sum
        else:
            y_pred_proba_sum = y_pred_proba.sum(axis=1, keepdims=True)
            y_pred_proba = np.hstack([1 - y_pred_proba, y_pred_proba])
        return y_pred_proba

def plsda(inp, prefix):
    # Read data
    data = pd.read_csv(inp)

    # Ensure 'SampleID' and 'Label' columns exist
    if 'SampleID' not in data.columns or 'Label' not in data.columns:
        raise ValueError("Input data must contain 'SampleID' and 'Label' columns.")

    # Extract SampleID
    sample_ids = data['SampleID']

    # Data processing
    X = data.drop(columns=['SampleID', 'Label'])
    y = data['Label']
    
    # Convert target variable to categorical
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_binarized = label_binarize(y_encoded, classes=np.unique(y_encoded))
    num_classes = len(np.unique(y_encoded))

    # Define cross-validation strategy
    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
    
    # Define the objective function for Optuna
    def objective(trial):
        n_components = trial.suggest_int('n_components', 1, min(100, X.shape[1]))
        clf = PLSDAClassifier(n_components=n_components)
        with SuppressOutput():
            scores = []
            for train_idx, valid_idx in cv_outer.split(X, y_encoded):
                X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
                y_train, y_valid = y_encoded[train_idx], y_encoded[valid_idx]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_valid)
                f1 = f1_score(y_valid, y_pred, average='weighted')  # Using F1 score
                scores.append(f1)
            return np.mean(scores)

    # Create an Optuna study
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=1234))
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    # Best hyperparameters
    best_params = study.best_params
    best_n_components = best_params['n_components']

    # Initialize the best model
    best_model = PLSDAClassifier(n_components=best_n_components)

    # Fit the model on the entire dataset
    with SuppressOutput():
        best_model.fit(X, y_encoded)

    # Save the best model and data
    joblib.dump(best_model, f"{prefix}_plsda_model.pkl")
    joblib.dump((X, y_encoded, le), f"{prefix}_plsda_data.pkl")

    # Output the best parameters
    print(f"Best parameters for PLSDA: {best_params}")

    # Predict probabilities
    y_pred_prob = cross_val_predict(best_model, X, y_encoded, cv=cv_outer, method='predict_proba', n_jobs=-1)

    # Convert probabilities to class labels
    if num_classes == 2:
        y_pred_class = (y_pred_prob[:, 1] >= 0.5).astype(int)
    else:
        y_pred_class = np.argmax(y_pred_prob, axis=1)

    # Calculate performance metrics
    acc = accuracy_score(y_encoded, y_pred_class)
    f1 = f1_score(y_encoded, y_pred_class, average='weighted')
    cm = confusion_matrix(y_encoded, y_pred_class)

    # Calculate sensitivity and specificity
    if num_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
    else:
        sensitivity = np.mean(np.diag(cm) / np.sum(cm, axis=1))
        specificity = np.mean(np.diag(cm) / np.sum(cm, axis=0))

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for PLSDA')
    plt.savefig(f"{prefix}_plsda_confusion_matrix.png", dpi=300)
    plt.close()

    # ROC and AUC
    fpr = {}
    tpr = {}
    roc_auc = {}

    # Different handling for binary and multi-class cases
    if num_classes == 2:
        fpr[0], tpr[0], _ = roc_curve(y_binarized[:, 0], y_pred_prob[:, 1])
        roc_auc[0] = auc(fpr[0], tpr[0])
    else:
        for i in range(y_binarized.shape[1]):
            fpr[i], tpr[i], _ = roc_curve(y_binarized[:, i], y_pred_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute overall ROC AUC for multi-class
        fpr["micro"], tpr["micro"], _ = roc_curve(y_binarized.ravel(), y_pred_prob.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Save ROC data
    roc_data = {
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc
    }
    np.save(f"{prefix}_plsda_roc_data.npy", roc_data)

    # Plot and save ROC curve
    plt.figure(figsize=(10, 8))

    if num_classes == 2:
        plt.plot(fpr[0], tpr[0], label=f'AUC = {roc_auc[0]:.2f}')
    else:
        for i in range(len(le.classes_)):
            plt.plot(fpr[i], tpr[i], label=f'{le.inverse_transform([i])[0]} (AUC = {roc_auc[i]:.2f})')
        plt.plot(fpr["micro"], tpr["micro"], label=f'Overall (AUC = {roc_auc["micro"]:.2f})', linestyle='--')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for PLSDA')
    plt.legend(loc="lower right")
    plt.savefig(f"{prefix}_plsda_roc_curve.png", dpi=300)
    plt.close()

    # Output performance metrics as a bar chart
    metrics = {'Accuracy': acc, 'F1 Score': f1, 'Sensitivity': sensitivity, 'Specificity': specificity}
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    ax = metrics_df.plot(kind='bar', x='Metric', y='Value', legend=False)
    plt.title('Performance Metrics for PLSDA')
    plt.ylabel('Value')
    plt.ylim(0, 1)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{prefix}_plsda_metrics.png', dpi=300)
    plt.close()

    # Save predictions to CSV
    predictions_df = pd.DataFrame({
        'SampleID': sample_ids,
        'Original Label': y,
        'Predicted Label': le.inverse_transform(y_pred_class)
    })
    predictions_df.to_csv(f"{prefix}_plsda_predictions.csv", index=False)

    print(f"Predictions saved to {prefix}_plsda_predictions.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run PLSDA with Optuna hyperparameter optimization.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input CSV file.')
    parser.add_argument('--output_prefix', type=str, required=True, help='Prefix for output files.')
    args = parser.parse_args()

    # Run the PLSDA function
    plsda(args.input, args.output_prefix)