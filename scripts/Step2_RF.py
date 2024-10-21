import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import warnings
import sys
import contextlib
import joblib
import optuna
from optuna.samplers import TPESampler

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

def random_forest(inp, prefix):
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
    y_binarized = pd.get_dummies(y_encoded).values
    num_classes = len(np.unique(y_encoded))

    # Define cross-validation strategy
    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
    
    # Define the objective function for Optuna
    def objective(trial):
        # Suggest hyperparameters
        n_estimators = trial.suggest_int('n_estimators', 100, 1000, step=100)
        max_depth = trial.suggest_int('max_depth', 5, 50, step=5)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        
        # Initialize RandomForest with suggested hyperparameters
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, random_state=1234)
        
        # Perform cross-validation
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
    best_n_estimators = best_params['n_estimators']
    best_max_depth = best_params['max_depth']
    best_max_features = best_params['max_features']

    # Initialize the best model
    best_model = RandomForestClassifier(n_estimators=best_n_estimators, max_depth=best_max_depth, max_features=best_max_features, random_state=1234)

    # Fit the model on the entire dataset
    with SuppressOutput():
        best_model.fit(X, y_encoded)

    # Save the best model and data
    joblib.dump(best_model, f"{prefix}_random_forest_model.pkl")
    joblib.dump((X, y_encoded, le), f"{prefix}_random_forest_data.pkl")

    # Output the best parameters
    print(f"Best parameters for Random Forest: {best_params}")

    # Prediction using cross_val_predict
    y_pred_prob = cross_val_predict(best_model, X, y_encoded, cv=cv_outer, method='predict_proba', n_jobs=-1)
    y_pred_class = np.argmax(y_pred_prob, axis=1)

    # Compute metrics
    acc = accuracy_score(y_encoded, y_pred_class)
    f1 = f1_score(y_encoded, y_pred_class, average='weighted')
    cm = confusion_matrix(y_encoded, y_pred_class)

    # Compute sensitivity and specificity
    if num_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        sensitivity = np.mean(np.diag(cm) / np.sum(cm, axis=1))
        specificity = np.mean(np.diag(cm) / np.sum(cm, axis=0))

    # Confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for Random Forest')
    plt.savefig(f"{prefix}_random_forest_confusion_matrix.png", dpi=300)
    plt.close()

    # ROC and AUC
    fpr = {}
    tpr = {}
    roc_auc = {}

    # Different handling for binary and multi-class cases
    if num_classes == 2:
        fpr[0], tpr[0], _ = roc_curve(y_binarized[:, 0], y_pred_prob[:, 0])
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
    np.save(f"{prefix}_random_forest_roc_data.npy", roc_data)

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
    plt.title('ROC Curves for Random Forest')
    plt.legend(loc="lower right")
    plt.savefig(f"{prefix}_random_forest_roc_curve.png", dpi=300)
    plt.close()

    # Output performance metrics as a bar chart
    metrics = {'Accuracy': acc, 'F1 Score': f1, 'Sensitivity': sensitivity, 'Specificity': specificity}
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    ax = metrics_df.plot(kind='bar', x='Metric', y='Value', legend=False)
    plt.title('Performance Metrics for Random Forest')
    plt.ylabel('Value')
    plt.ylim(0, 1)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{prefix}_random_forest_metrics.png", dpi=300)
    plt.close()

    # Save predictions to CSV
    predictions_df = pd.DataFrame({
        'SampleID': sample_ids,
        'Original Label': y,
        'Predicted Label': le.inverse_transform(y_pred_class)
    })
    predictions_df.to_csv(f"{prefix}_random_forest_predictions.csv", index=False)

    print(f"Predictions saved to {prefix}_random_forest_predictions.csv")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run Random Forest with Optuna hyperparameter optimization.')
    parser.add_argument('--i', type=str, required=True, help='Path to the input CSV file.')
    parser.add_argument('--p', type=str, required=True, help='Prefix for output files.')
    args = parser.parse_args()

    # Run the random forest function
    random_forest(args.i, args.p)
    