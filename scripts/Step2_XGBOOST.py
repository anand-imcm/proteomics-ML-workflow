import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier
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

def xgboost_nested_cv(inp, prefix):
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

    # Define outer cross-validation strategy
    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

    # Lists to store outer fold metrics
    outer_f1_scores = []
    outer_auc_scores = []

    # Iterate over each outer fold
    fold_idx = 1
    for train_idx, test_idx in cv_outer.split(X, y_encoded):
        print(f"Processing outer fold {fold_idx}...")
        X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
        y_train_outer, y_test_outer = y_encoded[train_idx], y_encoded[test_idx]

        # Define inner cross-validation strategy
        cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

        # Define the objective function for Optuna within the outer fold
        def objective_inner(trial):
            # Suggest hyperparameters
            n_estimators = trial.suggest_int('n_estimators', 10, 300)
            max_depth = trial.suggest_int('max_depth', 3, 15)
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 0.1)
            reg_alpha = trial.suggest_loguniform('reg_alpha', 1e-8, 10.0)
            reg_lambda = trial.suggest_loguniform('reg_lambda', 1e-8, 10.0)
            gamma = trial.suggest_loguniform('gamma', 1e-8, 10.0)
            
            # Initialize XGBClassifier with suggested hyperparameters
            clf = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                gamma=gamma,
                eval_metric='mlogloss',
                use_label_encoder=False,
                verbosity=0,
                random_state=1234
            )
            
            # Perform inner cross-validation
            with SuppressOutput():
                f1_scores = []
                for inner_train_idx, inner_valid_idx in cv_inner.split(X_train_outer, y_train_outer):
                    X_train_inner, X_valid_inner = X_train_outer.iloc[inner_train_idx], X_train_outer.iloc[inner_valid_idx]
                    y_train_inner, y_valid_inner = y_train_outer[inner_train_idx], y_train_outer[inner_valid_idx]
                    clf.fit(X_train_inner, y_train_inner)
                    y_pred_inner = clf.predict(X_valid_inner)
                    f1 = f1_score(y_valid_inner, y_pred_inner, average='weighted')
                    f1_scores.append(f1)
                return np.mean(f1_scores)

        # Create an Optuna study for the inner fold
        study_inner = optuna.create_study(direction='maximize', sampler=TPESampler(seed=1234))
        study_inner.optimize(objective_inner, n_trials=50, show_progress_bar=False)

        # Best hyperparameters from the inner fold
        best_params_inner = study_inner.best_params

        # Initialize the best model with the best hyperparameters
        best_model_inner = XGBClassifier(
            n_estimators=best_params_inner['n_estimators'],
            max_depth=best_params_inner['max_depth'],
            learning_rate=best_params_inner['learning_rate'],
            reg_alpha=best_params_inner['reg_alpha'],
            reg_lambda=best_params_inner['reg_lambda'],
            gamma=best_params_inner['gamma'],
            eval_metric='mlogloss',
            use_label_encoder=False,
            verbosity=0,
            random_state=1234
        )

        # Fit the model on the outer training set
        with SuppressOutput():
            best_model_inner.fit(X_train_outer, y_train_outer)

        # Predict probabilities on the outer test set
        y_pred_prob_outer = best_model_inner.predict_proba(X_test_outer)
        y_pred_class_outer = best_model_inner.predict(X_test_outer)

        # Compute F1 score
        f1_outer = f1_score(y_test_outer, y_pred_class_outer, average='weighted')
        outer_f1_scores.append(f1_outer)

        # Compute AUC
        if num_classes == 2:
            fpr, tpr, _ = roc_curve(y_test_outer, y_pred_prob_outer[:, 1])
            auc_outer = auc(fpr, tpr)
        else:
            # Compute micro-average ROC AUC for multi-class
            fpr, tpr, _ = roc_curve(y_binarized[test_idx].ravel(), y_pred_prob_outer.ravel())
            auc_outer = auc(fpr, tpr)
        outer_auc_scores.append(auc_outer)

        print(f"Fold {fold_idx} - F1 Score: {f1_outer:.4f}, AUC: {auc_outer:.4f}")
        fold_idx += 1

    # Plot and save the F1 and AUC scores per outer fold
    plt.figure(figsize=(10, 6))
    folds = range(1, cv_outer.get_n_splits() + 1)
    plt.plot(folds, outer_f1_scores, marker='o', label='F1 Score')
    plt.plot(folds, outer_auc_scores, marker='s', label='AUC')
    plt.xlabel('Outer Fold')
    plt.ylabel('Score')
    plt.title('F1 and AUC Scores per Outer Fold')
    plt.xticks(folds)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{prefix}_xgboost_nested_cv_f1_auc.png", dpi=300)
    plt.close()

    print("Nested cross-validation completed.")
    print(f"Average F1 Score: {np.mean(outer_f1_scores):.4f} ± {np.std(outer_f1_scores):.4f}")
    print(f"Average AUC: {np.mean(outer_auc_scores):.4f} ± {np.std(outer_auc_scores):.4f}")

    # After nested CV, perform hyperparameter tuning on the entire dataset
    print("Starting hyperparameter tuning on the entire dataset...")

    # Define the objective function for Optuna on the entire dataset
    def objective_full(trial):
        # Suggest hyperparameters
        n_estimators = trial.suggest_int('n_estimators', 10, 300)
        max_depth = trial.suggest_int('max_depth', 3, 15)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 0.1)
        reg_alpha = trial.suggest_loguniform('reg_alpha', 1e-8, 10.0)
        reg_lambda = trial.suggest_loguniform('reg_lambda', 1e-8, 10.0)
        gamma = trial.suggest_loguniform('gamma', 1e-8, 10.0)
        
        # Initialize XGBClassifier with suggested hyperparameters
        clf = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            gamma=gamma,
            eval_metric='mlogloss',
            use_label_encoder=False,
            verbosity=0,
            random_state=1234
        )
        
        # Perform cross-validation
        with SuppressOutput():
            f1_scores = []
            for train_idx_full, valid_idx_full in cv_outer.split(X, y_encoded):
                X_train_full, X_valid_full = X.iloc[train_idx_full], X.iloc[valid_idx_full]
                y_train_full, y_valid_full = y_encoded[train_idx_full], y_encoded[valid_idx_full]
                clf.fit(X_train_full, y_train_full)
                y_pred_full = clf.predict(X_valid_full)
                f1 = f1_score(y_valid_full, y_pred_full, average='weighted')
                f1_scores.append(f1)
            return np.mean(f1_scores)

    # Create an Optuna study for the entire dataset
    study_full = optuna.create_study(direction='maximize', sampler=TPESampler(seed=1234))
    study_full.optimize(objective_full, n_trials=50, show_progress_bar=True)

    # Best hyperparameters from the entire dataset
    best_params_full = study_full.best_params
    print(f"Best parameters for XGBoost: {best_params_full}")

    # Initialize the best model with the best hyperparameters
    best_model = XGBClassifier(
        n_estimators=best_params_full['n_estimators'],
        max_depth=best_params_full['max_depth'],
        learning_rate=best_params_full['learning_rate'],
        reg_alpha=best_params_full['reg_alpha'],
        reg_lambda=best_params_full['reg_lambda'],
        gamma=best_params_full['gamma'],
        eval_metric='mlogloss',
        use_label_encoder=False,
        verbosity=0,
        random_state=1234
    )

    # Fit the model on the entire dataset
    with SuppressOutput():
        best_model.fit(X, y_encoded)

    # Save the best model and data
    joblib.dump(best_model, f"{prefix}_xgboost_model.pkl")
    joblib.dump((X, y_encoded, le), f"{prefix}_xgboost_data.pkl")

    # Output the best parameters
    print(f"Best parameters for XGBoost: {best_params_full}")

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
    plt.title('Confusion Matrix for XGBoost')
    plt.savefig(f"{prefix}_xgboost_confusion_matrix.png", dpi=300)
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
    np.save(f"{prefix}_xgboost_roc_data.npy", roc_data)

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
    plt.title('ROC Curves for XGBoost')
    plt.legend(loc="lower right")
    plt.savefig(f"{prefix}_xgboost_roc_curve.png", dpi=300)
    plt.close()

    # Output performance metrics as a bar chart
    metrics = {'Accuracy': acc, 'F1 Score': f1, 'Sensitivity': sensitivity, 'Specificity': specificity}
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    ax = metrics_df.plot(kind='bar', x='Metric', y='Value', legend=False)
    plt.title('Performance Metrics for XGBoost')
    plt.ylabel('Value')
    plt.ylim(0, 1)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{prefix}_xgboost_metrics.png", dpi=300)
    plt.close()

    # Save predictions to CSV
    predictions_df = pd.DataFrame({
        'SampleID': sample_ids,
        'Original Label': y,
        'Predicted Label': le.inverse_transform(y_pred_class)
    })
    predictions_df.to_csv(f"{prefix}_xgboost_predictions.csv", index=False)

    print(f"Predictions saved to {prefix}_xgboost_predictions.csv")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run XGBoost with Nested Cross-Validation and Optuna hyperparameter optimization.')
    parser.add_argument('--i', type=str, required=True, help='Path to the input CSV file.')
    parser.add_argument('--p', type=str, required=True, help='Prefix for output files.')
    args = parser.parse_args()

    # Run the XGBoost function
    xgboost_nested_cv(args.i, args.p)