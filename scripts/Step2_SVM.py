import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC
from sklearn.linear_model import ElasticNet
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import warnings
import sys
import contextlib
import joblib
import optuna
from optuna.samplers import TPESampler
import argparse

# Suppress all warnings except ConvergenceWarning from ElasticNet
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

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

def svm_nested_cv(inp, prefix, use_elasticnet):
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
        cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=1234)

        # Define the objective function for Optuna within the outer fold
        def objective_inner(trial):
            if use_elasticnet:
                # Suggest hyperparameters for ElasticNet
                alpha = trial.suggest_loguniform('alpha', 1e-4, 1e1)
                l1_ratio = trial.suggest_uniform('l1_ratio', 0.0, 1.0)
                
                # Suggest hyperparameters for SVM
                C = trial.suggest_loguniform('C', 1e-2, 1e4)
                gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
                
                # Create pipeline with StandardScaler, ElasticNet feature selection, and SVM
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('feature_selection', SelectFromModel(
                        ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, tol=1e-4, random_state=1234)
                    )),
                    ('svm', SVC(kernel='rbf', C=C, gamma=gamma, probability=True, random_state=1234))
                ])
            else:
                # Suggest hyperparameters for SVM
                C = trial.suggest_loguniform('C', 1e-2, 1e4)
                gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
                
                # Create pipeline with StandardScaler and SVM
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('svm', SVC(kernel='rbf', C=C, gamma=gamma, probability=True, random_state=1234))
                ])
            
            # Perform inner cross-validation
            with SuppressOutput():
                f1_scores = []
                for inner_train_idx, inner_valid_idx in cv_inner.split(X_train_outer, y_train_outer):
                    X_train_inner, X_valid_inner = X_train_outer.iloc[inner_train_idx], X_train_outer.iloc[inner_valid_idx]
                    y_train_inner, y_valid_inner = y_train_outer[inner_train_idx], y_train_outer[inner_valid_idx]
                    pipeline.fit(X_train_inner, y_train_inner)
                    y_pred_inner = pipeline.predict(X_valid_inner)
                    f1 = f1_score(y_valid_inner, y_pred_inner, average='weighted')
                    f1_scores.append(f1)
                return np.mean(f1_scores)

        # Create an Optuna study for the inner fold
        study_inner = optuna.create_study(direction='maximize', sampler=TPESampler(seed=1234))
        study_inner.optimize(objective_inner, n_trials=50, show_progress_bar=False)

        # Best hyperparameters from the inner fold
        best_params_inner = study_inner.best_params

        # Initialize the best model with the best hyperparameters
        if use_elasticnet:
            best_alpha = best_params_inner['alpha']
            best_l1_ratio = best_params_inner['l1_ratio']
            best_C = best_params_inner['C']
            best_gamma = best_params_inner['gamma']
            
            best_model_inner = Pipeline([
                ('scaler', StandardScaler()),
                ('feature_selection', SelectFromModel(
                    ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio, max_iter=10000, tol=1e-4, random_state=1234)
                )),
                ('svm', SVC(kernel='rbf', C=best_C, gamma=best_gamma, probability=True, random_state=1234))
            ])
        else:
            best_C = best_params_inner['C']
            best_gamma = best_params_inner['gamma']
            
            best_model_inner = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(kernel='rbf', C=best_C, gamma=best_gamma, probability=True, random_state=1234))
            ])

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
    plt.savefig(f"{prefix}_svm_nested_cv_f1_auc.png", dpi=300)
    plt.close()

    print("Nested cross-validation completed.")
    print(f"Average F1 Score: {np.mean(outer_f1_scores):.4f} ± {np.std(outer_f1_scores):.4f}")
    print(f"Average AUC: {np.mean(outer_auc_scores):.4f} ± {np.std(outer_auc_scores):.4f}")

    # After nested CV, perform hyperparameter tuning on the entire dataset
    print("Starting hyperparameter tuning on the entire dataset...")

    # Define the objective function for Optuna on the entire dataset
    def objective_full(trial):
        if use_elasticnet:
            # Suggest hyperparameters for ElasticNet
            alpha = trial.suggest_loguniform('alpha', 1e-4, 1e1)
            l1_ratio = trial.suggest_uniform('l1_ratio', 0.0, 1.0)
            
            # Suggest hyperparameters for SVM
            C = trial.suggest_loguniform('C', 1e-2, 1e4)
            gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
            
            # Create pipeline with StandardScaler, ElasticNet feature selection, and SVM
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('feature_selection', SelectFromModel(
                    ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, tol=1e-4, random_state=1234)
                )),
                ('svm', SVC(kernel='rbf', C=C, gamma=gamma, probability=True, random_state=1234))
            ])
        else:
            # Suggest hyperparameters for SVM
            C = trial.suggest_loguniform('C', 1e-2, 1e4)
            gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
            
            # Create pipeline with StandardScaler and SVM
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(kernel='rbf', C=C, gamma=gamma, probability=True, random_state=1234))
            ])
        
        # Perform cross-validation
        with SuppressOutput():
            f1_scores = []
            for train_idx_full, valid_idx_full in cv_outer.split(X, y_encoded):
                X_train_full, X_valid_full = X.iloc[train_idx_full], X.iloc[valid_idx_full]
                y_train_full, y_valid_full = y_encoded[train_idx_full], y_encoded[valid_idx_full]
                pipeline.fit(X_train_full, y_train_full)
                y_pred_full = pipeline.predict(X_valid_full)
                f1 = f1_score(y_valid_full, y_pred_full, average='weighted')
                f1_scores.append(f1)
            return np.mean(f1_scores)

    # Create an Optuna study for the entire dataset
    study_full = optuna.create_study(direction='maximize', sampler=TPESampler(seed=1234))
    study_full.optimize(objective_full, n_trials=50, show_progress_bar=True)

    # Best hyperparameters from the entire dataset
    best_params_full = study_full.best_params
    print(f"Best parameters for SVM: {best_params_full}")

    # Initialize the best model with the best hyperparameters
    if use_elasticnet:
        best_alpha_full = best_params_full['alpha']
        best_l1_ratio_full = best_params_full['l1_ratio']
        best_C_full = best_params_full['C']
        best_gamma_full = best_params_full['gamma']
        
        best_model = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectFromModel(
                ElasticNet(alpha=best_alpha_full, l1_ratio=best_l1_ratio_full, max_iter=10000, tol=1e-4, random_state=1234)
            )),
            ('svm', SVC(kernel='rbf', C=best_C_full, gamma=best_gamma_full, probability=True, random_state=1234))
        ])
    else:
        best_C_full = best_params_full['C']
        best_gamma_full = best_params_full['gamma']
        
        best_model = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', C=best_C_full, gamma=best_gamma_full, probability=True, random_state=1234))
        ])

    # Fit the model on the entire dataset
    with SuppressOutput():
        best_model.fit(X, y_encoded)

    # Save the best model and data
    joblib.dump(best_model, f"{prefix}_svm_model.pkl")
    joblib.dump((X, y_encoded, le), f"{prefix}_svm_data.pkl")

    # Output the best parameters
    print(f"Best parameters for SVM: {best_params_full}")

    # If ElasticNet is used, save the selected features
    if use_elasticnet:
        # Extract feature selection step
        feature_selection_step = best_model.named_steps['feature_selection']
        selected_features_mask = feature_selection_step.get_support()
        selected_features = X.columns[selected_features_mask]
        selected_features_df = pd.DataFrame(selected_features, columns=['Selected_Features'])
        selected_features_df.to_csv(f"{prefix}_svm_selected_features.csv", index=False)
        print(f"Selected {selected_features_mask.sum()} features saved to {prefix}_svm_selected_features.csv")

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
    plt.title('Confusion Matrix for SVM')
    plt.savefig(f"{prefix}_svm_confusion_matrix.png", dpi=300)
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
    np.save(f"{prefix}_svm_roc_data.npy", roc_data)

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
    plt.title('ROC Curves for SVM')
    plt.legend(loc="lower right")
    plt.savefig(f"{prefix}_svm_roc_curve.png", dpi=300)
    plt.close()

    # Output performance metrics as a bar chart
    metrics = {'Accuracy': acc, 'F1 Score': f1, 'Sensitivity': sensitivity, 'Specificity': specificity}
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    ax = metrics_df.plot(kind='bar', x='Metric', y='Value', legend=False)
    plt.title('Performance Metrics for SVM')
    plt.ylabel('Value')
    plt.ylim(0, 1)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{prefix}_svm_metrics.png", dpi=300)
    plt.close()

    # Save predictions to CSV
    predictions_df = pd.DataFrame({
        'SampleID': sample_ids,
        'Original Label': y,
        'Predicted Label': le.inverse_transform(y_pred_class)
    })
    predictions_df.to_csv(f"{prefix}_svm_predictions.csv", index=False)

    print(f"Predictions saved to {prefix}_svm_predictions.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SVM with Nested Cross-Validation, Optional ElasticNet Feature Selection, and Optuna hyperparameter optimization.')
    parser.add_argument('--i', type=str, required=True, help='Path to the input CSV file.')
    parser.add_argument('--p', type=str, required=True, help='Prefix for output files.')
    parser.add_argument('--use_elasticnet', action='store_true', help='Enable ElasticNet feature selection.')
    args = parser.parse_args()

    # Run the SVM nested cross-validation function
    svm_nested_cv(args.i, args.p, args.use_elasticnet)