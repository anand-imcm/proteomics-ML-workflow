import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import warnings
import sys
import contextlib
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

def neural_network(inp,prefix):
    # Read data
    data = pd.read_csv(inp)
    
    # Data processing
    X = data.drop(columns=['Label'])
    y = data['Label']
    
    # Convert target variable to categorical
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_binarized = pd.get_dummies(y_encoded).values
    
    # Model and parameters
    clf = MLPClassifier(activation='relu', max_iter=200000, random_state=1234)
    param_grid = {
        'hidden_layer_sizes': [(i,) for i in range(1, 20)],
        'alpha': np.linspace(0.01, 0.1, 10),
        'learning_rate_init': np.logspace(-4, -2, 5)
    }
    
    # Cross-validation
    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
    cv_inner = StratifiedKFold(n_splits=10, shuffle=True, random_state=1234)
    
    with SuppressOutput():
        # Inner cross-validation to search for best parameters
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=cv_inner, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X, y_encoded)
        best_model = grid_search.best_estimator_
    
    # Save the best model and data
    joblib.dump(best_model, f"{prefix}_neural_network_model.pkl")
    joblib.dump((X, y_encoded, le), f"{prefix}_neural_network_data.pkl")
    
    # Output the best parameters
    print(f"Best parameters for Neural Network: {grid_search.best_params_}")
    
    # Predict
    y_pred_prob = cross_val_predict(best_model, X, y_encoded, cv=cv_outer, method='predict_proba')
    y_pred_class = np.argmax(y_pred_prob, axis=1)
    
    # Compute metrics
    acc = accuracy_score(y_encoded, y_pred_class)
    f1 = f1_score(y_encoded, y_pred_class, average='weighted')
    cm = confusion_matrix(y_encoded, y_pred_class)
    
    # Compute sensitivity and specificity
    num_classes = len(le.classes_)
    if num_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
    else:
        sensitivity = np.mean(np.diag(cm) / np.sum(cm, axis=1))
        specificity = np.mean(np.diag(cm) / np.sum(cm, axis=0))
    
    # Confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for Neural Network')
    plt.savefig(f"{prefix}_neural_network_confusion_matrix.png")
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
    np.save(f"{prefix}_neural_network_roc_data.npy", roc_data)
    
    # Plot and save ROC curve
    plt.figure(figsize=(10, 8))
    
    if num_classes == 2:
        plt.plot(fpr[0], tpr[0], label=f'Class {le.classes_[0]} (AUC = {roc_auc[0]:.2f})')
    else:
        for i in range(len(le.classes_)):
            plt.plot(fpr[i], tpr[i], label=f'{le.inverse_transform([i])[0]} (AUC = {roc_auc[i]:.2f})')
        plt.plot(fpr["micro"], tpr["micro"], label=f'Overall (AUC = {roc_auc["micro"]:.2f})', linestyle='--')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Neural Network')
    plt.legend(loc="lower right")
    plt.savefig(f"{prefix}_neural_network_roc_curve.png")
    plt.close()
    
    # Output performance metrics as a bar chart
    metrics = {'Accuracy': acc, 'F1 Score': f1, 'Sensitivity': sensitivity, 'Specificity': specificity}
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    ax = metrics_df.plot(kind='bar', x='Metric', y='Value', legend=False)
    plt.title('Performance Metrics for Neural Network')
    plt.ylabel('Value')
    plt.ylim(0, 1)
    for i in ax.containers:
        ax.bar_label(i,)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{prefix}_neural_network_metrics.png")
    plt.close()
