import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

model_map = {
    "RF": "random_forest",
    "KNN": "knn",
    "NN": "neural_network",
    "SVM": "svm",
    "XGB": "xgboost",
    "PLSDA": "plsda",
    "VAE": "vae"
}

def parse_arguments():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Plot ROC curves for selected models.")
    parser.add_argument('-m', '--models', type=str, nargs='+', choices=['KNN', 'RF', 'NN', 'SVM', 'XGB', 'PLSDA','VAE'], help='Name of the model(s)', required=True)
    parser.add_argument('-p','--prefix',type=str, required=True, help='Output prefix')
    return parser.parse_args()

def plot_roc(model_names,file_prefix):
    # Load ROC data for all models
    roc_data = {}
    for model in model_names:
        try:
            roc_data[model] = np.load(f'{file_prefix}_{model}_roc_data.npy', allow_pickle=True).item()
        except FileNotFoundError:
            print(f"ROC data file for model '{model}' not found.")
            sys.exit(1)

    # Plot ROC curves for the selected models
    plt.figure(figsize=(10, 8))
    for model in model_names:
        if 'micro' in roc_data[model]['fpr']:
            # Multi-class case
            fpr = roc_data[model]['fpr']['micro']
            tpr = roc_data[model]['tpr']['micro']
            roc_auc = roc_data[model]['roc_auc']['micro']
            plt.plot(fpr, tpr, label=f'{model.upper()} (Overall AUC = {roc_auc:.2f})')
        else:
            # Binary case
            fpr = roc_data[model]['fpr'][0]
            tpr = roc_data[model]['tpr'][0]
            roc_auc = roc_data[model]['roc_auc'][0]
            plt.plot(fpr, tpr, label=f'{model.upper()} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Selected Models')
    plt.legend(loc="lower right")
    plt.savefig(f"{file_prefix}_overall_roc_curves.png")


if __name__ == "__main__":
    args = parse_arguments()
    models = args.models
    model_choices = [model_map[item] for item in models]
    prefix = args.prefix.lower()
    plot_roc(model_choices, prefix)