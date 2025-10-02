import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os
from sklearn.metrics import auc
from itertools import cycle

# Model map with uppercase model abbreviations to lowercase model names
model_map = {
    "RF": "random_forest",
    "KNN": "knn",
    "NN": "neural_network",
    "SVM": "svm",
    "XGB": "xgboost",
    "PLSDA": "plsda",
    "VAE": "vae",
    "LR": "logistic_regression",     # Logistic Regression
    "GNB": "gaussiannb",             # Gaussian Naive Bayes
    "LGBM": "lightgbm",              # LightGBM
    "MLPVAE": "vaemlp"
}

def parse_arguments():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Plot ROC curves for selected models.")
    parser.add_argument('-m', '--models', type=str, nargs='+', 
                        choices=['KNN', 'RF', 'NN', 'SVM', 'XGB', 'PLSDA', 'VAE', 'LR', 'GNB', 'LGBM','MLPVAE'], 
                        help='Name of the model(s)', required=True)
    parser.add_argument('-p', '--prefix', type=str, required=True, help='Output prefix (include path and prefix name)')
    return parser.parse_args()

def compute_macro_average_roc(fpr_dict, tpr_dict, keys=None):
    """
    Compute macro-average ROC curve from per-class curves.
    Only integer keys (i.e., real classes) are used.
    """
    if keys is None:
        keys = [k for k in fpr_dict.keys() if isinstance(k, (int, np.integer))]

    if len(keys) == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.5

    all_fpr = np.unique(np.concatenate([fpr_dict[k] for k in keys]))
    mean_tpr = np.zeros_like(all_fpr)

    for k in keys:
        mean_tpr += np.interp(all_fpr, fpr_dict[k], tpr_dict[k])

    mean_tpr /= len(keys)
    roc_auc_macro = auc(all_fpr, mean_tpr)
    return all_fpr, mean_tpr, roc_auc_macro

def plot_roc(model_abbrs, file_prefix):
    # Load ROC data for all models
    roc_data = {}
    valid_model_abbrs = []  # store successfully loaded model abbreviations
    for model_abbr in model_abbrs:
        if model_abbr not in model_map:
            print(f"Model abbreviation '{model_abbr}' not recognized. Skipping.")
            continue
        model_name = model_map[model_abbr]  # e.g., 'knn'
        model_file = f"{file_prefix}_{model_name}_roc_data.npy"
        print(f"Loading ROC data from: {model_file}")
        if not os.path.isfile(model_file):
            print(f"ROC data file for model '{model_name}' not found at '{model_file}'. Skipping.")
            continue
        try:
            roc_data[model_abbr] = np.load(model_file, allow_pickle=True).item()
            valid_model_abbrs.append(model_abbr)
        except Exception as e:
            print(f"Error loading ROC data for model '{model_name}': {e}. Skipping.")
            continue

    if not valid_model_abbrs:
        print("No ROC data found for any of the specified models. Exiting.")
        return

    # Plot ROC curves for the loaded models
    plt.figure(figsize=(15, 12))
    
    # Colors for different models
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 'brown', 'pink', 'gray', 'olive','yellow'])
    
    for model_abbr, color in zip(valid_model_abbrs, colors):
        model_roc = roc_data[model_abbr]

        if isinstance(model_roc['fpr'], dict):
            fpr_dict = model_roc['fpr']
            tpr_dict = model_roc['tpr']
            auc_dict = model_roc['roc_auc']

            # Only count integer keys as classes
            class_keys = [k for k in fpr_dict.keys() if isinstance(k, (int, np.integer))]
            print(f"Model '{model_abbr}' class keys: {class_keys}")

            if len(class_keys) > 1:
                # Prefer stored macro; otherwise compute from class curves
                if 'macro' in fpr_dict and 'macro' in tpr_dict and 'macro' in auc_dict:
                    plt.plot(
                        fpr_dict['macro'], tpr_dict['macro'],
                        color=color, linestyle='-', linewidth=2,
                        label=f'{model_abbr} Macro AUC = {auc_dict["macro"]:.2f}'
                    )
                else:
                    fpr_macro, tpr_macro, roc_auc_macro = compute_macro_average_roc(fpr_dict, tpr_dict, keys=class_keys)
                    plt.plot(
                        fpr_macro, tpr_macro,
                        color=color, linestyle='-', linewidth=2,
                        label=f'{model_abbr} Macro AUC = {roc_auc_macro:.2f}'
                    )

                # Plot stored micro if available
                if 'micro' in fpr_dict and 'micro' in tpr_dict and 'micro' in auc_dict:
                    plt.plot(
                        fpr_dict['micro'], tpr_dict['micro'],
                        color=color, linestyle='--', linewidth=2,
                        label=f'{model_abbr} Micro AUC = {auc_dict["micro"]:.2f}'
                    )
            else:
                # Binary case stored as a dict with a single integer key
                if len(class_keys) == 1:
                    key = class_keys[0]
                else:
                    if 'micro' in fpr_dict:
                        key = 'micro'
                    elif 'macro' in fpr_dict:
                        key = 'macro'
                    else:
                        key = next(iter(fpr_dict))
                     #key = list(fpr_dict.keys())[0]
                fpr = fpr_dict[key]
                tpr = tpr_dict[key]
                roc_auc_val = auc_dict[key]
                plt.plot(
                    fpr, tpr,
                    color=color, linestyle='-', linewidth=2,
                    label=f'{model_abbr} AUC = {roc_auc_val:.2f}'
                )
        else:
            # Binary case stored as arrays
            fpr = model_roc['fpr']
            tpr = model_roc['tpr']
            roc_auc_val = model_roc['roc_auc']
            plt.plot(
                fpr, tpr,
                color=color, linestyle='-', linewidth=2,
                label=f'{model_abbr} AUC = {roc_auc_val:.2f}'
            )

    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=28, labelpad=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=28, labelpad=12)
    plt.title('ROC Curves for Selected Models', fontsize=36, fontweight='bold')
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=22, title_fontsize=22, borderaxespad=0.)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{file_prefix}_overall_roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    args = parse_arguments()
    models = args.models
    prefix = args.prefix
    plot_roc(models, prefix)
