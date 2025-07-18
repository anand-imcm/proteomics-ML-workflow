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
    "LGBM": "lightgbm",               # LightGBM
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

def compute_macro_average_roc(fpr_dict, tpr_dict):
    """
    Compute macro-average ROC curve.
    """
    # Collect all unique false positive rates
    all_fpr = np.unique(np.concatenate([fpr_dict[k] for k in fpr_dict]))
    
    # Initialize mean TPR
    mean_tpr = np.zeros_like(all_fpr)
    
    # Interpolate each ROC curve and accumulate the TPR
    for k in fpr_dict:
        mean_tpr += np.interp(all_fpr, fpr_dict[k], tpr_dict[k])
    
    # Average the accumulated TPR
    mean_tpr /= len(fpr_dict)
    
    # Compute AUC for macro-average ROC
    roc_auc_macro = auc(all_fpr, mean_tpr)
    
    return all_fpr, mean_tpr, roc_auc_macro

def plot_roc(model_abbrs, file_prefix):
    # Load ROC data for all models
    roc_data = {}
    valid_model_abbrs = []  # 保存加载成功的模型缩写
    for model_abbr in model_abbrs:
        if model_abbr not in model_map:
            print(f"Model abbreviation '{model_abbr}' not recognized. Skipping.")
            continue
        model_name = model_map[model_abbr]  # e.g., 'knn'
        # Construct the full file path
        model_file = f"{file_prefix}_{model_name}_roc_data.npy"
        print(f"Loading ROC data from: {model_file}")  # Debugging output
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
        # Determine if it's binary or multi-class by checking the type of fpr
        if isinstance(model_roc['fpr'], dict):
            num_classes = len(model_roc['fpr'])
            if num_classes > 1:
                # Multi-class case: compute macro-average ROC
                fpr_dict = model_roc['fpr']
                tpr_dict = model_roc['tpr']
                roc_auc_dict = model_roc['roc_auc']
                print(f"Model '{model_abbr}' has {num_classes} classes: {list(fpr_dict.keys())}")  # Debugging output
                fpr_macro, tpr_macro, roc_auc_macro = compute_macro_average_roc(fpr_dict, tpr_dict)
                
                # Plot macro-average ROC
                plt.plot(fpr_macro, tpr_macro, color=color, linestyle='-', linewidth=2,
                         label=f'{model_abbr} Macro AUC = {roc_auc_macro:.2f}')
                
                # Plot micro-average ROC if exists
                if 'micro' in model_roc['fpr']:
                    fpr_micro = model_roc['fpr']['micro']
                    tpr_micro = model_roc['tpr']['micro']
                    roc_auc_micro = model_roc['roc_auc']['micro']
                    plt.plot(fpr_micro, tpr_micro, color=color, linestyle='--', linewidth=2,
                             label=f'{model_abbr} Micro AUC = {roc_auc_micro:.2f}')
            else:
                # Binary case stored as a dict with a single key
                key = list(model_roc['fpr'].keys())[0]
                fpr = model_roc['fpr'][key]
                tpr = model_roc['tpr'][key]
                roc_auc = model_roc['roc_auc'][key]
                plt.plot(fpr, tpr, color=color, linestyle='-', linewidth=2,
                         label=f'{model_abbr} AUC = {roc_auc:.2f}')
        else:
            # Binary case stored as arrays
            fpr = model_roc['fpr']
            tpr = model_roc['tpr']
            roc_auc = model_roc['roc_auc']
            plt.plot(fpr, tpr, color=color, linestyle='-', linewidth=2,
                     label=f'{model_abbr} AUC = {roc_auc:.2f}')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=28, labelpad=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=28, labelpad=12)
    plt.title('ROC Curves for Selected Models', fontsize=36, fontweight='bold')
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=22, title_fontsize=22, borderaxespad=0.)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{file_prefix}_overall_roc_curves.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    args = parse_arguments()
    models = args.models
    prefix = args.prefix
    plot_roc(models, prefix)
