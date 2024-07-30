import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

# Set up command line argument parsing
parser = argparse.ArgumentParser(description="Plot ROC curves for selected models.")
parser.add_argument('--models', nargs='+', required=True, help='List of model names.')

args = parser.parse_args()
models = args.models

# Check if any models were provided
if not models:
    print("Please provide at least one model name using --models.")
    sys.exit(1)

# Load ROC data for all models
roc_data = {}
for model in models:
    try:
        roc_data[model] = np.load(f'{model}_roc_data.npy', allow_pickle=True).item()
    except FileNotFoundError:
        print(f"ROC data file for model '{model}' not found.")
        sys.exit(1)

# Plot ROC curves for the selected models
plt.figure(figsize=(10, 8))
for model in models:
    if 'micro' in roc_data[model]['fpr']:
        # Multi-class case
        fpr = roc_data[model]['fpr']['micro']
        tpr = roc_data[model]['tpr']['micro']
        roc_auc = roc_data[model]['roc_auc']['micro']
        plt.plot(fpr, tpr, label=f'{model} (Overall AUC = {roc_auc:.2f})')
    else:
        # Binary case
        fpr = roc_data[model]['fpr'][0]
        tpr = roc_data[model]['tpr'][0]
        roc_auc = roc_data[model]['roc_auc'][0]
        plt.plot(fpr, tpr, label=f'{model} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Selected Models')
plt.legend(loc="lower right")
plt.savefig('overall_roc_curves.png')
plt.show()
