import argparse
from fpdf import FPDF
import os

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
    parser = argparse.ArgumentParser(description='Script for generating PDF report using the plots generated from previous steps.')
    parser.add_argument('-m', '--models', type=str, nargs='+', choices=['KNN', 'RF', 'NN', 'SVM', 'XGB', 'PLSDA','VAE'], help='Name of the model(s)', required=True)
    parser.add_argument('-p','--prefix',type=str, help='Output prefix')
    return parser.parse_args()

# Function to add a page with images and text for each model
def add_model_page(pdf, model_name, file_prefix):
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt=f"{model_name.capitalize()} Model Report", ln=True, align='L')

    # Add confusion matrix image
    pdf.image(f"{file_prefix}_{model_name}_confusion_matrix.png", x=10, y=20, w=90)

    # Add evaluation metrics image
    pdf.image(f"{file_prefix}_{model_name}_metrics.png", x=10, y=120, w=90)

    # Add ROC curve image
    pdf.image(f"{file_prefix}_{model_name}_roc_curve.png", x=105, y=20, w=90)

    # Add SHAP radar chart image if it exists
    shap_image_path = f"{file_prefix}_{model_name}_shap_radar.png"
    if os.path.exists(shap_image_path):
        pdf.image(shap_image_path, x=105, y=120, w=90)


if __name__ == "__main__":
    args = parse_arguments()
    models = args.models
    prefix = args.prefix.lower()
    model_choices = [model_map[item] for item in models]
    # Create PDF report
    pdf = FPDF()

    # Add the overall ROC curve page
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt="Overall ROC Curves for All Models", ln=True, align='L')
    pdf.image(f"{prefix}_overall_roc_curves.png", x=10, y=20, w=190)

    # Add pages for each model
    for model_name in model_choices:
        add_model_page(pdf, model_name, prefix)

    # Save PDF
    pdf.output(f"{prefix}_model_reports.pdf")
