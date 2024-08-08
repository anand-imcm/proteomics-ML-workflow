import argparse
from fpdf import FPDF
from PIL import Image
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

def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        return img.size

def add_model_page(pdf, model_name, file_prefix):
    pdf.add_page()
    pdf.set_font("Arial", size=15)
    pdf.cell(0, 10, txt=f"{model_name.capitalize()} Model Report", ln=True, align='L')

    effective_page_width = pdf.w - 2 * pdf.l_margin
    available_page_height = pdf.h - pdf.t_margin - pdf.b_margin - pdf.get_y()
    row_height = available_page_height / 2
    column_width = effective_page_width / 2
    y_offset = pdf.get_y()

    images = [f"{file_prefix}_{model_name}_roc_curve.png", f"{file_prefix}_{model_name}_metrics.png", f"{file_prefix}_{model_name}_confusion_matrix.png", f"{file_prefix}_{model_name}_shap_radar.png"]
    
    for i, image in enumerate(images):
        if os.path.exists(image):
            image_width, image_height = get_image_dimensions(image)
            aspect_ratio = image_height / image_width
            width = column_width
            height = width * aspect_ratio
            if height > row_height:
                height = row_height
                width = height / aspect_ratio
            x_offset = pdf.l_margin + (i % 2) * column_width
            pdf.image(image, x=x_offset, y=y_offset, w=width, h=height)
            if i % 2 == 1:
                y_offset += row_height + 5
                pdf.set_y(y_offset)

if __name__ == "__main__":
    args = parse_arguments()
    models = args.models
    prefix = args.prefix.lower()
    model_choices = [model_map[item] for item in models]
    # Create PDF report
    pdf = FPDF()

    # Add the overall ROC curve page
    pdf.add_page()
    pdf.set_font("Arial", size=15)
    pdf.cell(0, 10, txt="Overall ROC Curves for All Models", ln=True, align='L')
    pdf.image(f"{prefix}_overall_roc_curves.png", x=10, y=20, w=190)

    # Add pages for each model
    for model_name in model_choices:
        add_model_page(pdf, model_name, prefix)

    # Save PDF
    pdf.output(f"{prefix}_model_reports.pdf")
