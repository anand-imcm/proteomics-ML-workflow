from fpdf import FPDF

# Function to add a page with images and text for each model
def add_model_page(pdf, model_name):
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt=f"{model_name.capitalize()} Model Report", ln=True, align='L')

    # Add confusion matrix image
    pdf.image(f'{model_name}_confusion_matrix.png', x=10, y=20, w=90)

    # Add evaluation metrics image
    pdf.image(f'{model_name}_metrics.png', x=10, y=120, w=90)

    # Add ROC curve image
    pdf.image(f'{model_name}_roc_curve.png', x=105, y=20, w=90)

    # Add SHAP radar chart image
    pdf.image(f'{model_name}_shap_radar_plot.png', x=105, y=120, w=90)

# Create PDF report
pdf = FPDF()

# Add the overall ROC curve page
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(0, 10, txt="Overall ROC Curves for All Models", ln=True, align='L')
pdf.image('overall_roc_curves.png', x=10, y=20, w=190)

# List of model names
model_names = [
    # 'gaussiannb', 
    # 'logistic_regression', 
    # 'lightgbm', 
    # 'vae', 
    'neural_network', 
    'random_forest', 
    'xgboost', 
    'svm', 
    #'plsda', 
    'knn'
]

# Add pages for each model
for model_name in model_names:
    add_model_page(pdf, model_name)

# Save PDF
pdf.output("model_reports.pdf")
