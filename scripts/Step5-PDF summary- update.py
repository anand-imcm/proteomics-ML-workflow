from fpdf import FPDF
import os

class PDFReport(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "Model Analysis Report", 0, 1, "C")
        self.ln(10)

    def chapter_title(self, title):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, title, 0, 1, "L")
        self.ln(5)

    def add_text(self, text):
        self.set_font("Arial", size=12)
        self.multi_cell(0, 10, text)
        self.ln(5)

    def add_image(self, img_path):
        if os.path.exists(img_path):
            self.image(img_path, x=10, y=self.get_y(), w=190)
            self.ln(95)  # Ensure there's enough space below the image

# Create PDF report
pdf = PDFReport()

# Introduction section
pdf.add_page()
pdf.chapter_title("Introduction")
pdf.set_font("Arial", size=12)
introduction_text = (
    "This report provides an analysis of various ML models and their performance metrics. "
    "It includes confusion matrices, evaluation metrics, ROC curves, and variable importance radar charts "
    "for each model. Additionally, it presents the results of different dimensionality "
    "reduction techniques such as PCA, t-SNE, UMAP, KPCA, and PLS."
)
pdf.multi_cell(0, 10, introduction_text)
pdf.ln(10)

# Overall ROC Curves section
pdf.add_page()
pdf.chapter_title("Overall ROC Curves for All Models")
overall_roc_curve_path = 'overall_roc_curves.png'
if os.path.exists(overall_roc_curve_path):
    overall_roc_caption = "This plot shows the ROC curves for all the models evaluated in this report, providing a comparative view of their performance."
    pdf.add_text(overall_roc_caption)
    pdf.add_image(overall_roc_curve_path)

# Analysis Images section
analysis_images = {
    'PC_result': ('Principal Component Analysis (PCA)', 'PCA results showing the distribution of data points in reduced dimensions PCs.'),
    't-SNE_result': ('t-Distributed Stochastic Neighbor Embedding (t-SNE)', 't-SNE results visualizing the data in a lower-dimensional space.'),
    'UMAP_result': ('Uniform Manifold Approximation and Projection (UMAP)', 'UMAP results displaying the clustering of data points.'),
    'KPCA_result': ('Kernel Principal Component Analysis (KPCA)', 'KPCA results providing a non-linear dimensionality reduction view.'),
    'PLS_result': ('Partial Least Squares (PLS)', 'PLS results demonstrating the relationship between features and target variables in LVs.')
}

for img_name, (title, caption) in analysis_images.items():
    img_path = f'{img_name}.png'
    pdf.add_page()
    pdf.chapter_title(title)
    pdf.add_text(caption)
    pdf.add_image(img_path)

# Model Images section
model_images = {
    'neural_network': 'Fully connected Neural Network model performance analysis including confusion matrix, evaluation metrics, ROC curve, and variable importance radar plot.',
    'random_forest': 'Random Forest model performance analysis including confusion matrix, evaluation metrics, ROC curve, and variable importance radar plot.',
    'xgboost': 'XGBoost model performance analysis including confusion matrix, evaluation metrics, ROC curve, and variable importance radar plot.',
    'svm': 'Support Vector Machine (SVM) model performance analysis including confusion matrix, evaluation metrics, ROC curve, and variable importance radar plot.',
    'knn': 'K-Nearest Neighbors (KNN) model performance analysis including confusion matrix, evaluation metrics, ROC curve, and variable importance radar plot.',
    'plsda': 'Partial Least Squares Discriminant Analysis (PLSDA) model performance analysis including confusion matrix, evaluation metrics, ROC curve, and variable importance radar plot.',
    'vae_mlp': 'Variational Autoencoder with MLP (VAE_MLP) model performance analysis including confusion matrix, evaluation metrics, ROC curve, and variable importance radar plot.'
}

for model_name, caption in model_images.items():
    confusion_matrix_path = f'{model_name}_confusion_matrix.png'
    metrics_path = f'{model_name}_metrics.png'
    roc_curve_path = f'{model_name}_roc_curve.png'
    shap_image_path = f'{model_name}_shap_radar_plot.png'
    
    pdf.add_page()
    pdf.chapter_title(f"{model_name.capitalize()} Model Report")
    pdf.add_text(f"{model_name.capitalize()} Confusion Matrix")
    pdf.add_text(caption)
    pdf.add_image(confusion_matrix_path)
    
    pdf.add_page()
    pdf.chapter_title(f"{model_name.capitalize()} Evaluation Metrics")
    pdf.add_text(caption)
    pdf.add_image(metrics_path)
    
    pdf.add_page()
    pdf.chapter_title(f"{model_name.capitalize()} ROC Curve")
    pdf.add_text(caption)
    pdf.add_image(roc_curve_path)
    
    pdf.add_page()
    pdf.chapter_title(f"{model_name.capitalize()} SHAP Radar Plot")
    pdf.add_text(caption)
    pdf.add_image(shap_image_path)

# Save PDF
pdf.output("model_reports.pdf")
