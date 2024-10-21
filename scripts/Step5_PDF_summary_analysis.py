# +
from fpdf import FPDF
import os
import glob

class PDFReport(FPDF):
    def header(self):
        # Header appears on every page
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "IMCM ML workflow Analysis Report", 0, 1, "C")
        self.ln(5)  # Small spacing after header

    def chapter_title_centered(self, title):
        """
        Add a centered and enlarged chapter title both horizontally and vertically.
        Font size is set to 40.
        """
        self.add_page()
        self.set_font("Arial", "B", 40)  # Set font size to 40

        # Calculate width of the title
        title_width = self.get_string_width(title) + 6

        # Calculate positions to center the title
        page_width = self.w
        page_height = self.h

        # Calculate vertical position to center the title
        # Convert font size from points to mm (1 pt = 0.3528 mm)
        title_height = 40 * 0.3528  # Approximate title height in mm
        y_position = (page_height / 2) - (title_height / 2)

        # Set position
        self.set_xy(0, y_position)

        # Add the title centered horizontally
        self.cell(0, title_height, title, 0, 1, "C")
        # No need for extra ln(10)

    def chapter_title_left_aligned(self, title):
        """
        Add a left-aligned chapter title positioned at one-third of the page vertically.
        Font size is set to 40.
        """
        self.add_page()
        self.set_font("Arial", "B", 40)  # Set font size to 40

        # Calculate vertical position at one-third of the page
        page_height = self.h
        y_position = page_height / 3

        # Set position
        self.set_xy(10, y_position)  # Left margin of 10 mm

        # Add the title left-aligned
        self.cell(0, 10, title, 0, 1, "L")
        # No need for extra ln(10)

    def chapter_title_regular(self, title):
        """
        Add a regular centered chapter title (used for image titles).
        Font size is set to 16.
        """
        self.set_font("Arial", "B", 16)  # Set font size to 16
        self.cell(0, 10, title, 0, 1, "C")  # Center the title
        self.ln(5)  # Spacing after title

    def add_text(self, text):
        """
        Add a block of text to the PDF.
        """
        self.set_font("Arial", size=12)
        self.multi_cell(0, 10, text)
        self.ln(5)

    def find_image_file(self, model_name, image_type=None):
        """
        Search for an image file that ends with '<model_name>_<image_type>.png'.
        If image_type is None, search for '<model_name>.png'.
        The prefix can be anything.
        Case-insensitive matching.
        """
        if image_type:
            expected_suffix = f"{model_name}_{image_type}.png".lower()
        else:
            expected_suffix = f"{model_name}.png".lower()

        for file in os.listdir('.'):
            if file.lower().endswith(expected_suffix):
                return file
        return None  # Return None if not found

    def add_image_with_caption(self, model_name, image_type, title, caption):
        """
        Add an image with a title and caption to the PDF.
        It searches for the image file based on model_name and image_type.
        If image_type is None, it searches for '<model_name>.png'.
        """
        img_path = self.find_image_file(model_name, image_type)
        if img_path:
            self.add_page()
            self.chapter_title_regular(title)
            self.add_text(caption)
            try:
                self.image(img_path, x=10, y=self.get_y(), w=190)
                self.ln(10)  # Spacing after image
            except RuntimeError:
                # If image loading fails, silently skip
                pass
        # If image not found, silently skip

# Create PDF report
pdf = PDFReport()

# Introduction section
pdf.chapter_title_left_aligned("Introduction")
introduction_text = (
    "This report provides an analysis of various ML models and their performance metrics. "
    "It includes confusion matrices, evaluation metrics, ROC curves, and variable importance radar charts "
    "for classification models. For regression models, it includes evaluation metrics such as MSE, MAE, "
    "RMSE, and R2, along with residual plots, actual vs predicted plots, SHAP mean bar plots, and SHAP beeswarm plots. "
    "Additionally, an overall performance table for regression models is provided, comparing MSE, MAE, RMSE, and R2 across models."
)
pdf.add_text(introduction_text)

# Dimensionality Reduction Section
pdf.chapter_title_centered("Dimensionality Reduction")
analysis_images = {
    'KPCA': ('Kernel Principal Component Analysis (KPCA)', 'KPCA results providing a non-linear dimensionality reduction view.'),
    'PC': ('Principal Component Analysis (PCA)', 'PCA results showing the distribution of data points in reduced dimensions PCs.'),
    'PLS': ('Partial Least Squares (PLS)', 'PLS results demonstrating the relationship between features and target variables in LVs.'),
    't-SNE': ('t-Distributed Stochastic Neighbor Embedding (t-SNE)', 't-SNE results visualizing the data in a lower-dimensional space.'),
    'UMAP': ('Uniform Manifold Approximation and Projection (UMAP)', 'UMAP results displaying the clustering of data points.')
}

for model_name, (title, caption) in analysis_images.items():
    image_type = 'result'  # Assuming image_type is 'result' for these images
    pdf.add_image_with_caption(model_name, image_type, title, caption)

# Classification Section
pdf.chapter_title_centered("Classification Section")

# First page after Classification Section: output_prefix_overall_roc_curves.png
# Assuming model_name='output_prefix_overall_roc_curves', image_type=None
pdf.add_image_with_caption("output_prefix_overall_roc_curves", None, "Overall ROC Curves for All Classification Models", "This plot shows the ROC curves for all the classification models, providing a comparative view of their performance.")

# Define different image types for classification models
classification_image_types = {
    'confusion_matrix': 'Confusion Matrix',
    'metrics': 'Evaluation Metrics',
    'roc_curve': 'ROC Curve',
    'shap_radar': 'Feature Importance - SHAP Plot'
}

# Classification models with unique names
classification_models = {
    'neural_network': 'Fully connected Neural Network model performance analysis.',
    'random_forest': 'Random Forest model performance analysis.',
    'xgboost': 'XGBoost model performance analysis.',
    'svm': 'Support Vector Machine (SVM) model performance analysis.',
    'knn': 'K-Nearest Neighbors (KNN) model performance analysis.',
    'plsda': 'Partial Least Squares Discriminant Analysis (PLSDA) model performance analysis.',
    'vae': 'Variational Autoencoder with MLP (VAE_MLP) model performance analysis.',
    'lightgbm': 'LightGBM model performance analysis.',
    'logistic_regression': 'Logistic Regression model performance analysis.',
    'gaussiannb': 'Gaussian Naive Bayes model performance analysis.',
    'vaemlp': 'MLP in VAE model performance analysis.'
}

for model_name, caption in classification_models.items():
    for img_type, img_description in classification_image_types.items():
        title = f"{model_name.replace('_', ' ').title()} {img_description}"
        pdf.add_image_with_caption(model_name, img_type, title, caption)

# Regression Section
pdf.chapter_title_centered("Regression Section")

# First page after Regression Section: Results_reg_model_performance_metrics.png
# Assuming model_name='Results_reg_model_performance_metrics', image_type=None
pdf.add_image_with_caption("Results_reg_model_performance_metrics", None, "Regression Model Performance Metrics", "This table compares the evaluation metrics (MSE, MAE, RMSE, R2) for all regression models.")

# Define regression model image types
regression_image_types = {
    'shap_summary_dot': 'SHAP Beeswarm Plot',
    'shap_summary_bar': 'SHAP Summary Bar Plot',
    'model_performance_metrics': 'Model Performance Metrics Table',
    'prediction': 'Actual vs Predicted Plot',
    'residuals': 'Residual Plot'
}

# Regression models with updated names
regression_models = {
    'Neural_Network_reg': 'Neural Network regression model performance analysis.',
    'Random_Forest_reg': 'Random Forest regression model performance analysis.',
    'SVM_reg': 'Support Vector Machine (SVM) regression model performance analysis.',
    'XGBoost_reg': 'XGBoost regression model performance analysis.',
    'PLS_reg': 'Partial Least Squares (PLS) regression model performance analysis.',
    'KNN_reg': 'K-Nearest Neighbors (KNN) regression model performance analysis.',
    'LightGBM_reg': 'LightGBM regression model performance analysis.'
}

for model_name, caption in regression_models.items():
    for img_type, img_description in regression_image_types.items():
        title = f"{model_name} {img_description}"
        pdf.add_image_with_caption(model_name, img_type, title, caption)

# Save PDF
pdf.output("model_reports.pdf")

