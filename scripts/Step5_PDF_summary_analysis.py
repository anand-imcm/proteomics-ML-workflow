from fpdf import FPDF
import os
import logging
import re  # Import the regex module

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class PDFReport(FPDF):
    def header(self):
        # Header appears on every page
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "IMCM ML Workflow Analysis Report", 0, 1, "C")
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

        # Approximate title height in mm (1 pt = 0.3528 mm)
        title_height = 40 * 0.3528
        y_position = (page_height / 2) - (title_height / 2)

        self.set_xy(0, y_position)
        self.cell(0, title_height, title, 0, 1, "C")

    def chapter_title_left_aligned(self, title):
        """
        Add a left-aligned chapter title positioned at one-third of the page vertically.
        Font size is set to 40.
        """
        self.add_page()
        self.set_font("Arial", "B", 40)
        y_position = self.h / 3
        self.set_xy(10, y_position)
        self.cell(0, 10, title, 0, 1, "L")

    def chapter_title_regular(self, title):
        """
        Add a regular centered chapter title (used for image titles).
        Font size is set to 16.
        """
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, title, 0, 1, "C")
        self.ln(5)

    def add_text(self, text):
        """
        Add a block of text to the PDF.
        """
        self.set_font("Arial", size=12)
        self.multi_cell(0, 10, text)
        self.ln(5)

    def find_image_file(self, model_name, image_type=None):
        """
        Search for an image file that matches the model_name and image_type.
        - If image_type is provided, look for files containing both model_name and image_type.
        - If image_type is None, look for files containing only model_name.
        - The search is case-insensitive and looks for .png files.
        
        Return the matched filename or None if not found.
        """
        pattern = re.escape(model_name.lower())
        if image_type:
            # Pattern: modelname_imagetype.png
            pattern += f"_{re.escape(image_type.lower())}"
        # Pattern ends with .png
        pattern += r"\.png$"

        regex = re.compile(pattern)
        matched_files = [file for file in os.listdir('.') if regex.search(file.lower())]

        if matched_files:
            return matched_files[0]  # Return the first match
        return None  # Not found

    def add_image_with_caption(self, model_name, image_type, title, caption):
        """
        Add an image (if found) with a title and caption to the PDF.
        """
        img_path = self.find_image_file(model_name, image_type)
        if img_path:
            self.add_page()
            self.chapter_title_regular(title)
            self.add_text(caption)
            try:
                # Position the image below the caption
                current_y = self.get_y()
                self.image(img_path, x=10, y=current_y, w=190)
                self.ln(10)
            except RuntimeError:
                logging.warning(f"Failed to load image: {img_path}")
        else:
            if image_type == "models_summary_bar_chart":
                expected = f"{model_name}.png"
            else:
                if image_type:
                    expected = f"{model_name}_{image_type}.png"
                else:
                    expected = f"{model_name}.png"
            logging.warning(
                f"Image not found for model '{model_name}' with type '{image_type}'. "
                f"Expected something like '{expected}'"
            )

# Create PDF report
pdf = PDFReport()

####################################
# Introduction section
####################################
pdf.chapter_title_left_aligned("Introduction")
introduction_text = (
    "This report provides an analysis of various ML models and their performance metrics. "
    "For regression models, it includes evaluation metrics such as MSE, MAE, RMSE, and R2, "
    "along with residual plots, actual vs predicted plots, and SHAP plots (summary bar and beeswarm). "
    "Additionally, an overall performance chart compares metrics across multiple regression models."
)
pdf.add_text(introduction_text)

####################################
# Dimensionality Reduction Section
####################################
pdf.chapter_title_centered("Dimensionality Reduction")
analysis_images = {
    'KPCA': ('Kernel Principal Component Analysis (KPCA)', 'KPCA results providing a non-linear dimensionality reduction view.'),
    'PCA': ('Principal Component Analysis (PCA)', 'PCA results showing the distribution of data points in reduced dimensions PCs.'),
    'PLS': ('Partial Least Squares (PLS)', 'PLS results demonstrating the relationship between features and target variables in LVs.'),
    't-SNE': ('t-Distributed Stochastic Neighbor Embedding (t-SNE)', 't-SNE results visualizing the data in a lower-dimensional space.'),
    'UMAP': ('Uniform Manifold Approximation and Projection (UMAP)', 'UMAP results displaying the clustering of data points.')
}
# Dimensionality reduction images are identified by their specific model names with '_result.png'
for model_name, (title, caption) in analysis_images.items():
    pdf.add_image_with_caption(model_name, 'result', title, caption)

####################################
# Classification Section
####################################
pdf.chapter_title_centered("Classification Section")

# Overall classification chart example
pdf.add_image_with_caption(
    "overall_roc_curves",
    None,
    "Overall ROC Curves for All Classification Models",
    "This plot shows the ROC curves for all classification models."
)

classification_image_types = {
    'confusion_matrix': 'Confusion Matrix',
    'metrics': 'Evaluation Metrics',
    'roc_curve': 'ROC Curve',
    'shap_radar': 'Feature Importance - SHAP Radar Plot',
    'shap_bar': 'Feature Importance - SHAP Bar Plot for Multiclass',
    'nested_cv_f1_auc': 'Nested CV F1 and AUC Scores Plot Showing Five-Fold Classification Results'
}

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

####################################
# Regression Section
####################################
pdf.chapter_title_centered("Regression Section")

# Overall regression summary bar chart
pdf.add_image_with_caption(
    "models_summary_bar_chart",
    None,
    "Overall Regression Model Performance Metrics",
    "This chart compares MSE, MAE, RMSE, and R2 across all regression models."
)

# Define the typical image suffixes for regression
regression_image_types_general = {
    'overall_prediction': 'Actual vs Predicted Plot',
    'overall_residuals': 'Residual Plot',
    'shap_summary_dot': 'SHAP Beeswarm Plot',
    'shap_summary_bar': 'SHAP Summary Bar Plot',
    'metrics_line_plot': 'Metrics Across Five-Fold CV Results'
}

regression_image_types_vae_reg = {
    'predictions': 'Actual vs Predicted Plot',
    'residuals': 'Residual Plot',
    'metrics_over_folds': 'Metrics Across Five-Fold CV Results',
    'shap_summary': 'SHAP Summary Plot',
    'average_metrics': 'Average Evaluation Metrics Plot'  # New image type
}

regression_image_types_vaemlp_reg = {
    'predictions': 'Actual vs Predicted Plot',
    'residuals': 'Residual Plot',
    'metrics_over_folds': 'Metrics Across Five-Fold CV Results',
    'shap_summary': 'SHAP Summary Plot',
    'average_metrics': 'Average Evaluation Metrics Plot'  # New image type
}

# Regression models
regression_models = {
    'Neural_Network_reg': 'Neural Network regression model performance analysis.',
    'Random_Forest_reg': 'Random Forest regression model performance analysis.',
    'SVM_reg': 'Support Vector Machine (SVM) regression model performance analysis.',
    'XGBoost_reg': 'XGBoost regression model performance analysis.',
    'PLS_reg': 'Partial Least Squares (PLS) regression model performance analysis.',
    'KNN_reg': 'K-Nearest Neighbors (KNN) regression model performance analysis.',
    'LightGBM_reg': 'LightGBM regression model performance analysis.',
    'VAE_reg': 'Variational Autoencoder (VAE) regression model performance analysis.',
    'vaemlp_reg': 'MLP in VAE regression model performance analysis.'
}

for model_name, caption in regression_models.items():
    # Determine which image types to use based on the model
    if model_name.lower() == 'vae_reg':
        image_types = regression_image_types_vae_reg
    elif model_name.lower() == 'vaemlp_reg':
        image_types = regression_image_types_vaemlp_reg
    else:
        image_types = regression_image_types_general

    for img_type, img_description in image_types.items():
        # Special handling for VAE_reg and vaemlp_reg image types
        if model_name.lower() in ['vae_reg', 'vaemlp_reg']:
            # Image types for VAE_reg and vaemlp_reg have different naming
            # e.g., 'predictions', 'residuals', etc.
            title = f"{model_name} {img_description}"
        else:
            # Image types for other regression models
            title = f"{model_name.replace('_', ' ').title()} {img_description}"
        pdf.add_image_with_caption(model_name, img_type, title, caption)

####################################
# Biological Analysis Section
####################################
pdf.chapter_title_centered("Biological Analysis")

all_models = list(classification_models.keys()) + list(regression_models.keys())

biological_image_types = {
    'PPI_S': 'Protein-Protein Interaction (PPI) network visualization. S plot based on STRING database.',
    'PPI_M': 'Protein-Protein Interaction (PPI) network visualization. M plot based on tree models.'
}

biological_titles = {
    'PPI_S': 'Function Enrichment S Plot',
    'PPI_M': 'Function Enrichment M Plot'
}

for model_name in all_models:
    for img_type, img_description in biological_image_types.items():
        title = f"{biological_titles[img_type]} for {model_name.replace('_', ' ').title()}"
        pdf.add_image_with_caption(model_name, img_type, title, img_description)

# Save final PDF
pdf.output("model_reports.pdf")
