from fpdf import FPDF
import os
import logging
import re

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
        Add a centered and enlarged chapter title on a new page.
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
        Add a left-aligned chapter title on a new page positioned at one-third of the page vertically.
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
        Returns True if an image is found and added, otherwise returns False.
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
            return True
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
            return False

# Create PDF report
pdf = PDFReport()

####################################
# Introduction Section (Text only)
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
analysis_images = {
    'KPCA': ('Kernel Principal Component Analysis (KPCA)', 'KPCA results providing a non-linear dimensionality reduction view.'),
    'PCA': ('Principal Component Analysis (PCA)', 'PCA results showing the distribution of data points in reduced dimensions PCs.'),
    'PLS': ('Partial Least Squares (PLS)', 'PLS results demonstrating the relationship between features and target variables in LVs.'),
    'tSNE': ('t-Distributed Stochastic Neighbor Embedding (t-SNE)', 't-SNE results visualizing the data in a lower-dimensional space.'),
    'UMAP': ('Uniform Manifold Approximation and Projection (UMAP)', 'UMAP results displaying the clustering of data points.')
}
# Check if at least one image exists for the section
found_any = False
for model_name, (title, caption) in analysis_images.items():
    if pdf.find_image_file(model_name, 'result'):
        found_any = True
        break
if found_any:
    pdf.chapter_title_centered("Dimensionality Reduction")
    for model_name, (title, caption) in analysis_images.items():
        if pdf.find_image_file(model_name, 'result'):
            pdf.add_image_with_caption(model_name, 'result', title, caption)

####################################
# Classification Section
####################################
classification_image_types = {
    'confusion_matrix': 'Confusion Matrix',
    'metrics': 'Evaluation Metrics',
    'roc_curve': 'ROC Curve',
    'shap_radar': 'Feature Importance - SHAP Radar Plot',
    'shap_bar': 'Feature Importance - SHAP Bar Plot for Multiclass',
    'nested_cv_f1_auc': 'Five-Fold Classification Results'
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
found_any = False
if pdf.find_image_file("overall_roc_curves", None):
    found_any = True
else:
    for model_name in classification_models.keys():
        for img_type in classification_image_types.keys():
            if pdf.find_image_file(model_name, img_type):
                found_any = True
                break
        if found_any:
            break
if found_any:
    pdf.chapter_title_centered("Classification Section")
    if pdf.find_image_file("overall_roc_curves", None):
        pdf.add_image_with_caption(
            "overall_roc_curves",
            None,
            "Overall ROC Curves for All Classification Models",
            "This plot shows the ROC curves for all classification models."
        )
    for model_name, caption in classification_models.items():
        for img_type, img_description in classification_image_types.items():
            if pdf.find_image_file(model_name, img_type):
                title = f"{model_name.replace('_', ' ').title()} {img_description}"
                pdf.add_image_with_caption(model_name, img_type, title, caption)

####################################
# Regression Section
####################################
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
    'average_metrics': 'Average Evaluation Metrics Plot'
}
regression_image_types_vaemlp_reg = {
    'predictions': 'Actual vs Predicted Plot',
    'residuals': 'Residual Plot',
    'metrics_over_folds': 'Metrics Across Five-Fold CV Results',
    'shap_summary': 'SHAP Summary Plot',
    'average_metrics': 'Average Evaluation Metrics Plot'
}
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
found_any = False
if pdf.find_image_file("models_summary_bar_chart", None):
    found_any = True
else:
    for model_name, caption in regression_models.items():
        if model_name.lower() == 'vae_reg':
            image_types = regression_image_types_vae_reg
        elif model_name.lower() == 'vaemlp_reg':
            image_types = regression_image_types_vaemlp_reg
        else:
            image_types = regression_image_types_general
        for img_type in image_types.keys():
            if pdf.find_image_file(model_name, img_type):
                found_any = True
                break
        if found_any:
            break
if found_any:
    pdf.chapter_title_centered("Regression Section")
    if pdf.find_image_file("models_summary_bar_chart", None):
        pdf.add_image_with_caption(
            "models_summary_bar_chart",
            None,
            "Overall Regression Model Performance Metrics",
            "This chart compares MSE, MAE, RMSE, and R2 across all regression models."
        )
    for model_name, caption in regression_models.items():
        if model_name.lower() == 'vae_reg':
            image_types = regression_image_types_vae_reg
        elif model_name.lower() == 'vaemlp_reg':
            image_types = regression_image_types_vaemlp_reg
        else:
            image_types = regression_image_types_general
        for img_type, img_description in image_types.items():
            if pdf.find_image_file(model_name, img_type):
                if model_name.lower() in ['vae_reg', 'vaemlp_reg']:
                    title = f"{model_name} {img_description}"
                else:
                    title = f"{model_name.replace('_', ' ').title()} {img_description}"
                pdf.add_image_with_caption(model_name, img_type, title, caption)

####################################
# Biological Analysis Section (Network Images)
####################################
# Read files matching the pattern "Network_X.png" where X is one or more digits.
network_files = [file for file in os.listdir('.') if re.match(r'Network_\d+\.png$', file)]
if network_files:
    pdf.chapter_title_centered("Biological Analysis")
    for netfile in network_files:
        m = re.search(r'Network_(\d+)\.png$', netfile)
        network_num = m.group(1) if m else ""
        title = f"PPI Network Analysis {network_num}"
        caption = f"This PPI figure illustrates the interactions among proteins {network_num}."
        pdf.add_page()
        pdf.chapter_title_regular(title)
        pdf.add_text(caption)
        try:
            current_y = pdf.get_y()
            pdf.image(netfile, x=10, y=current_y, w=190)
            pdf.ln(10)
        except RuntimeError:
            logging.warning(f"Failed to load image: {netfile}")

# Save final PDF
pdf.output("model_reports.pdf")
