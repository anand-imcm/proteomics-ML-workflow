# proteomics-ML-workflow

[![Open](https://img.shields.io/badge/Open-Dockstore-blue)](https://dockstore.org/workflows/github.com/anand-imcm/proteomics-ML-workflow)&nbsp;&nbsp;
[![publish](https://img.shields.io/github/actions/workflow/status/anand-imcm/proteomics-ML-workflow/publish_gen.yml)](https://github.com/anand-imcm/proteomics-ML-workflow/releases)&nbsp;&nbsp;
![GitHub release (with filter)](https://img.shields.io/github/v/release/anand-imcm/proteomics-ML-workflow)&nbsp;&nbsp;
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13378490.svg)](https://doi.org/10.5281/zenodo.13378490)&nbsp;&nbsp;

> [!TIP]
> To import the workflow into your Terra workspace, click on the above Dockstore badge, and select 'Terra' from the 'Launch with' widget on the Dockstore workflow page.

## Introduction

High-throughput affinity and mass-spectrometry-based proteomic studies of large clinical cohorts generate vast proteomic data and can enable rapid disease biomarker discovery. Here, we introduce an advanced machine learning (ML) workflow designed to streamline the ML analysis of proteomics data, thus enabling researchers to efficiently leverage sophisticated algorithms in the search for critical disease biomarkers.

The workflow: takes proteomic data and sample labels as input, imputing missing values where necessary; pre-processes the data for ML models and optionally performs dimensionality reduction; makes available as standard a catalogue of machine learning and deep learning classification and regression models, including both well established and cutting-edge methods; calculates accuracy, sensitivity and specificity of models, enabling the evaluation and comparison of models based on these metrics; and carries out feature selection in models using SHapley Additive exPlanations (SHAP) values. In addition to these ML capabilities, the workflow also provides downstream modules for functional enrichment and protein-protein interaction (PPI) network analyses of feature-selected proteins.

The workflow is implemented in Python, R and Workflow Description Language (WDL), and can be executed on a cloud-based platform for biomedical data analysis. Deployment in this manner provides a standardized, user-friendly interface, and ensures the reproducibility and reliability of analytical outputs. Furthermore, such deployment renders the workflow scalable and streamlines the analysis of large, complex proteomic data. This ML workflow thus represents a significant advancement, empowering researchers to efficiently explore proteomic landscapes and identify biomarkers critical for early detection and treatment of diseases.

## Workflow Steps

- **Preprocessing** : By default, Z-score standardisation is applied to the input data. Optionally, users can choose to apply dimensionality reduction to the dataset. Display scatter plots for every two dimensions based on the selected number of output dimensions. The available methods include:
  - `PCA` (Principal Component Analysis for linear data)
  - `ELASTICNET` (ElasticNet Regularization)
  - `UMAP` (Uniform Manifold Approximation and Projection)
  - `TSNE` (t-Distributed Stochastic Neighbor Embedding)
  - `KPCA` (Kernel Principal Component Analysis for non-linear data)
  - `PLS` (Partial Least Squares Regression)

- **Classification** : This step applies the machine learning models to the standardized data and generates a confusion matrix, ROC plots for all classes and averages, and other relevant evaluation metrics (Accuracy, F1, sensitivity, specificity) for all the models. The available algorithms are as follows:
  - `RF` (Random Forest)
  - `KNN` (K-Nearest Neighbors)
  - `NN` (Neural Network)
  - `SVM` (Support Vector Machine)
  - `XGB` (XGBoost)
  - `PLSDA` (Partial Least Squares Discriminant Analysis)
  - `VAE` (Variational Autoencoder with Multilayer Perceptron)
  - `LR` (Logistic Regression)
  - `GNB` (Gaussian Naive Bayes)
  - `LGBM` (LightGBM)
  - `MLPVAE` (Multilayer Perceptron inside Variational Autoencoder)
  
- **Regression** : This step applies the machine learning models to the standardized data and generates a confusion matrix, ROC plots for all classes and averages, and other relevant evaluation metrics (Accuracy, F1, sensitivity, specificity) for all the models. The available algorithms are as follows:
  - `RF_REG` (Random Forest Regression)
  - `NN_REG` (Neural Network Regression)
  - `SVM_REG` (Support Vector Regression)
  - `XGB_REG` (XGBoost Regression)
  - `PLS_REG` (Partial Least Squares Regression)
  - `KNN_REG` (K-Nearest Neighbors Regression)
  - `LGBM_REG` (LightGBM Regression)
  - `VAE_REG` (Variational Autoencoder with Multilayer Perceptron)
  - `MLPVAE_reg` (Multilayer Perceptron inside Variational Autoencoder)

- **SHAP analysis** : (Optional) This step calculates SHapley Additive exPlanations (SHAP) values for variable importance (CSV file and radar plot for top features) and plots ROC curves for all the models specified by the user.

- **Report generation** : This step aggregates all output plots from the previous steps and compiles them into a `.pdf` report.

## Inputs

> [!TIP]
User can run multiple dimensionality reduction methods on the input dataset, and skip the ML models (`skip_ML_models = true`) and directly, view the pdf report and access the results.

- **Required**
  - **`main.input_csv`** : [File] Input file in `.csv` format, includes a `Label` column, with each row representing a sample and each column representing a feature. An example of the `.csv` is shown below:
    | Label  | Protein1 | Protein2 | ... | ProteinN |
    |:-------|:---------|:---------|-----|:---------|
    | Label1 | 0.1      | 0.4      | ... | 0.01     |
    | Label2 | 0.2      | 0.1      | ... | 0.3      |
  - **`main.output_prefix`** : [String] Analysis ID. This will be used as prefix for all the output files.

> [!WARNING]
It is recommended to select only one dimensionality reduction method when using it alongside ML models. If multiple dimensionality reduction methods are specified, the pipeline will run only the specified methods and then proceed directly to the final report generation step.

- **Optional**
  - **`main.mode`** : [String] Specify the dimensionality method name(s) to use. Options include `Classification`, `Regression`, and `Summary`. Default value: `Summary`.
  - **`main.dimensionality_reduction_choices`** : [String] Specify the dimensionality method name(s) to use. Options include `PCA`, `UMAP`, `TSNE`, `KPCA` and `PLS`. Multiple methods can be entered together, separated by a space. Default value: `PCA`
  - **`main.num_of_dimensions`**: [Int] Total number of expected dimensions after applying dimensionality reduction. Default value: `3`.
  - **`main.skip_ML_models`** : [Boolean] Use this option to skip running ML models. Default value: `false`
  - **`main.classification_model_choices`** : [String] Specify the classification model name(s) to use. Options include `RF`, `KNN`, `NN`, `SVM`, `XGB`, `PLSDA`, `VAE`, `LR`, `GNB`, `LGBM` and `MLPVAE`. Multiple model names can be entered together, separated by a space. Default value: `RF`
  - **`main.regression_model_choices`** : [String] Specify the regression model name(s) to use. Options include `RF_reg`, `NN_reg`, `SVM_reg`, `XGB_reg`, `PLS_reg`, `KNN_reg`, `LGBM_reg`, `VAE_reg` and `MLPVAE_reg`. Multiple model names can be entered together, separated by a space. Default value: `RF_reg`
  - **`main.calculate_shap`**: [Boolean] Top features to display on the radar chart. Default value: `false`
  - **`main.shap_features`**: [Int] Top features to display on the radar chart. Default value: `10`
  - **`*.memory_gb`** : [Int] Amount of memory in GB needed to execute the specific task. Default value: `128`
  - **`*.cpu`** : [Int] Number of CPUs needed to execute the specific task. Default value: `64`

## Outputs

- `report` : [File] A `.pdf` file containing the final reports, including the plots generated through the analyses.
- `results` : [File] A `.gz` file containing the results and plots from all steps in the workflow.

## Components

| Package | License |
|:---------|:---------|
| [micromamba==1.5.5](www.github.com/mamba-org/mamba#micromamba) | BSD-3-Clause |
| [python](www.python.org/) | PSF/GPL-compat |
| [joblib](www.github.com/joblib/joblib) | BSD-3-Clause |
| [matplotlib](www.matplotlib.org) | PSF/BSD-compat |
| [numpy](www.numpy.org/) | BSD |
| [pandas](www.pandas.pydata.org/) | BSD 3-Clause |
| [scikit-learn](www.scikit-learn.org) | BSD-3-Clause |
| [xgboost](https://github.com/dmlc/xgboost) |  Apache-2.0 |
| [shap](https://github.com/shap/shap) |  MIT |
| [pillow](https://github.com/python-pillow/Pillow) |  Open Source HPND |
| [PyTorch](https://github.com/pytorch/pytorch) |  BSD |
| [Optuna](https://github.com/optuna) | MIT |
| [fpdf](https://github.com/reingart/pyfpdf) |  LGPL-3.0 |
| [seaborn](https://github.com/mwaskom/seaborn) |  BSD-3-Clause |
| [umap-learn](https://github.com/lmcinnes/umap) |  BSD-3-Clause |

## Citations

> Zhou, Y., Maurya, A., Deng, Y., & Taylor, A. (2024). A cloud-based proteomics ML workflow for biomarker discovery. Zenodo. [https://doi.org/10.5281/zenodo.13378490](https://doi.org/10.5281/zenodo.13378490)

If you use `proteomics-ML-workflow` for your analysis, please cite the Zenodo record for that specific version using the following DOI: [10.5281/zenodo.13378490](https://zenodo.org/doi/10.5281/zenodo.13378490).
