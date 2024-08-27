# proteomics-ML-workflow

[![Open](https://img.shields.io/badge/Open-Dockstore-blue)](https://dockstore.org/workflows/github.com/anand-imcm/proteomics-ML-workflow)&nbsp;&nbsp;
[![publish](https://img.shields.io/github/actions/workflow/status/anand-imcm/proteomics-ML-workflow/publish_gen.yml)](https://github.com/anand-imcm/proteomics-ML-workflow/releases)&nbsp;&nbsp;
![GitHub release (with filter)](https://img.shields.io/github/v/release/anand-imcm/proteomics-ML-workflow)&nbsp;&nbsp;
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13378490.svg)](https://doi.org/10.5281/zenodo.13378490)&nbsp;&nbsp;


> [!TIP]
> To import the workflow into your Terra workspace, click on the above Dockstore badge, and select 'Terra' from the 'Launch with' widget on the Dockstore workflow page.

## Introduction

This cloud-based platform integrates key algorithms, including Principal Component Analysis (PCA), Kernel PCA (KPCA), Partial Least Squares (PLS), t-SNE, and UMAP, for robust pre-processing, visualization, and dimensionality reduction. Incorporating state-of-the-art machine learning and deep learning methods, such as Multilayer Perceptron (MLP), Random Forest (RF), Support Vector Machine (SVM), PLS Discriminant Analysis (PLSDA), XGBoost, K-Nearest Neighbors (KNN), and Variational Autoencoder (VAE-MLP) etc., the workflow ensures comprehensive data analysis. SHapley Additive exPlanations (SHAP) are used to quantify the significance of identified proteins, enhancing the interpretability of results. Functional enrichment and protein-protein interaction (PPI) network analyses are performed, focusing on visualization, to facilitate understanding of disease mechanisms. This workflow advances the early diagnosis and treatment of neurodegenerative diseases by enabling the efficient identification of critical biomarkers.

## Workflow Steps

- **Preprocessing** : By default, Z-score standardisation is applied to the input data. Optionally, users can choose to apply dimensionality reduction to the dataset. Display scatter plots for every two dimensions based on the selected number of output dimensions. The available methods include:
  - `PCA` (Principal Component Analysis for linear data)
  - `UMAP` (Uniform Manifold Approximation and Projection)
  - `t-SNE` (t-Distributed Stochastic Neighbor Embedding)
  - `KPCA` (Kernel Principal Component Analysis for non-linear data)
  - `PLS` (Partial Least Squares)

- **Classification** : This step applies the machine learning models to the standardized data and generates a confusion matrix, ROC plots for all classes and averages, and other relevant evaluation metrics (Accuracy, F1, sensitivity, specificity) for all the models. The available algorithms are as follows:
  - `KNN` (K-Nearest Neighbors)
  - `RF` (Random Forest)
  - `NN` (Neural Network)
  - `SVM` (Support Vector Machine)
  - `XGB` (XGBoost)
  - `PLSDA` (Partial Least Squares Discriminant Analysis)
  - `VAE` (Variational autoencoder)

- **SHAP summary** : This step calculates SHAP values for variable importance (CSV file and radar plot for top features) and plots ROC curves for all the models specified by the user.

- **Combined report** : This step aggregates all output plots from the previous steps and compiles them into a `.pdf` report.

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
It is recommended to select only one dimensionality reduction method when using it alongside ML models. Set the `skip_ML_models` option to `true` if applying multiple dimensionality reduction methods. If `skip_ML_models` is `false` while using multiple dimensionality reduction methods, the pipeline will automatically select one of the output files from the dimensionality reduction step for classification.

- **Optional**
  - **`main.use_dimensionality_reduction`** : [Boolean] Use this option to apply dimensionality reduction to the input data. Default value: `false`
  - **`main.num_of_dimensions`**: [Int] Total number of expected dimensions after applying dimensionality reduction. Default value: `3`.
  - **`main.skip_ML_models`** : [Boolean] Use this option to skip running ML models. Default value: `false`
  - **`main.model_choices`** : [String] Specify the model name(s) to use. Options include `KNN`, `RF`, `NN`, `XGB`, `PLSDA`, `VAE`, and `SVM`. Multiple model names can be entered together, separated by a space. Default value: `RF`
  - **`main.dimensionality_reduction_choices`** : [String] Specify the dimensionality method name(s) to use. Options include `PCA`, `UMAP`, `t-SNE`, `KPCA` and `PLS`. Multiple methods can be entered together, separated by a space. Default value: `PCA`
  - **`main.shap_radar_num_features`**: [Int] Top features to display on the radar chart. Default value: `10`
  - **`main.memory_*`** : [Int] Amount of memory in GB needed to execute the specific task. Default value: `32`
  - **`*main.cpu_*`** : [Int] Number of CPUs needed to execute the specific task. Default value: `16`

## Outputs

- `report` : [File] A `.pdf` file containing the result plots from all the required analyses.
- `plots` : [File] A `.gz` file containing the result plots from all the required analyses.
- `shap_csv` : Array[File] A list of `.csv` files containing SHAP values for each input variable.
- `std_preprocessing_csv` : [File] A `.csv` file with data standardized using the default method.
- `dimensionality_reduction_csv` : [File] A `.csv` file with the selected dimensional data using the user-selected dimensionality reduction method.
- `dimensionality_reduction_plots` : Array[File] A list of `.png` files with the selected dimensional output plots using the user-selected dimensionality reduction method.

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
| [tensorflow](https://github.com/tensorflow/tensorflow) |  Apache-2.0 |
| [fpdf](https://github.com/reingart/pyfpdf) |  LGPL-3.0 |
| [seaborn](https://github.com/mwaskom/seaborn) |  BSD-3-Clause |
| [umap-learn](https://github.com/lmcinnes/umap) |  BSD-3-Clause |

## Citations

> Zhou, Y., Maurya, A., Deng, Y., & Taylor, A. (2024). A cloud-based proteomics ML workflow for biomarker discovery. Zenodo. [https://doi.org/10.5281/zenodo.13378490](https://doi.org/10.5281/zenodo.13378490)

If you use `proteomics-ML-workflow` for your analysis, please cite the Zenodo record for that specific version using the following DOI: [10.5281/zenodo.13378490](https://zenodo.org/doi/10.5281/zenodo.13378490).
