# proteomics-ML-workflow

> [!WARNING]
> This project is under development and is not ready for production use.

## todo

- [x] Initial Layout
- [ ] Workflow description
- [ ] Describe Inputs
- [ ] Describe Outputs
- [ ] Usage instructions
- [x] List all the components and their licenses

## Workflow Steps

- **Preprocessing** : Performs Z-score normalization on the input data.
- **Classification** : This step applies the machine learning models to the standardized data and generates predictions, plots, and other relevant evaluation metrics for all the models. The available algorithms are as follows:
  - `KNN` (K-Nearest Neighbors)
  - `RF` (Random Forest)
  - `NN` (Neural Network)
  - `SVM` (Support Vector Machine)
  - `XGB` (XGBoost)
  - `PLSDA` (Partial Least Squares Discriminant Analysis)
  - `VAE` (Variational autoencoder)
- **Combined ROC plot** : This step generates a plot containing the ROC curves for all the models specified by the user.

## Inputs

The main inputs to the workflow are:

- **Required**
  - `main.input_csv` : [File] Input file in `.csv` format.
  - `main.output_prefix` : [String] Sample name. This will be used as prefix for all the output files.
  - `main.model_choices` : [String] Specify the model name(s) to use. Options include `KNN`, `RF`, `NN`, `XGB`, `PLSDA`, `VAE`, and `SVM`. Multiple model names can be entered together, separated by a space.
  - `main.method_name` : [String] Specify the model name(s) to use. Options include `PCA`, `UMAP`, `t-SNE`, `KPCA` and `PLS`. Multiple model names can be entered together, separated by a space. Default value: `PCA`
- **Optional**
  - `main.use_dimensionality_reduction` : [Boolean] Use this switch to apply dimensionality reduction to the input data. Default value: `false`
  - `main.skip_ML_models` : [Boolean] Use this switch to skip running ML models. Default value: `false`

## Outputs

- `report` : [File] A `.pdf` file containing the standardized data through the default method.
- `shap_csv` : Array[File] A list of `.csv` files containing SHAP values.
- `std_preprocessing_csv` : [File] A `.csv` file with data standardized using the default method.
- `dimensionality_reduction_csv` : [File] A `.csv` file with data standardized using the user-selected dimensionality reduction method.

## Components

| Package | License |
|---------|---------|
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
