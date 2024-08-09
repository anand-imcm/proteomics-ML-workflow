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
  - `input_csv` : [File] Input file in `.csv` format.
  - `output_prefix` : [String] Sample name. This will be used as prefix for all the output files.
  - `model_choices` : [String] Specify the model name(s) to use. Options include `KNN`, `RF`, `NN`, `XGB`, `PLSDA`, `VAE`, and `SVM`. Multiple model names can be entered together, separated by a space.
- **Optional**
  - `use_dimensionality_reduction` : [Boolean] Use this switch to apply dimensionality reduction to the input data. Default value: `false`
  - `skip_ML_models` : [Boolean] Use this switch to skip running ML models. Default value: `false`
  - `preprocessing_std.cpu` : [Integer] Total number of CPUs to be used in the `preprocessing_std` step. Default value: `8`
  - `preprocessing_std.memory_gb` : [Integer] Total number of RAM to be used in the `preprocessing_std` step. Default value: `8`
  - `preprocessing_dim.dim_reduction_method` : [String] Specify the model name(s) to use. Options include `PCA`, `UMAP`, `t-SNE`, `KPCA` and `PLS`. Multiple model names can be entered together, separated by a space. Default value: `PCA`
  - `preprocessing_dim.num_dimensions` : Int (optional, default = 3)
  - `preprocessing_dim.memory_gb` : Int (optional, default = 16)
  - `preprocessing_dim.cpu` : Int (optional, default = 16)
  - `classification_gen.cpu` : [Integer] Total number of CPUs to be used in the `classification_gen` step. Default value: `16`
  - `classification_gen.memory_gb` : [Integer] Total number of RAM to be used in the `classification_gen` step. Default value: `24`
  - `classification_vae.cpu` : [Integer] Total number of CPUs to be used in the `classification_vae` step. Default value: `16`
  - `classification_vae.memory_gb` : [Integer] Total number of RAM to be used in the `classification_vae` step. Default value: `24`
  - `plot.cpu` : [Integer] Total number of CPUs to be used in the `roc_plot` step. Default value: `16`
  - `plot.memory_gb` : [Integer] Total number of RAM to be used in the `roc_plot` step. Default value: `24`
  - `plot.shap_radar_num_features` : [Integer] Number of top features to display in radar chart. Default value: `10`
  - `pdf.memory_gb` : Int (optional, default = 24)
  - `pdf.cpu` : Int (optional, default = 16)

## Outputs

The main output files are listed below:

- **Preprocessing**
  - `processed_csv` : [File] A `.csv` file containing the standardized data.
- **Classification**
  - `confusion_matrix_plot` : Array[File] An array of files for confusion matrix plots, showing the prediction results for each class.
  - `roc_curve_plot` : Array[File] An array of files for ROC curve plots, indicating the model's discriminative ability, including ROC & AUC for each group and the overall performance ROC & AUC of the model.
  - `metrics_plot` : Array[File] An array of files for metrics plots, including evaluation metrics for each model: Accuracy, F1, Sensitivity, and Specificity.
  - `data_pkl` : Array[File] An array of pickle files containing data used for SHAP calculations.
  - `model_pkl` : Array[File] An array of pickle files containing the best model for SHAP calculations.
  - `data_npy` : Array[File]  An array of NumPy files containing overall ROC data for models, used for comparing the overall ROC of each model.
- **Combined ROC plot**
  - `overall_roc_plot` : [File] A `.png` file containing the ROC curves plot for all the models specified by the user.
  - `shap_radar_plot` : Array[File] An array of `.png` files with radar plots for each model specified by the user.
  - `shap_values` : Array[File] An array of `.csv` files with SHAP values for each model specified by the user.

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
