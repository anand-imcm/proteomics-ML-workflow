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
- **Classification** : This step applies the user defined machine learning models (`KNN` or `RF`, or both) to the preprocessed data and generates predictions, plots, and other relevant evaluation metrics for all the models.

## Inputs

The main inputs to the workflow are:

- **Required**
  - `input_csv` : [File] Input file in `.csv` format.
  - `output_prefix` : [String] Sample name. This will be used as prefix for all the output files.
  - `model` : [String] Specify the model name(s) to use. Options include `KNN` and `RF`. Multiple model names can be entered together, separated by a space.
- **Optional**
  - `preprocessing.cpu` : [Integer] Total number of CPUs to be used in the `preprocessing` step. Default value: `8`
  - `preprocessing.memory_gb` : [Integer] Total number of RAM to be used in the `preprocessing` step. Default value: `8`
  - `classification.cpu` : [Integer] Total number of CPUs to be used in the `classification` step. Default value: `16`
  - `classification.memory_gb` : [Integer] Total number of RAM to be used in the `classification` step. Default value: `16`

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

## Components

| Package | License |
|---------|---------|
| [micromamba==1.5.5](www.github.com/mamba-org/mamba#micromamba) | BSD-3-Clause |
| [python==3.10](www.python.org/) | PSF/GPL-compat |
| [joblib==1.4.2](www.github.com/joblib/joblib) | BSD-3-Clause |
| [matplotlib==3.9.1](www.matplotlib.org) | PSF/BSD-compat |
| [numpy==2.0.0](www.numpy.org/) | BSD |
| [pandas==2.2.2](www.pandas.pydata.org/) | BSD 3-Clause License |
| [scikit-learn==1.5.1](www.scikit-learn.org) | BSD-3-Clause |
