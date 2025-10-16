# BiomarkerML

[![Open](https://img.shields.io/badge/Open-Dockstore-blue)](https://dockstore.org/workflows/github.com/anand-imcm/proteomics-ML-workflow)&nbsp;&nbsp;
[![publish](https://img.shields.io/github/actions/workflow/status/anand-imcm/proteomics-ML-workflow/publish_gen.yml)](https://github.com/anand-imcm/proteomics-ML-workflow/releases)&nbsp;&nbsp;
![GitHub release (with filter)](https://img.shields.io/github/v/release/anand-imcm/proteomics-ML-workflow)&nbsp;&nbsp;
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17367501.svg)](https://doi.org/10.5281/zenodo.17367501)&nbsp;&nbsp;

> [!TIP]
> To import the workflow into your Terra workspace, click on the above Dockstore badge, and select 'Terra' from the 'Launch with' widget on the Dockstore workflow page.

**Background**: High-throughput affinity and mass-spectrometry-based proteomic studies of large clinical cohorts generate high-dimensional proteomic data useful for accelerated disease biomarker discovery. A powerful approach to realizing the potential of these big, complex, and non-linear data, whilst ensuring reproducible results, is to use automated machine learning (ML) and deep learning (DL) pipelines for their analysis. However, there remains a gap in comprehensive ML workflows tailored to proteomic biomarker discovery and designed for biomedical researchers who need pipelines to optimally self-configure and automatically avoid over-fitting.

**Findings**: We present BiomarkerML, a cloud-based workflow for automated, reproducible, and efficient ML/DL analysis of proteomic data for biomarker discovery, designed for novice-ML users and implemented in Python, R and Workflow Description Language (WDL). BiomarkerML: ingests proteomic and clinical data alongside sample labels; pre-processes data for model fitting and optionally performs dimensionality reduction and visualization; fits a catalogue of ML and DL classification and regression models; and calculates model performance metrics for model comparison. Next, the workflow applies mean SHapley Additive exPlanations (SHAP) to quantify the contribution of each protein to model predictions across all samples. Finally, proteins with high mean SHAP values, and their co-expressed protein network interactors, are identified as candidate biomarkers. Importantly, hyperparameters - configuration variables set prior to training models - are automatically fine-tuned via grid-search, and BiomarkerML employs weighted, nested cross-validation to avoid model over-fitting and data leakage.

**Conclusions**: BiomarkerML is scalable, provides a standardized, user-friendly interface, and streamlines analyses to ensure reproducibility of results. Overall, BiomarkerML is a significant advancement, enabling novice-ML researchers to use cutting-edge ML/DL tools to identify disease biomarkers in complex proteomic data.

**Keywords**:
machine learning, cloud-based workflow, classification and regression, proteomic biomarker identification

<!-- markdownlint-disable MD033 -->
<br>
<div align="center">
  <img src="assets/proteomics-ml-WDL_white.png" width="900" height="auto"">
</div>

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

- **Protein–Protein Interaction analysis** : (Optional) Biological functional analyses through protein–protein interaction network diagrams for top-ranked biomarkers and first-degree network expansions combining protein coexpression patterns to highlight functional connectivity.

- **Report generation** : This step aggregates all output plots from the previous steps and compiles them into a `.pdf` report.

## Installation (local)

> [!IMPORTANT]
> **This workflow is primarily designed for cloud-based platforms (e.g., Terra.bio, DNANexus, Verily) that support WDL workflows.**
>
> However, you can also run it locally using the Cromwell workflow management system.
>
> This workflow has also been tested locally on `Ubuntu 22.04` with `Docker v28.3.1` and `Cromwell v40`, running on a 12th Gen Intel Core i7-1270P with 32 GB RAM.
>
> ARM64 (`linux/arm64`) architecture is currently not supported.

### Requirements

- **Docker**
  - Please checkout the [Docker installation](https://docs.docker.com/get-docker/) guide.

- **Mamba package manager**
  - Please checkout the [mamba or micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) official installation guide.
  - We prefer `mamba` over [`conda`](https://docs.conda.io/en/latest/) since it is faster and uses `libsolv` to effectively resolve the dependencies.

### Steps

1. **Create a new environment with Cromwell**

   Using `mamba` (recommended):
   ```bash
   mamba create --name biomarkerml bioconda::cromwell
   ```

   Or, using `conda`:
   ```bash
   conda create --name biomarkerml -c bioconda cromwell
   ```

2. **Activate the environment**

   With `mamba`:
   ```bash
   mamba activate biomarkerml
   ```

   Or, with `conda`:
   ```bash
   conda activate biomarkerml
   ```

3. **Prepare your input file**

   - All workflow inputs must be specified in a JSON file.
   - Use the provided `example/inputs.json` file as a template. You can find this file in the `example/` directory of the repository.
   - Make a copy of `example/inputs.json` and edit it to specify your own input data file and desired output prefix. At a minimum, update these two fields:
      ```json
        "main.input_csv": "/full/path/to/your/input/data.csv",
        "main.output_prefix": "your_output_prefix"
        ```
   - Replace `/full/path/to/your/input/data.csv` with the absolute path to your CSV data file, and set `your_output_prefix` to a name you want for your analysis outputs.
   - You can adjust other parameters in the JSON file as needed. See the **Inputs** section below for descriptions of all available options.

4. **Run the workflow locally**

   ```bash
   cromwell run workflows/main.wdl -i example/inputs.json
   ```

## Inputs

- **`main.input_csv`** : [File] Input file in `.csv` format, includes a `Label` column, with each row representing a sample and each column representing a feature. An example of the `.csv` is shown below:
  SampleID | Label  | Protein1 | Protein2 | ... | ProteinN |
  |:-------|:-------|:---------|:---------|-----|:---------|
  | ID1 | Label1 | 0.1      | 0.4      | ... | 0.01     |
  | ID2 | Label2 | 0.2      | 0.1      | ... | 0.3      |

- **`main.output_prefix`** : [String] Analysis ID. This will be used as prefix for all the output files.

- **`main.mode`** : [String] Specify the mode of the analysis. Options include `Classification`, `Regression`, and `Summary`. Default value: `Summary`.

- **`main.dimensionality_reduction_choices`** : [String] Specify the dimensionality method name(s) to use. Options include `PCA`, `UMAP`, `TSNE`, `KPCA` and `PLS`. Multiple methods can be entered together, separated by a space. Default value: `PCA`

> [!WARNING]
> It is recommended to select only one dimensionality reduction method when using it alongside classification or regression models.
>
> If multiple dimensionality reduction methods are specified, the workflow will only perform the dimentinality reduction and generate a report.


- **`main.num_of_dimensions`**: [Int] Total number of expected dimensions after applying dimensionality reduction for the visualization. This option only works when multiple `dimensionality_reduction_choices` are selected. Default value: `3`.

- **`main.classification_model_choices`** : [String] Specify the classification model name(s) to use. Options include `RF`, `KNN`, `NN`, `SVM`, `XGB`, `PLSDA`, `VAE`, `LR`, `GNB`, `LGBM` and `MLPVAE`. Multiple model names can be entered together, separated by a space. Default value: `RF`

- **`main.regression_model_choices`** : [String] Specify the regression model name(s) to use. Options include `RF_reg`, `NN_reg`, `SVM_reg`, `XGB_reg`, `PLS_reg`, `KNN_reg`, `LGBM_reg`, `VAE_reg` and `MLPVAE_reg`. Multiple model names can be entered together, separated by a space. Default value: `RF_reg`

- **`main.calculate_shap`**: [Boolean] Top features to display on the radar/bar chart. Default value: `false`

- **`main.shap_features`**: [Int] Number of features to display on the radar/bar chart. Default value: `10`

- **`main.run_ppi`**: [Boolean] Execute Protein-Protein interaction (ppi) analysis. Default value: `false`

> [!WARNING]
> The Protein-Protein interaction analysis can be performed only when the `dimensionality_reduction_choices` option is set to either `ELASTICNET` or `NONE`, and `calculate_shap` option is set to `true`.

- **`main.ppi_analysis.score_threshold`** : [Int] Confidence score threshold for loading STRING database. Default value: `400`

- **`main.ppi_analysis.combined_score_threshold`** : [Int] Confidence score threshold for selecting nodes to plot in the network. Default value: `800`

- **`main.ppi_analysis.SHAP_threshold`** : [Int] The number of top important proteins selected for network analysis based on SHAP values. Default value: `100`

- **`main.ppi_analysis.protein_name_mapping`** : [Boolean] Whether to perform protein name mapping from UniProt IDs to Entrez Symbols. Default value: `TRUE`

- **`main.ppi_analysis.correlation_method`** : [String] Correlation method used to define strongly co-expressed proteins. Options include `spearman`, `pearson` and `kendall`. Default value: `spearman`

- **`main.ppi_analysis.correlation_threshold`** : [Float] Threshold value of the correlation coefficient used to identify strongly co-expressed proteins. Default value: `0.8`

- **`main.*.memory_gb`** : [Int] Amount of memory in GB needed to execute the specific task. Default value: `24`

- **`main.*.cpu`** : [Int] Number of CPUs needed to execute the specific task. Default value: `16`

> [!NOTE]
> We recommend that users adopt *unique Entrez Symbols* as the protein naming convention for our network analysis, although we provide an approach using the R/Bioconductor annotation package **`org.Hs.eg.db`** to map UniProt IDs to Entrez Symbols.
>
> The protein name mapping process handles edge cases as follows:
>
> - **UniProt IDs mapped to multiple Entrez symbols**: All matched Entrez symbols corresponding to the same UniProt ID are concatenated using a semicolon (`;`) and later deconcatenated during network plot mapping to STRINGdb. This may occur in cases where protein complexes are composed of    subunits encoded by different genes and etc.
>
> - **Multiple UniProt IDs mapping to the same Entrez symbol**: Only the first occurrence — corresponding to the protein with the highest SHAP value for that symbol — is retained in the final dataset. This may happen in cases involving protein isoforms, fusion proteins and etc.
>
> - **UniProt IDs with no associated Entrez symbol**: These entries are removed from the dataset.

## Outputs

- `report` : [File] A `.pdf` file containing the final reports, including the plots generated through the analyses.
- `results` : [File] A `.gz` file containing the results and plots from all steps in the workflow.

## Components

| Package | License |
|:--------|:--------|
| [micromamba==1.5.5](www.github.com/mamba-org/mamba#micromamba) | BSD-3-Clause |
| [python](www.python.org/) | PSF/GPL-compat |
| [joblib](www.github.com/joblib/joblib) | BSD-3-Clause |
| [matplotlib](www.matplotlib.org) | PSF/BSD-compat |
| [numpy](www.numpy.org/) | BSD |
| [pandas](www.pandas.pydata.org/) | BSD 3-Clause |
| [scikit-learn](www.scikit-learn.org) | BSD-3-Clause |
| [xgboost](https://github.com/dmlc/xgboost) | Apache-2.0 |
| [shap](https://github.com/shap/shap) | MIT |
| [pillow](https://github.com/python-pillow/Pillow) | Open Source HPND |
| [PyTorch](https://github.com/pytorch/pytorch) | BSD |
| [Optuna](https://github.com/optuna) | MIT |
| [fpdf](https://github.com/reingart/pyfpdf) | LGPL-3.0 |
| [seaborn](https://github.com/mwaskom/seaborn) | BSD-3-Clause |
| [umap-learn](https://github.com/lmcinnes/umap) | BSD-3-Clause |
| [AnnotationDbi](https://bioconductor.org/packages/release/bioc/html/AnnotationDbi.html) | Artistic-2.0 |
| [BiocManager](https://bioconductor.github.io/BiocManager/) | Artistic-2.0 |
| [fields](https://cran.r-project.org/web/packages/fields/index.html) | GPL (>= 2) |
| [ggplot2](https://ggplot2.tidyverse.org/index.html) | MIT |
| [igraph](https://r.igraph.org/) | GPL (>= 2) |
| [magrittr](https://magrittr.tidyverse.org/) | MIT |
| [optparse](https://cran.r-project.org/web/packages/optparse/index.html) | GPL (>= 2) |
| [STRINGdb](https://bioconductor.org/packages/release/bioc/html/STRINGdb.html) | GPL (>= 2) |
| [tidyverse](https://www.tidyverse.org/) | GPL-3 |
| [writexl](https://cloud.r-project.org/web/packages/writexl/index.html) | BSD-2-clause |
| [org.Hs.eg.db](https://bioconductor.org/packages/release/data/annotation/html/org.Hs.eg.db.html) | Artistic-2.0 |

## Citations

If you use `BiomarkerML` for your analysis, please cite the following Zenodo record:

> Zhou, Y., Maurya, A.K., Deng, Y., Fletcher, M.P., Ren, C., & Taylor, A. (2025). A cloud-based proteomics ML workflow for biomarker discovery. Zenodo. https://doi.org/10.5281/zenodo.17367501