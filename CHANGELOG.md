# Changelog

All notable changes to this project will be documented in this file.

## [1.0.2] - 2025-07-04

### Added

- New Protein-Protein Interaction (PPI) module.
- Additional docker image to handle the dependencies related to PPI module.

### Fixed

- Edge cases to handle protein IDs in the PPI module.
- Conditional flow of execution for the PPI module based on `ELASTICNET` and `NONE`.
- Multi-arch image for `biomarkerml-gen`.
- The `biomarkerml-net` image is only supported on `linux/amd64` because the `bioconductor-org.hs.eg.db` package is currently not available for `linux/arm64` builds on Bioconda or from the R community.


### Updated

- Simplified the inputs and default parameters.
- Usage guide includes the instruction to setup the workflow locally.

## [1.0.1] - 2024-08-27

### Fixed

- Missing Overall ROC curve and SHAP Radar plot in the PDF report.
- The report generation script had the wrong prefix, which prevented the PNG files from being included in the PDF report.

## [1.0.0] - 2024-08-23

### Release Notes

This is a WDL (Workflow Description Language) based workflow designed to be executed on a variety of compute platforms such as Terra.bio.

This release includes the following components of the workflow:

- **Preprocessing**: Z-score standardization and optional dimensionality reduction using PCA, UMAP, t-SNE, KPCA, and PLS.
- **Classification**: Machine learning models including KNN, RF, NN, SVM, XGB, PLSDA, and VAE with comprehensive evaluation metrics.
- **SHAP Summary**: Calculation of SHAP values for variable importance and ROC curve plotting.
- **Combined Report**: Aggregation of all output plots into a single `.pdf` report.
