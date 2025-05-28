options(repos = BiocManager::repositories())

required_cran_packages <- c("BiocManager", "optparse", "tidyverse", "magrittr", "igraph", "fields", "ggplot2", "writexl", "pdftools")
required_bioconductor_packages <- c("STRINGdb", "org.Hs.eg.db", "AnnotationDbi")

missing_cran_packages <- required_cran_packages[!(required_cran_packages %in% installed.packages()[, "Package"])]
missing_bioconductor_packages <- required_bioconductor_packages[!(required_bioconductor_packages %in% installed.packages()[, "Package"])]

if (length(missing_cran_packages)) {
    install.packages(missing_cran_packages, dependencies = TRUE)
}

BiocManager::install(version = "3.21")

if (length(missing_bioconductor_packages)) {
    BiocManager::install(missing_bioconductor_packages, dependencies = TRUE)
}