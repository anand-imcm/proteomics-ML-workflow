options(repos = c(CRAN = "https://cran.r-project.org"))

# required_cran_packages <- c("BiocManager", "graphics", "grDevices", "Matrix", "methods", "stats", "utils", "viridisLite", "optparse", "dplyr", "magrittr")
# required_bioconductor_packages <- c("GENIE3", "STRINGdb", "igraph", "fields", "ggplot2", "clusterProfiler", "enrichplot", "org.Hs.eg.db", "gridExtra", "grid")
required_cran_packages <- c("BiocManager", "optparse", "tidyverse", "magrittr", "igraph", "fields", "ggplot2", "writexl", "pdftools")
required_bioconductor_packages <- c("STRINGdb", "igraph", "biomaRt")

missing_cran_packages <- required_cran_packages[!(required_cran_packages %in% installed.packages()[, "Package"])]
missing_bioconductor_packages <- required_bioconductor_packages[!(required_bioconductor_packages %in% installed.packages()[, "Package"])]

if (length(missing_cran_packages)) {
    install.packages(missing_cran_packages, dependencies = TRUE)
}

if (length(missing_bioconductor_packages)) {
    BiocManager::install(missing_bioconductor_packages, dependencies = TRUE, update = TRUE)
}
