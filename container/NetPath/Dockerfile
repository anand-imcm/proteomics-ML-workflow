# Use the official R image with R 4.4 as the base image
FROM r-base:4.3.0

# Install system dependencies for the R packages
RUN apt-get update && apt-get install -y \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    liblapack-dev \
    libboost-all-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install BiocManager
RUN R -e "install.packages('BiocManager', repos='https://cloud.r-project.org/')"

# Install CRAN packages
RUN R -e "install.packages(c('graphics', 'grDevices', 'Matrix', 'methods', 'stats', 'utils', 'viridisLite', 'optparse', 'dplyr', 'magrittr'), dependencies=TRUE, repos='https://cloud.r-project.org/')"

# Install Bioconductor packages and other R packages
RUN R -e "BiocManager::install(c('GENIE3', 'STRINGdb', 'igraph', 'fields', 'ggplot2', 'clusterProfiler', 'enrichplot', 'org.Hs.eg.db', 'gridExtra', 'grid'), dependencies=TRUE, update=TRUE)"