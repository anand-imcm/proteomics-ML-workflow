FROM rocker/r-ver:4.4

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV MAKEFLAGS=-j1

RUN apt-get update && apt-get install -y \
    wget libcurl4-openssl-dev libssl-dev libxml2-dev libgit2-dev libjpeg-dev libpng-dev libtiff-dev \
    zlib1g-dev libbz2-dev liblzma-dev liblapack-dev libfftw3-dev libhdf5-dev libpoppler-cpp-dev \
    libglpk-dev libboost-all-dev libncurses-dev xz-utils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/R && \
    echo "options(repos = c(CRAN = 'https://packagemanager.posit.co/cran/latest'))" > /etc/R/Rprofile.site

RUN Rscript -e "install.packages(c('BiocManager', 'optparse', 'tidyverse', 'magrittr', 'igraph', 'fields', 'ggplot2', 'writexl', 'pdftools'))"

RUN Rscript -e " \
  options(repos = BiocManager::repositories()); \
  BiocManager::install(version = '3.20', ask = FALSE, update = FALSE); \
  BiocManager::install(c('STRINGdb', 'org.Hs.eg.db', 'AnnotationDbi', 'ensembldb', 'EnsDb.Hsapiens.v75'), ask = FALSE, update = FALSE)"

COPY ./scripts/NetPath /scripts

RUN wget -q http://stringdb-downloads.org/download/protein.aliases.v12.0/9606.protein.aliases.v12.0.txt.gz -P /scripts && \
    wget -q http://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz -P /scripts && \
    wget -q http://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz -P /scripts