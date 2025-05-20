FROM rocker/tidyverse:4.3

RUN apt-get update && apt-get install -y \
    build-essential \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    liblapack-dev \
    libboost-all-dev \
    libpoppler-cpp-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY ./scripts/NetPath /scripts

RUN Rscript /scripts/install_packages.R \
    && wget -q http://stringdb-downloads.org/download/protein.aliases.v12.0/9606.protein.aliases.v12.0.txt.gz -P /scripts \
    && wget -q http://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz -P /scripts \
    && wget -q http://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz -P /scripts