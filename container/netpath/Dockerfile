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

RUN Rscript /scripts/install_packages.R