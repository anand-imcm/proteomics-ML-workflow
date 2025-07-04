FROM mambaorg/micromamba:1.5.5

USER root

RUN apt-get update && \
  apt-get install -y \
  wget && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

USER $MAMBA_USER

COPY --chown=$MAMBA_USER:$MAMBA_USER ./container/netpath/environment.yml /tmp/

COPY --chown=$MAMBA_USER:$MAMBA_USER ./scripts /scripts

RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes

RUN wget -q http://stringdb-downloads.org/download/protein.aliases.v12.0/9606.protein.aliases.v12.0.txt.gz -P /scripts && \
  wget -q http://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz -P /scripts && \
  wget -q http://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz -P /scripts

ARG MAMBA_DOCKERFILE_ACTIVATE=1

ENV PATH="/opt/conda/bin:$PATH"