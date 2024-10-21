FROM mambaorg/micromamba:1.5.5

USER root

RUN apt-get update && apt-get install -yq --no-install-recommends \
    libx11-6 libx11-dev tk tk-dev

USER $MAMBA_USER

COPY --chown=$MAMBA_USER:$MAMBA_USER ./container/reg/environment.yml /tmp/

COPY --chown=$MAMBA_USER:$MAMBA_USER ./scripts /scripts

RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1

# adding the path to the tools managed by mamba
ENV PATH="/opt/conda/bin:$PATH"