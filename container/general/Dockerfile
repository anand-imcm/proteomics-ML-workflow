FROM mambaorg/micromamba:1.5.5

USER $MAMBA_USER

COPY --chown=$MAMBA_USER:$MAMBA_USER ./container/general/environment.yml /tmp/

COPY --chown=$MAMBA_USER:$MAMBA_USER ./scripts /scripts

RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1

# adding the path to the tools managed by mamba
ENV PATH="/opt/conda/bin:$PATH"