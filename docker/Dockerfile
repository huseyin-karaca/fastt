# Levels 1 and 2 of the reproduction in a container: the package, the notebooks, a CPU torch.
#
#   docker pull ghcr.io/huseyin-karaca/fastt:cpu        # or: docker build -f docker/Dockerfile -t fastt:cpu .
#   docker run --rm -v "$PWD/out:/out" ghcr.io/huseyin-karaca/fastt:cpu                  # AMI, the record in ./out
#   docker run --rm -v "$PWD/out:/out" -v "$PWD/cache:/cache" ghcr.io/huseyin-karaca/fastt:cpu \
#       python -m fastt.experiment voxpopuli --level 2
FROM python:3.12-slim
LABEL org.opencontainers.image.source="https://github.com/huseyin-karaca/fastt" \
      org.opencontainers.image.description="FASTT reproducibility package, levels 1 and 2 (CPU)"

ENV PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=1 FASTT_CACHE=/cache FASTT_RECORDS_DIR=/out
RUN apt-get update && apt-get install -y --no-install-recommends git libgomp1 && rm -rf /var/lib/apt/lists/*
RUN python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

WORKDIR /fastt
COPY . .
RUN python -m pip install ".[paper]" jupyter

VOLUME ["/out", "/cache"]
WORKDIR /out
CMD ["python", "-m", "fastt.experiment", "ami_sdm"]
