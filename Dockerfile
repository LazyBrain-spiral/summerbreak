# v3 runtime image: CUDA COLMAP from the official image plus the Python stack.
#   docker build -t drone3d .
#   docker run --gpus all -v "$PWD:/work" drone3d python -m src.pipeline --config configs/live.yaml ...
FROM colmap/colmap:latest
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-venv ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements-v3.txt /tmp/requirements-v3.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements-v3.txt \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
WORKDIR /work
CMD ["python", "scripts/check_env.py", "--profile", "live"]
