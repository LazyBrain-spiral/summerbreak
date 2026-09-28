# syntax=docker/dockerfile:1
# PS158 v3 runtime: video -> georeferenced, metric, textured 3D model + dashboard.
#
#   docker compose up --build                 # NVIDIA GPU host, dashboard on http://localhost:8765
#   docker compose --profile cpu up --build app-cpu   # any x86-64 / arm64 host, no GPU (slow, no SURVEY lane)
#
# Plain docker:
#   docker build -t ps158 .
#   docker run --gpus all -p 8765:8765 -v ps158-runs:/app/runs -v ps158-cache:/cache ps158
#   docker run --rm --gpus all -v "$PWD:/data" -v ps158-cache:/cache ps158 \
#       python -m src.pipeline --config configs/live.yaml --video /data/flight.mp4 --run-id flight
#
# Runs on: Linux, Windows (Docker Desktop + WSL2) and macOS (Docker Desktop, CPU only).
# GPU needs an NVIDIA driver >= 570 on the host and the NVIDIA Container Toolkit
# (built in to Docker Desktop on Windows). Without a GPU the server switches to CPU
# automatically and hides the lanes that need one.
FROM ubuntu:22.04

ARG TARGETARCH
# PyTorch wheel index. Empty = cu128 on amd64 (Turing..Blackwell GPUs, CPU also works), cpu on arm64.
ARG TORCH_INDEX=""
# Pi3 (BSD licence), pinned to the commit the benchmark numbers were produced with.
ARG PI3_REPO=https://github.com/yyfz/Pi3.git
ARG PI3_COMMIT=9fa3ddb3f8d53041f8b2738df404f62223bbaa7b

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    MPLBACKEND=Agg

# OS libraries: open3d (GL/EGL/gomp/usb), opencv (glib), git/curl for the installers.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git libgl1 libegl1 libgomp1 libglib2.0-0 libusb-1.0-0 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Miniforge + COLMAP 4.0.4 from conda-forge. On amd64 we force the CUDA build
# (CONDA_OVERRIDE_CUDA lets the solver pick it on a build machine without a GPU);
# it still runs on CPU when no GPU is present at run time.
# libfaiss is pinned: 1.14 breaks colmap 4.0.4 ("undefined symbol faiss::IndexIVFFlat").
RUN curl -fsSL -o /tmp/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh" \
    && bash /tmp/miniforge.sh -b -p /opt/conda && rm /tmp/miniforge.sh \
    && if [ "$(uname -m)" = "x86_64" ]; then \
           export CONDA_OVERRIDE_CUDA=12.9; COLMAP_SPEC="colmap=4.0.4=*cuda*"; \
       else COLMAP_SPEC="colmap=4.0.4"; fi \
    && /opt/conda/bin/conda create -y -n drone3d -c conda-forge python=3.11 "$COLMAP_SPEC" "libfaiss=1.10.0" \
    && /opt/conda/bin/conda clean -afy \
    && /opt/conda/envs/drone3d/bin/colmap help | head -3
ENV PATH=/opt/conda/envs/drone3d/bin:/opt/conda/bin:$PATH \
    CONDA_DEFAULT_ENV=drone3d

# PyTorch first, so nothing below pulls a different torch build from PyPI.
RUN idx="$TORCH_INDEX"; \
    if [ -z "$idx" ]; then \
        if [ "$(uname -m)" = "x86_64" ]; then idx=https://download.pytorch.org/whl/cu128; \
        else idx=https://download.pytorch.org/whl/cpu; fi; fi; \
    pip install --upgrade pip && pip install torch torchvision --index-url "$idx" \
    && python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# Python stack, pinned to the versions the pipeline was validated with.
COPY requirements-v3.txt docker/constraints.txt /tmp/
RUN pip install -r /tmp/requirements-v3.txt -c /tmp/constraints.txt \
    # simple-lama-inpainting pins ancient numpy/pillow; its code runs fine with ours
    && pip install --no-deps "simple-lama-inpainting==0.1.2" \
    && python -c "import open3d, pycolmap, rasterio, transformers, simple_lama_inpainting; print('python stack ok')"

# Pi3X multi-view depth (optional lane). Weights download on first use into /cache.
RUN git clone --filter=blob:none "$PI3_REPO" /opt/Pi3 \
    && git -C /opt/Pi3 checkout -q "$PI3_COMMIT" && rm -rf /opt/Pi3/.git \
    && PYTHONPATH=/opt/Pi3 python -c "from pi3.models.pi3x import Pi3X; print('pi3 ok')"
ENV PI3_DIR=/opt/Pi3

# Model caches (Hugging Face, torch hub / LaMa) live on a volume so they survive
# container rebuilds. Prefetch for offline use: python scripts/prefetch_models.py
ENV HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    YOLO_CONFIG_DIR=/cache/ultralytics \
    HOST=0.0.0.0 \
    PORT=8765
VOLUME ["/cache", "/app/runs"]

WORKDIR /app
COPY . /app
RUN mkdir -p /app/runs/uploads /cache && python scripts/check_env.py --profile core --allow-cpu || true

EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,os; urllib.request.urlopen('http://127.0.0.1:%s/api/health' % os.environ.get('PORT','8765'), timeout=4)"
CMD ["python", "webapp/server_v3.py"]
