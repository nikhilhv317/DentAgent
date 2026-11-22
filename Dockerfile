# DentAgent — app image.
#
# What's baked into this image: Python + all dependencies + your source code
# (app.py, agents/, ml/, static/, etc.).
#
# What is NOT baked in — mounted as volumes at runtime instead (see
# docker-compose.yml): the SQLite database, the trained noshow.pkl model
# registry, and your trained .pt vision weights (gate_best.pt, gate_meta.json,
# detector_best.pt). Baking those into the image would mean every code change
# requires re-shipping your trained models, and every retrain requires
# rebuilding the image — neither should be true. Data and code have separate
# lifecycles; this Dockerfile only owns the code's.

FROM python:3.11-slim

# System libraries some of torchvision/ultralytics/opencv's image codecs need
# at import time, even in "headless" mode. libgl1 specifically is what
# opencv-python (pulled in by ultralytics) needs just to IMPORT — without it,
# `from ultralytics import YOLO` fails with a libGL.so.1 error that the app's
# own except-ImportError handler mislabels as "install ultralytics", even
# though the package is genuinely installed. curl is only for the HEALTHCHECK.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Torch/torchvision installed separately from the CPU-only wheel index — the
# default PyPI wheel pulls in CUDA libraries (multiple GB) you don't need for
# CPU inference on a small EC2 instance. This alone is the difference between
# a ~1.5GB image and a ~5GB one.
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Declared so `docker run -v` / compose volumes attach cleanly even if you
# run this image directly without compose. compose.yml sets the actual
# mounts and the matching env var (DPMS_DB_PATH).
VOLUME ["/app/data", "/app/models", "/app/vision"]

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
