#
# CUDA-enabled Dockerfile for running FastAPI + PyTorch on GPU.
# Assumes an AMD (or Intel) CPU host with an NVIDIA GPU and
# NVIDIA Container Toolkit installed on the server.
#

FROM --platform=linux/amd64 pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /app

# System dependencies for scientific stack, OpenCV, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies, but keep CUDA-enabled torch/torchvision
COPY requirements.txt .
RUN pip install --upgrade pip && \
    grep -Ev '^(torch|torchvision)' requirements.txt > /tmp/requirements.no-torch.txt && \
    pip install -r /tmp/requirements.no-torch.txt

# Copy application code
COPY . .

EXPOSE 8000

# Use all visible GPUs by default (can be overridden at runtime)
ENV CUDA_VISIBLE_DEVICES=0

# Start the FastAPI app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

