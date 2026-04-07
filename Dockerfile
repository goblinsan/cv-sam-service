FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-pip \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CUDA 12.x support first (large layer, cache separately)
RUN python3 -m pip install --no-cache-dir \
        torch \
        torchvision \
        --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 5201

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5201"]
