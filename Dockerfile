FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip \
        git \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python libraries
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install --no-cache-dir \
        packaging \
        ninja \
        flash-attn --no-build-isolation \
        git+https://github.com/huggingface/parler-tts.git \
        git+https://github.com/getuka/RubyInserter.git \
        fastapi \
        pydantic \
        openai \
        uvicorn

# Create and set the working directory
WORKDIR /app
