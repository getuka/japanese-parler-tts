FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# 使うソフトウェアのインストール
RUN apt-get update
RUN apt-get install -y python3 python3-pip
RUN apt-get install -y git curl

# ライブラリのインストール
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install packaging ninja
RUN MAX_JOBS=4 pip install flash-attn --no-build-isolation
RUN pip install git+https://github.com/huggingface/parler-tts.git
RUN pip install git+https://github.com/getuka/RubyInserter.git
RUN pip install fastapi
RUN pip install pydantic
RUN pip install openai
RUN pip install uvicorn

RUN mkdir /app
WORKDIR /app