FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# 使うソフトウェアのインストール
RUN apt-get update
RUN apt-get install -y python3 python3-pip
RUN apt-get install -y git curl

WORKDIR /
RUN git clone https://github.com/getuka/japanese-parler-tts.git
WORKDIR ./japanese-parler-tts

# ライブラリのインストール
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install packaging ninja
RUN pip install -r requirements.txt