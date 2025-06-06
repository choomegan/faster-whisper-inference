FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04
WORKDIR /root
RUN apt-get update -y && apt-get install -y python3-pip

COPY requirements.txt ./requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt
