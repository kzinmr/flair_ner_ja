FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

WORKDIR /app

ENV LANG=C.UTF-8

RUN apt-get update -y && apt-get install -y \
    git \
    wget \
    curl \
    cmake \
    unzip \
    gcc \
    g++

COPY requirements.txt .
# hadolint ignore=DL3013
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY ner.py .
RUN mkdir data
RUN mkdir models

RUN python ner.py


