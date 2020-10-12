# FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8-slim
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
    g++ \
    mecab \
    libmecab-dev mecab-ipadic-utf8
RUN ln -s /etc/mecabrc /usr/local/etc/mecabrc

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt


RUN mkdir /app/model
COPY model/best-model.pt /app/model/best-model.pt
COPY app.py /app/app.py

ENV PORT=8080
EXPOSE ${PORT}
# EXPOSE 80

CMD exec uvicorn --host 0.0.0.0 --port ${PORT} app:app