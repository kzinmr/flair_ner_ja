FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8-slim

WORKDIR /app

ENV LANG=C.UTF-8
ENV PORT=8080
EXPOSE ${PORT}

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

CMD exec uvicorn --host 0.0.0.0 --port ${PORT} app:app