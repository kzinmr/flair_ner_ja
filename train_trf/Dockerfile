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
# hadolint ignore=DL3013
RUN pip install --upgrade pip && \
    pip install -r requirements.txt \
    pip install hydra-core --upgrade

COPY config.yaml .
COPY app.py .
RUN mkdir -p outputs/data
RUN mkdir -p outputs/models

CMD ["python", "app.py"]