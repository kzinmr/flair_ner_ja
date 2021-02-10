FROM ubuntu:20.04

ENV LANG C.UTF-8
ENV APP_ROOT /app

ENV DEBIAN_FRONTEND noninteractive

RUN mkdir -p $APP_ROOT
WORKDIR $APP_ROOT

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
    git \
    make \
    cmake \
    curl \
    xz-utils \
    file \
    sudo \
    build-essential \
    software-properties-common \
    mecab \
    libmecab-dev \
    mecab-ipadic-utf8 \
    python3.8 \
    python3.8-dev \
    python3-pip \
    python3.8-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN ln -s /etc/mecabrc /usr/local/etc/mecabrc

RUN pip3 install -U pip
COPY ./requirements.txt .
RUN pip install -r requirements.txt

RUN mkdir ${APP_ROOT}/data
COPY data/*.txt data/*.bio data/*.conll data/
COPY *.py ./

CMD ["python3", "retokenize.py", "--download_file"]