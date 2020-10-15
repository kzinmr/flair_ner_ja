FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

WORKDIR /app

ENV LANG=C.UTF-8
ENV PORT=8080

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
    pip install -r requirements.txt && \
    pip install --no-cache-dir jsonlines && \
    pip install git+https://github.com/kzinmr/flair.git


RUN mkdir /app/model
COPY model/best-model.pt /app/model/best-model.pt
COPY app.py /app/app.py

RUN mkdir -p /app/data
COPY data/test.jsonl /app/data/test.jsonl

EXPOSE ${PORT}

CMD ["python", "app.py"]