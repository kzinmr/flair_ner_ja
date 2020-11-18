# flair_ner_ja

## retokenize
- build: `docker build -t retokenize .`
- run: `docker run --rm -v /where/to/flair_ner_ja/retokenize/data:/app/data retokenize`

## train
- build: `docker build -t flair-ner-ja-train .`
- run: `docker run --rm --gpus all -v /where/to/flair_ner_ja/train/outputs:/app/outputs flair-ner-ja-train`

## predict (deployment)
- `cd deployment`
- build: `docker build -t flair-ner-ja-deployment .`
- run: `docker run --rm --gpus all -p 10122:8080 flair-ner-ja-deployment`

## predict (batch)
- `cd batch`
- build: `docker build -t flair-ner-ja-batch .`
- run: `docker run --rm --gpus all -v /where/to/flair_ner_ja/batch/data:/app/data flair-ner-ja-batch`
