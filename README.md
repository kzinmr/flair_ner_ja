# flair_ner_ja

## train
- build: `docker build -t flair-ner-ja .`
- run: `docker run --rm --gpus all -v /where/to/flair_ner_ja/outputs:/app/outputs flair-ner-ja`

## predict (serving)
- `cd deployment`
- build: `docker build -t flair-ner-ja-deployment .`
- run: `docker run --rm --gpus all -p 10122:8080 flair-ner-ja-deployment`
