from flair.datasets import ColumnCorpus
from flair.embeddings import StackedEmbeddings, FlairEmbeddings
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

import requests
from pathlib import Path

def download_conll_data(filepath):
    """ conllフォーマットデータのダウンロード
    """
    url = 'https://raw.githubusercontent.com/Hironsan/IOB2Corpus/master/ja.wikipedia.conll'
    response = requests.get(url)
    if response.ok:
        with open(filepath, "w") as fp:
            fp.write(response.content.decode('utf8'))
        return filepath

def make_conll_corpus(dirname):
    """ conllフォーマットデータのダウンロード
    """
    data_folder = Path(dirname)
    train_file = 'ja.wikipedia.conll'
    filepath = data_folder / train_file
    if download_conll_data(filepath):
        columns = {0: 'text', 1: 'ner'}
        corpus = ColumnCorpus(data_folder, columns,
                              train_file=train_file)

        return corpus

def make_tagger(corpus):
    tag_type = 'ner'
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary.idx2item)

    # Contextual string embeddings of words, as proposed in Akbik et al., 2018.
    # using a character-level language model
    # Trained with 439M words of Japanese Web crawls
    embedding_types = [
        FlairEmbeddings('ja-forward'),
        FlairEmbeddings('ja-backward'),
    ]
    embeddings = StackedEmbeddings(embeddings=embedding_types)

    tagger = SequenceTagger(hidden_size=256,
                            embeddings=embeddings,
                            tag_dictionary=tag_dictionary,
                            tag_type=tag_type,
                            use_crf=True)
    return tagger

def load_tagger_model(model_dir):
    model_path = model_dir / 'final-model.pt'
    model = SequenceTagger.load(model_path)
    return model

if __name__ == '__main__':
    dirname = 'data'
    corpus = make_conll_corpus(dirname)
    tagger = make_tagger(corpus)
    trainer = ModelTrainer(tagger, corpus)
    model_dir = Path('models')
    trainer.train(model_dir,
                learning_rate=0.1,
                mini_batch_size=32,
                max_epochs=150)