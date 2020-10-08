from flair.datasets import ColumnCorpus
from flair.embeddings import StackedEmbeddings, FlairEmbeddings, PooledFlairEmbeddings
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from typing import List, Tuple, Dict, Iterable, Optional
import MeCab
import tokenizations
import os
from pathlib import Path
import jsonlines
import requests
import hydra
from omegaconf import DictConfig


class ConllConverter:
    """ Convert text and spans to CoNLL2003-like column data
    """

    def __init__(self):
        self.wakati = MeCab.Tagger("-Owakati")

    def tokenize(self, text: str) -> List[str]:
        return self.wakati.parse(text).split()

    @staticmethod
    def get_superspan(query_span: Tuple[int, int], superspans: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """ return superspan for given query span from set of superspans if any
        """
        for superspan in superspans:
            if query_span[0] >= superspan[0] and query_span[1] <= superspan[1]:
                return superspan
        return None

    @classmethod
    def get_token_labels(cls, token_spans: Tuple[int, int], chunk_spans: Tuple[int, int], chunk_labels: List[str]) -> List[str]:
        """ chunk単位のラベルから、token単位のラベルを構成
        """

        chunkspan2tagtype = dict(zip(chunk_spans, chunk_labels))

        # token_spansに含まれるchunk(span)を決定し、chunkのtagtypeも同時に記録
        target_token_spans = []
        tagtypes = []
        for token_span in token_spans:
            chunk_span = cls.get_superspan(token_span, chunk_spans)
            if chunk_span is not None and chunk_span in chunkspan2tagtype:
                target_token_spans.append(token_span)
                tagtypes.append(chunkspan2tagtype[chunk_span])
        tokenspan2tagtype = dict(zip(target_token_spans, tagtypes))

        # token に対応する label をchunkのtagtypeを基に構成
        label = 'O'
        token_labels = []
        for token_span in token_spans:
            if token_span in tokenspan2tagtype:
                tagtype = tokenspan2tagtype[token_span]
                if label == 'O':
                    label = f'B-{tagtype}'
                else:
                    label = f'I-{tagtype}'
            else:
                label = 'O'

            token_labels.append(label)
        return token_labels


    def tokenize_and_align_spans(self, text: str, chunk_spans: List[Tuple[int, int]], chunk_labels: List[str]) -> List[Tuple[str, str]]:

        # text -> tokens
        tokens = self.tokenize(text)

        # 各tokenがtextのどこにあるか(token_spans)を計算
        token_spans = tokenizations.get_original_spans(tokens, text)

        # 各tokenに対応するchunk NE-typeを同定(token-span vs chunk-span の包含関係計算)
        token_labels = self.get_token_labels(token_spans, chunk_spans, chunk_labels)

        # CoNLL2003-likeなtoken行単位の列データを返す
        token_label_columns = [(token, label) for token, label in zip(tokens, token_labels)]
        # 計算したspanが合ってるかのassertion
        spannedtoken_label_columns = [(text[span[0]:span[1]], label) for span, label in zip(token_spans, token_labels)]
        assert spannedtoken_label_columns == token_label_columns

        return token_label_columns

def download_conll_data(filepath):
    """ conllフォーマットデータのダウンロード
    """
    url = 'https://raw.githubusercontent.com/Hironsan/IOB2Corpus/master/ja.wikipedia.conll'
    response = requests.get(url)
    if response.ok:
        with open(filepath, "w") as fp:
            fp.write(response.content.decode('utf8'))
        return filepath

def make_sample_conll_corpus(data_folder):
    """ conllフォーマットデータのダウンロード
    """
    train_file = 'ja.wikipedia.conll'
    filepath = data_folder / train_file
    if download_conll_data(filepath):
        columns = {0: 'text', 1: 'ner'}
        corpus = ColumnCorpus(data_folder, columns,
                              train_file=train_file)

        return corpus

def make_conll(cc, reader):
    conll = []
    for text, entd in reader.iter():
        labels = [label for _, _, label in entd['entities']]
        spans = [(s, e) for s, e, _ in entd['entities']]
        tokens_labels = cc.tokenize_and_align_spans(text, spans, labels)
        conll.append("\n".join([f"{token}\t{label}" for (token, label) in tokens_labels]))
    return conll

def make_conll_corpus(data_folder):
    """ camphr形式からtokenize & conllフォーマットに変換し、Corpus化
    """
    cc = ConllConverter()
    train_file = 'train.jsonl'
    test_file = 'test.jsonl'
    with jsonlines.open(data_folder / train_file, 'r') as reader:
        conll = make_conll(cc, reader)
    train_file = 'train.conll'
    with open(data_folder / train_file, 'w') as fp:
        fp.write("\n\n".join(conll))

    with jsonlines.open(data_folder / test_file, 'r') as reader:
        conll = make_conll(cc, reader)
    test_file = 'test.conll'
    with open(data_folder / test_file, 'w') as fp:
        fp.write("\n\n".join(conll))

    columns = {0: 'text', 1: 'ner'}
    corpus = ColumnCorpus(data_folder, columns,
                          train_file=train_file,
                          test_file=test_file)

    return corpus

def make_tagger(corpus, cfg):
    tag_type = 'ner'
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary.idx2item)

    # Contextual string embeddings of words, as proposed in Akbik et al., 2018.
    # using a character-level language model
    # Trained with 439M words of Japanese Web crawls
    embedding_types = [
        PooledFlairEmbeddings('ja-forward'),
        PooledFlairEmbeddings('ja-backward'),
        # FlairEmbeddings('ja-forward'),
        # FlairEmbeddings('ja-backward'),
    ]
    embeddings = StackedEmbeddings(embeddings=embedding_types)

    tagger = SequenceTagger(hidden_size=cfg.model.hidden_size,
                            dropout=cfg.model.dropout,
                            word_dropout=cfg.model.word_dropout,
                            locked_dropout=cfg.model.locked_dropout,
                            embeddings=embeddings,
                            tag_dictionary=tag_dictionary,
                            tag_type=tag_type,
                            use_crf=True)
    return tagger

@hydra.main(config_name="config")
def main(cfg: DictConfig):
    print(f'Current working directory: {os.getcwd()}')
    print(f'Orig working directory : {hydra.utils.get_original_cwd()}')
    print(f'to_absolute_path("outputs/data") : {hydra.utils.to_absolute_path("outputs/data")}')

    data_folder = Path(hydra.utils.get_original_cwd()) / 'outputs/data'
    corpus = make_sample_conll_corpus(data_folder)
    tagger = make_tagger(corpus, cfg)
    trainer = ModelTrainer(tagger, corpus)
    model_dir = Path(hydra.utils.get_original_cwd()) / 'outputs/models'
    trainer.train(model_dir,
                  learning_rate=cfg.training.learning_rate,
                  mini_batch_size=cfg.training.mini_batch_size,
                  max_epochs=cfg.training.max_epochs,
                  patience=cfg.training.patience,
                  initial_extra_patience=cfg.training.initial_extra_patience,
                  anneal_factor=cfg.training.anneal_factor,
                  shuffle=cfg.training.shuffle,
                  train_with_dev=cfg.training.train_with_dev,
                  batch_growth_annealing=cfg.training.batch_growth_annealing,
                  )

if __name__ == '__main__':
    main()