from flair.datasets import ColumnCorpus
from flair.embeddings import StackedEmbeddings, FlairEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
import torch
from typing import List, Tuple, Optional, Iterable
import MeCab
import os
from pathlib import Path
import jsonlines
import requests
import hydra
from omegaconf import DictConfig
import tokenizations
import iobes
import pickle

from dataclasses import dataclass


@dataclass(frozen=True)
class Token:
    text: str
    label: str


@dataclass(frozen=True)
class Chunk:
    tokens: List[Token]
    label: str
    span: List[Tuple[int, int]]

    def __iter__(self):
        for token in self.tokens:
            yield token


@dataclass(frozen=True)
class Sentence(Iterable[Token]):
    tokens: List[Token]
    chunks: List[Chunk]

    def __init__(self, tokens):
        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(self, "text", "".join([token.text for token in self.tokens]))
        object.__setattr__(self, "chunks", self.__build_chunks(self.tokens))
        self.assert_spans()

    def __iter__(self):
        for token in self.tokens:
            yield token

    def __build_chunks(self, tokens: List[Token]) -> List[Chunk]:
        chunks = self.__chunk_tokens(tokens)
        chunk_spans = self.__chunk_span(tokens)
        return [
            Chunk(
                tokens=chunk_tokens,
                label=chunk_tokens[0].label.split("-")[1],
                span=chunk_span,
            )
            for chunk_tokens, chunk_span in zip(chunks, chunk_spans)
        ]

    @staticmethod
    def __chunk_tokens(tokens: List[Token]) -> List[List[Token]]:
        chunks = []
        chunk = []
        for token in tokens:
            if token.label.startswith("B"):
                if chunk:
                    chunks.append(chunk)
                    chunk = []
                chunk = [token]
            elif token.label.startswith("I"):
                chunk.append(token)
            elif chunk:
                chunks.append(chunk)
                chunk = []
        return chunks

    @staticmethod
    def __chunk_span(tokens: List[Token]) -> List[Tuple[int, int]]:
        pos = 0
        spans = []
        chunk_spans = []
        for token in tokens:

            token_len = len(token.text)
            span = (pos, pos + token_len)
            pos += token_len

            if token.label.startswith("B"):
                # I->B
                if len(spans) > 0:
                    chunk_spans.append((spans[0][0], spans[-1][1]))
                    spans = []
                spans.append(span)
            elif token.label.startswith("I"):
                spans.append(span)
            elif len(spans) > 0:
                # B|I -> O
                chunk_spans.append((spans[0][0], spans[-1][1]))
                spans = []

        return chunk_spans

    def assert_spans(self):
        for chunk in self.chunks:
            assert self.text[chunk.span[0] : chunk.span[1]] == "".join(
                [t.text for t in chunk]
            )


def sentences_to_necolumns(sentences_str):
    sentences_columns = []
    for s in sentences_str:
        rows = []
        for t in s.splitlines():
            if len(t.split("\t")) > 1:
                token = t.split("\t")[1]
                details = dict(
                    [
                        l.split("=")
                        for l in t.split("\t")[-1].split("|")
                        if len(l.split("=")) == 2
                    ]
                )
                rows.append("\t".join([token, details.get("NE", "O")]))
        sentences_columns.append(rows)
    return sentences_columns


class ConllConverter:
    """Convert text and spans to CoNLL2003-like column data"""

    def __init__(self):
        self.wakati = MeCab.Tagger("-Owakati")

    def tokenize(self, text: str) -> List[str]:
        return self.wakati.parse(text).split()

    @staticmethod
    def get_superspan(
        query_span: Tuple[int, int], superspans: List[Tuple[int, int]]
    ) -> Optional[Tuple[int, int]]:
        """return superspan for given query span from set of superspans if any"""
        for superspan in superspans:
            if query_span[0] >= superspan[0] and query_span[1] <= superspan[1]:
                return superspan
        return None

    @classmethod
    def get_token_labels(
        cls,
        token_spans: Tuple[int, int],
        chunk_spans: Tuple[int, int],
        chunk_labels: List[str],
    ) -> List[str]:
        """chunk単位のラベルから、token単位のラベルを構成"""

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
        label = "O"
        token_labels = []
        for token_span in token_spans:
            if token_span in tokenspan2tagtype:
                tagtype = tokenspan2tagtype[token_span]
                if label == "O":
                    label = f"B-{tagtype}"
                else:
                    label = f"I-{tagtype}"
            else:
                label = "O"

            token_labels.append(label)
        return token_labels

    def tokenize_and_align_spans(
        self, text: str, chunk_spans: List[Tuple[int, int]], chunk_labels: List[str]
    ) -> List[Tuple[str, str]]:

        # text -> tokens
        tokens = self.tokenize(text)

        # 各tokenがtextのどこにあるか(token_spans)を計算
        token_spans = tokenizations.get_original_spans(tokens, text)

        # 各tokenに対応するchunk NE-typeを同定(token-span vs chunk-span の包含関係計算)
        token_labels = self.get_token_labels(token_spans, chunk_spans, chunk_labels)

        # CoNLL2003-likeなtoken行単位の列データを返す
        token_label_columns = [
            (token, label) for token, label in zip(tokens, token_labels)
        ]
        # 計算したspanが合ってるかのassertion
        spannedtoken_label_columns = [
            (text[span[0] : span[1]], label)
            for span, label in zip(token_spans, token_labels)
        ]
        assert spannedtoken_label_columns == token_label_columns

        return token_label_columns


class TokenizationAligner:
    def __init__(self):
        self.wakati = MeCab.Tagger("-Owakati")

    def tokenize(self, text: str) -> List[str]:
        return self.wakati.parse(text).split()

    @staticmethod
    def align_token_spans(token_chunk_spans, tokens, new_tokens):
        """get spans for new tokenization by token alignment"""
        org2new, _ = tokenizations.get_alignments(tokens, new_tokens)
        new_spans = []
        for token_s, token_e in token_chunk_spans:
            new_s = org2new[token_s][0]
            new_e = org2new[token_e - 1][-1] + 1
            new_spans.append((new_s, new_e))
        return new_spans

    @classmethod
    def convert_new_tokenization(cls, tokens, text, labels, tokenize, mode="bilou"):

        # ラベルからアノテーションのtoken位置スパンを再構成
        if mode == "bio":
            chunks = iobes.parse_spans_bio(labels)
        elif mode == "bilou":
            chunks = iobes.parse_spans_bilou(labels)
        elif mode == "bmeow":
            chunks = iobes.parse_spans_bmeow(labels)
        elif mode == "iobes":
            chunks = iobes.parse_spans_iobes(labels)
        else:
            chunks = None
            print("ERROR!!")
            return labels
        chunk_types = [chunk.type for chunk in chunks]
        token_chunk_spans = [(chunk.start, chunk.end) for chunk in chunks]

        # 新しい分かち書き単位における、アノテーションのtoken位置スパンを同定
        new_tokens = tokenize(text)
        new_token_chunk_spans = cls.align_token_spans(
            token_chunk_spans, tokens, new_tokens
        )

        # 新たなスパンからBIOラベルを再構成
        tokenspan2tagtype = dict(zip(new_token_chunk_spans, chunk_types))
        all_new_token_spans = [(i, i + 1) for i, _ in enumerate(new_tokens)]
        label = "O"
        new_labels = []
        for token_span in all_new_token_spans:
            if token_span in tokenspan2tagtype:
                tagtype = tokenspan2tagtype[token_span]
                if label == "O":
                    label = f"B-{tagtype}"
                else:
                    label = f"I-{tagtype}"
            else:
                label = "O"
            new_labels.append(label)
        # TODO: 必要ならiobesでラベル変換
        return new_tokens, new_labels

    def get_new_tokenization(self, sentences):
        columns = []
        for sentence in sentences:
            text = sentence.text
            tokens = [token.text for token in sentence]
            labels = [token.label for token in sentence]
            new_tokens, new_labels = self.convert_new_tokenization(
                tokens, text, labels, self.tokenize
            )
            columns.append(list(zip(new_tokens, new_labels)))
        return columns


def download_data(url, filepath):
    response = requests.get(url)
    if response.ok:
        with open(filepath, "w") as fp:
            fp.write(response.content.decode("utf8"))
        return filepath


def make_sample_conll_corpus(data_folder):
    """conllフォーマットデータのダウンロード"""
    url = "https://raw.githubusercontent.com/Hironsan/IOB2Corpus/master/ja.wikipedia.conll"
    train_file = "ja.wikipedia.conll"
    filepath = data_folder / train_file
    if download_data(url, filepath):
        columns = {0: "text", 1: "ner"}
        corpus = ColumnCorpus(data_folder, columns, train_file=train_file)

        return corpus


def make_gsd_conll_corpus(data_folder):
    """conllフォーマットデータのダウンロード"""
    train_url = "https://github.com/megagonlabs/UD_Japanese-GSD/releases/download/v2.6-NE/ja_gsd-ud-train.ne.conllu"
    train_file = "ja_gsd-ud-train.ne.conllu"
    train_path = data_folder / train_file
    dev_url = "https://github.com/megagonlabs/UD_Japanese-GSD/releases/download/v2.6-NE/ja_gsd-ud-dev.ne.conllu"
    dev_file = "ja_gsd-ud-dev.ne.conllu"
    dev_path = data_folder / dev_file
    test_url = "https://github.com/megagonlabs/UD_Japanese-GSD/releases/download/v2.6-NE/ja_gsd-ud-test.ne.conllu"
    test_file = "ja_gsd-ud-test.ne.conllu"
    test_path = data_folder / test_file
    if (
        download_data(train_url, train_path)
        and download_data(dev_url, dev_path)
        and download_data(test_url, test_path)
    ):
        ta = TokenizationAligner()
        for mode in ["train", "dev", "test"]:

            with open(data_folder / f"ja_gsd-ud-{mode}.ne.conllu") as fp:
                sentences_necolumns = sentences_to_necolumns(fp.read().split("\n\n"))
                sentences = [
                    Sentence(
                        [
                            Token(*row.split("\t"))
                            for row in rows
                            if len(row.split("\t")) == 2
                        ]
                    )
                    for rows in sentences_necolumns
                ]
            token_labels = ta.get_new_tokenization(sentences)

            with open(data_folder / f"ja.gsd-ud.{mode}.conll", "w") as fp:
                for data in token_labels:
                    for token, label in data:
                        fp.write(f"{token}\t{label}")
                        fp.write("\n")
                    fp.write("\n")

        train_file = "ja.gsd-ud.train.conll"
        dev_file = "ja.gsd-ud.dev.conll"
        test_file = "ja.gsd-ud.test.conll"
        columns = {0: "text", 1: "ner"}
        corpus = ColumnCorpus(
            data_folder,
            columns,
            train_file=train_file,
            dev_file=dev_file,
            test_file=test_file,
        )
        return corpus


def make_conll(cc, reader):
    conll = []
    for text, entd in reader.iter():
        labels = [label for _, _, label in entd["entities"]]
        spans = [(s, e) for s, e, _ in entd["entities"]]
        tokens_labels = cc.tokenize_and_align_spans(text, spans, labels)
        conll.append(
            "\n".join([f"{token}\t{label}" for (token, label) in tokens_labels])
        )
    return conll


def make_conll_corpus(data_folder):
    """camphr形式からtokenize & conllフォーマットに変換し、Corpus化"""
    cc = ConllConverter()
    train_file = "train.jsonl"
    test_file = "test.jsonl"
    with jsonlines.open(data_folder / train_file, "r") as reader:
        conll = make_conll(cc, reader)
    train_file = "train.conll"
    with open(data_folder / train_file, "w") as fp:
        fp.write("\n\n".join(conll))

    with jsonlines.open(data_folder / test_file, "r") as reader:
        conll = make_conll(cc, reader)
    test_file = "test.conll"
    with open(data_folder / test_file, "w") as fp:
        fp.write("\n\n".join(conll))

    columns = {0: "text", 1: "ner"}
    corpus = ColumnCorpus(
        data_folder, columns, train_file=train_file, test_file=test_file
    )

    return corpus


def make_tagger(corpus, cfg):
    tag_type = "ner"
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary.idx2item)

    # Contextual string embeddings of words, as proposed in Akbik et al., 2018.
    # using a character-level language model
    # Trained with 439M words of Japanese Web crawls
    embedding_types = [
        FlairEmbeddings("ja-forward"),
        FlairEmbeddings("ja-backward"),
    ]
    embeddings = StackedEmbeddings(embeddings=embedding_types)

    tagger = SequenceTagger(
        hidden_size=cfg.model.hidden_size,
        dropout=cfg.model.dropout,
        word_dropout=cfg.model.word_dropout,
        locked_dropout=cfg.model.locked_dropout,
        embeddings=embeddings,
        tag_dictionary=tag_dictionary,
        tag_type=tag_type,
        use_crf=True,
    )
    return tagger


@hydra.main(config_name="config")
def main(cfg: DictConfig):
    print(f"Current working directory: {os.getcwd()}")
    print(f"Orig working directory : {hydra.utils.get_original_cwd()}")
    print(
        f'to_absolute_path("outputs/data") : {hydra.utils.to_absolute_path("outputs/data")}'
    )

    data_folder = Path(hydra.utils.get_original_cwd()) / "outputs/data"
    corpus = make_gsd_conll_corpus(data_folder)
    tagger = make_tagger(corpus, cfg)
    trainer = ModelTrainer(tagger, corpus)
    model_dir = Path(hydra.utils.get_original_cwd()) / "outputs/models"
    trainer.train(
        model_dir,
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

    tagger = SequenceTagger.load(model_dir / "best-model.pt")
    qtagger = torch.quantization.quantize_dynamic(
        tagger, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8
    )
    with open(model_dir / "quantized-best-model.pkl", "wb") as fp:
        pickle.dump(qtagger, fp)


if __name__ == "__main__":
    main()
