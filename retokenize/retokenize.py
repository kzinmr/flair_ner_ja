# Convert CoNLL2003-like column data to chunks and spans
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union
import click
import jsonlines
import MeCab
import requests
import tokenizations


@dataclass(frozen=True)
class Token:
    text: str
    label: str


@dataclass(frozen=True)
class Chunk:
    tokens: List[Token]
    label: str
    span: Tuple[int, int]

    def __iter__(self):
        for token in self.tokens:
            yield token


@dataclass(frozen=True)
class Sentence(Iterable[Token]):
    text: str
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


class ConllConverter:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)

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
        token_spans: List[Tuple[int, int]],
        chunk_spans: List[Tuple[int, int]],
        chunk_labels: List[str],
    ) -> List[str]:
        """chunk単位のNE-typeラベルから、token単位のBIOラベルを構成"""

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
        # assertion
        spannedtokens = [text[span[0] : span[1]] for span in token_spans]
        assert spannedtokens == tokens

        # 各tokenに対応するchunk NE-typeを同定(token-span vs chunk-span の包含関係計算)
        token_labels = self.get_token_labels(token_spans, chunk_spans, chunk_labels)

        # CoNLL2003-likeなtoken行単位の列データを返す
        return [(token, label) for token, label in zip(tokens, token_labels)]


class MecabTokenizer:
    def __init__(self):
        self.tagger = MeCab.Tagger("-Owakati")

    def tokenize(self, text: str) -> List[str]:
        return self.tagger.parse(text).split()


def make_sentences_from_conll(conll_filepath: str) -> List[Sentence]:
    """conll (token-label columns) -> sentences (token-span-label containers)"""
    with open(conll_filepath) as fp:
        sentences = fp.read().split("\n\n")
        sentences = [
            Sentence(
                [
                    Token(*token.split("\t"))
                    for token in sentence.split("\n")
                    if len(token.split("\t")) == 2
                ]
            )
            for sentence in sentences
        ]
        return [s for s in sentences if s.text]


def retokenize_sentences(sentences: List[Sentence], tokenizer=None) -> str:
    """sentences -> re-tokenization -> conll"""
    if tokenizer is None:
        tokenizer = MecabTokenizer()
    conll = ConllConverter(tokenizer)
    sentence_columns = []
    for sentence in sentences:
        text, chunks = sentence.text, sentence.chunks
        chunk_spans = [chunk.span for chunk in chunks]
        chunk_tagtypes = [chunk.label for chunk in chunks]
        # print(text, chunk_spans, chunk_tagtypes)

        sentence_column: List[Tuple[str, str]] = conll.tokenize_and_align_spans(
            text, chunk_spans, chunk_tagtypes
        )
        sentence_column_str = "\n".join(
            [f"{token}\t{label}" for token, label in sentence_column]
        )
        sentence_columns.append(sentence_column_str + "\n")
    return "\n".join(sentence_columns)


def retokenize_conll(filepath: Union[str, Path], tokenizer=None):
    sentences: List[Sentence] = make_sentences_from_conll(filepath)
    new_conll = retokenize_sentences(sentences, tokenizer)
    filepath_ret = filepath + ".retokenize"
    with open(filepath_ret, "w") as fp:
        fp.write(new_conll)

    sentences: List[Sentence] = make_sentences_from_conll(filepath_ret)
    filepath_jl = filepath + ".jsonl"
    make_jsonl(sentences, filepath_jl)


def make_jsonl(sentences, filepath):
    with open(filepath, "w") as fp:
        for sentence in sentences:
            text = sentence.text
            jl = [
                text,
                {
                    "entities": [
                        [c.span[0], c.span[1], c.label] for c in sentence.chunks
                    ]
                },
            ]
            fp.write(json.dumps(jl))
            fp.write("\n")


def download_conll_data(filepath: str = "test.bio"):
    """conllフォーマットデータのダウンロード"""
    filename = Path(filepath).name
    url = f"https://github.com/megagonlabs/UD_Japanese-GSD/releases/download/v2.6-NE/{filename}"
    response = requests.get(url)
    if response.ok:
        with open(filepath, "w") as fp:
            fp.write(response.content.decode("utf8"))
        return filepath


def make_conll_corpus(jsonl_path: Path, tokenizer=None):
    """camphr形式からtokenize & conllフォーマットに変換し、Corpus化"""

    def __make_conll(cc, reader):
        conll = []
        for text, entd in reader.iter():
            labels = [label for _, _, label in entd["entities"]]
            spans = [(s, e) for s, e, _ in entd["entities"]]
            tokens_labels = cc.tokenize_and_align_spans(text, spans, labels)
            conll.append(
                "\n".join([f"{token}\t{label}" for (token, label) in tokens_labels])
            )
        return conll

    if tokenizer is None:
        tokenizer = MecabTokenizer()
    cc = ConllConverter(tokenizer)

    with jsonlines.open(jsonl_path, "r") as reader:
        conll = __make_conll(cc, reader)
        conll_path = f"{jsonl_path}.conll"
        with open(conll_path, "w") as fp:
            fp.write("\n\n".join(conll))


@click.command()
@click.option(
    "--train_file",
    type=click.Path(exists=True),
    default="/app/data/train.jsonl",
)
@click.option(
    "--dev_file",
    type=click.Path(exists=True),
    default="/app/data/dev.jsonl",
)
@click.option(
    "--test_file",
    type=click.Path(exists=True),
    default="/app/data/test.jsonl",
)
@click.option(
    "--download_file",
    is_flag=True,
)
def main(
    train_file: Optional[Path] = None,
    dev_file: Optional[Path] = None,
    test_file: Optional[Path] = None,
    download_file: bool = False,
):
    train_file = Path(train_file)
    dev_file = Path(dev_file)
    test_file = Path(test_file)
    if download_file:
        for mode in ["train", "dev", "test"]:
            filepath = f"/app/data/{mode}.bio"
            if download_conll_data(filepath):
                retokenize_conll(filepath)

    elif train_file is not None and dev_file is not None and test_file is not None:
        if (
            train_file.suffix in [".txt", ".bio", ".conll"]
            and dev_file.suffix in [".txt", ".bio", ".conll"]
            and test_file.suffix in [".txt", ".bio", ".conll"]
        ):
            retokenize_conll(train_file)
            retokenize_conll(dev_file)
            retokenize_conll(test_file)
        elif (
            train_file.suffix == ".jsonl"
            and dev_file.suffix == ".jsonl"
            and test_file.suffix == ".jsonl"
        ):
            make_conll_corpus(train_file)
            make_conll_corpus(dev_file)
            make_conll_corpus(test_file)
        else:
            print(
                "invalid suffix exists in [{train_file.suffix} {dev_file.suffix} {test_file.suffix}]"
            )


if __name__ == "__main__":
    main()