import inspect
import logging
import mmap
import os
import pickle
import re
import shutil
import tempfile
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from operator import itemgetter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import konoha
import langdetect
import numpy as np
import requests
import torch
import torch.nn
import torch.nn.functional as F
import torch.utils.data.dataloader
from sklearn import metrics
from tabulate import tabulate
from torch.nn import Parameter
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset
from torch.utils.data.dataset import ConcatDataset, Dataset, Subset
from tqdm import tqdm
from transformers import file_utils
from zipfile import ZipFile

# import flair
# import flair.nn
# from flair.data import Dictionary, Sentence, Label
# from flair.datasets import DataLoader, SentenceDataset
# from flair.embeddings import StackedEmbeddings, TokenEmbeddings
# from flair.embeddings.base import Embeddings
# from flair.file_utils import cached_path, unzip_file, Tqdm
# from flair.training_utils import Metric, Result, store_embeddings
logger = logging.getLogger("flair")

if torch.cuda.is_available():
    flair_device = torch.device("cuda:0")
else:
    flair_device = torch.device("cpu")

flair_embedding_storage_mode = "cpu"

flair_cache_root = os.getenv("FLAIR_CACHE_ROOT", Path(Path.home(), ".flair"))

START_TAG: str = "<START>"
STOP_TAG: str = "<STOP>"


class Tqdm:
    # These defaults are the same as the argument defaults in tqdm.
    default_mininterval: float = 0.1

    @staticmethod
    def set_default_mininterval(value: float) -> None:
        Tqdm.default_mininterval = value

    @staticmethod
    def set_slower_interval(use_slower_interval: bool) -> None:
        """
        If ``use_slower_interval`` is ``True``, we will dramatically slow down ``tqdm's`` default
        output rate.  ``tqdm's`` default output rate is great for interactively watching progress,
        but it is not great for log files.  You might want to set this if you are primarily going
        to be looking at output through log files, not the terminal.
        """
        if use_slower_interval:
            Tqdm.default_mininterval = 10.0
        else:
            Tqdm.default_mininterval = 0.1

    @staticmethod
    def tqdm(*args, **kwargs):
        new_kwargs = {"mininterval": Tqdm.default_mininterval, **kwargs}

        return tqdm(*args, **new_kwargs)


def cached_path(url_or_filename: str, cache_dir: Union[str, Path]) -> Path:
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    """

    def get_from_cache(url: str, cache_dir: Path = None) -> Path:
        """
        Given a URL, look for the corresponding dataset in the local cache.
        If it's not there, download it. Then return the path to the cached file.
        """
        cache_dir.mkdir(parents=True, exist_ok=True)

        filename = re.sub(r".+/", "", url)
        # get cache path to put the file
        cache_path = cache_dir / filename
        if cache_path.exists():
            return cache_path

        # make HEAD request to check ETag
        response = requests.head(
            url, headers={"User-Agent": "Flair"}, allow_redirects=True
        )
        if response.status_code != 200:
            raise IOError(
                f"HEAD request failed for url {url} with status code {response.status_code}."
            )

        # add ETag to filename if it exists
        # etag = response.headers.get("ETag")

        if not cache_path.exists():
            # Download to temporary file, then copy to cache dir once finished.
            # Otherwise you get corrupt cache entries if the download gets interrupted.
            fd, temp_filename = tempfile.mkstemp()
            logger.info("%s not found in cache, downloading to %s", url, temp_filename)

            # GET file object
            req = requests.get(url, stream=True, headers={"User-Agent": "Flair"})
            content_length = req.headers.get("Content-Length")
            total = int(content_length) if content_length is not None else None
            progress = Tqdm.tqdm(unit="B", total=total)
            with open(temp_filename, "wb") as temp_file:
                for chunk in req.iter_content(chunk_size=1024):
                    if chunk:  # filter out keep-alive new chunks
                        progress.update(len(chunk))
                        temp_file.write(chunk)

            progress.close()

            logger.info("copying %s to cache at %s", temp_filename, cache_path)
            shutil.copyfile(temp_filename, str(cache_path))
            logger.info("removing temp file %s", temp_filename)
            os.close(fd)
            os.remove(temp_filename)

        return cache_path

    if type(cache_dir) is str:
        cache_dir = Path(cache_dir)
    dataset_cache = Path(flair_cache_root) / cache_dir

    parsed = urlparse(url_or_filename)

    if parsed.scheme in ("http", "https"):
        # URL, so get it from the cache (downloading if necessary)
        return get_from_cache(url_or_filename, dataset_cache)
    elif parsed.scheme == "" and Path(url_or_filename).exists():
        # File, and it exists.
        return Path(url_or_filename)
    elif parsed.scheme == "":
        # File, but it doesn't exist.
        raise FileNotFoundError("file {} not found".format(url_or_filename))
    else:
        # Something unknown
        raise ValueError(
            "unable to parse {} as a URL or as a local path".format(url_or_filename)
        )


def unzip_file(file: Union[str, Path], unzip_to: Union[str, Path]):
    with ZipFile(Path(file), "r") as zipObj:
        # Extract all the contents of zip file in current directory
        zipObj.extractall(Path(unzip_to))


class Dictionary:
    """
    This class holds a dictionary that maps strings to IDs, used to generate one-hot encodings of strings.
    """

    def __init__(self, add_unk=True):
        # init dictionaries
        self.item2idx: Dict[str, int] = {}
        self.idx2item: List[str] = []
        self.multi_label: bool = False

        # in order to deal with unknown tokens, add <unk>
        if add_unk:
            self.add_item("<unk>")

    def add_item(self, item: str) -> int:
        """
        add string - if already in dictionary returns its ID. if not in dictionary, it will get a new ID.
        :param item: a string for which to assign an id.
        :return: ID of string
        """
        item = item.encode("utf-8")
        if item not in self.item2idx:
            self.idx2item.append(item)
            self.item2idx[item] = len(self.idx2item) - 1
        return self.item2idx[item]

    def get_idx_for_item(self, item: str) -> int:
        """
        returns the ID of the string, otherwise 0
        :param item: string for which ID is requested
        :return: ID of string, otherwise 0
        """
        item = item.encode("utf-8")
        if item in self.item2idx.keys():
            return self.item2idx[item]
        else:
            return 0

    def get_idx_for_items(self, items: List[str]) -> List[int]:
        """
        returns the IDs for each item of the list of string, otherwise 0 if not found
        :param items: List of string for which IDs are requested
        :return: List of ID of strings
        """
        if not hasattr(self, "item2idx_not_encoded"):
            d = dict(
                [(key.decode("UTF-8"), value) for key, value in self.item2idx.items()]
            )
            self.item2idx_not_encoded = defaultdict(int, d)

        if not items:
            return []
        results = itemgetter(*items)(self.item2idx_not_encoded)
        if isinstance(results, int):
            return [results]
        return list(results)

    def get_items(self) -> List[str]:
        items = []
        for item in self.idx2item:
            items.append(item.decode("UTF-8"))
        return items

    def __len__(self) -> int:
        return len(self.idx2item)

    def get_item_for_index(self, idx):
        return self.idx2item[idx].decode("UTF-8")

    def save(self, savefile):
        with open(savefile, "wb") as f:
            mappings = {"idx2item": self.idx2item, "item2idx": self.item2idx}
            pickle.dump(mappings, f)

    @staticmethod
    def load_from_file(filename: Union[str, Path]):
        dictionary: Dictionary = Dictionary()
        with open(filename, "rb") as f:
            mappings = pickle.load(f, encoding="latin1")
            idx2item = mappings["idx2item"]
            item2idx = mappings["item2idx"]
            dictionary.item2idx = item2idx
            dictionary.idx2item = idx2item
        return dictionary

    @staticmethod
    def load(name: str):
        # from flair.file_utils import cached_path

        hu_path: str = "https://flair.informatik.hu-berlin.de/resources/characters"
        if name == "chars" or name == "common-chars":
            char_dict = cached_path(
                f"{hu_path}/common_characters", cache_dir="datasets"
            )
            return Dictionary.load_from_file(char_dict)

        if name == "chars-large" or name == "common-chars-large":
            char_dict = cached_path(
                f"{hu_path}/common_characters_large", cache_dir="datasets"
            )
            return Dictionary.load_from_file(char_dict)

        if name == "chars-xl" or name == "common-chars-xl":
            char_dict = cached_path(
                f"{hu_path}/common_characters_xl", cache_dir="datasets"
            )
            return Dictionary.load_from_file(char_dict)

        return Dictionary.load_from_file(name)

    def __str__(self):
        tags = ", ".join(self.get_item_for_index(i) for i in range(min(len(self), 30)))
        return f"Dictionary with {len(self)} tags: {tags}"


class Label:
    """
    This class represents a label. Each label has a value and optionally a confidence score. The
    score needs to be between 0.0 and 1.0. Default value for the score is 1.0.
    """

    def __init__(self, value: str, score: float = 1.0):
        self.value = value
        self.score = score
        super().__init__()

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if not value and value != "":
            raise ValueError(
                "Incorrect label value provided. Label value needs to be set."
            )
        else:
            self._value = value

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, score):
        if 0.0 <= score <= 1.0:
            self._score = score
        else:
            self._score = 1.0

    def to_dict(self):
        return {"value": self.value, "confidence": self.score}

    def __str__(self):
        return f"{self._value} ({round(self._score, 4)})"

    def __repr__(self):
        return f"{self._value} ({round(self._score, 4)})"


class DataPoint:
    """
    This is the parent class of all data points in Flair (including Token, Sentence, Image, etc.). Each DataPoint
    must be embeddable (hence the abstract property embedding() and methods to() and clear_embeddings()). Also,
    each DataPoint may have Labels in several layers of annotation (hence the functions add_label(), get_labels()
    and the property 'label')
    """

    def __init__(self):
        self.annotation_layers = {}

    @property
    @abstractmethod
    def embedding(self):
        pass

    @abstractmethod
    def to(self, device: str, pin_memory: bool = False):
        pass

    @abstractmethod
    def clear_embeddings(self, embedding_names: List[str] = None):
        pass

    def add_label(self, label_type: str, value: str, score: float = 1.0):

        if label_type not in self.annotation_layers:
            self.annotation_layers[label_type] = [Label(value, score)]
        else:
            self.annotation_layers[label_type].append(Label(value, score))

        return self

    def set_label(self, label_type: str, value: str, score: float = 1.0):
        self.annotation_layers[label_type] = [Label(value, score)]

        return self

    def remove_labels(self, label_type: str):
        if label_type in self.annotation_layers.keys():
            del self.annotation_layers[label_type]

    def get_labels(self, label_type: str = None):
        if label_type is None:
            return self.labels

        return (
            self.annotation_layers[label_type]
            if label_type in self.annotation_layers
            else []
        )

    @property
    def labels(self) -> List[Label]:
        all_labels = []
        for key in self.annotation_layers.keys():
            all_labels.extend(self.annotation_layers[key])
        return all_labels


class Token(DataPoint):
    """
    This class represents one word in a tokenized sentence. Each token may have any number of tags. It may also point
    to its head in a dependency tree.
    """

    def __init__(
        self,
        text: str,
        idx: int = None,
        head_id: int = None,
        whitespace_after: bool = True,
        start_position: int = None,
    ):
        super().__init__()

        self.text: str = text
        self.idx: int = idx
        self.head_id: int = head_id
        self.whitespace_after: bool = whitespace_after

        self.start_pos = start_position
        self.end_pos = (
            start_position + len(text) if start_position is not None else None
        )

        self.sentence: Sentence = None
        self._embeddings: Dict = {}
        self.tags_proba_dist: Dict[str, List[Label]] = {}

    def add_tag_label(self, tag_type: str, tag: Label):
        self.set_label(tag_type, tag.value, tag.score)

    def add_tags_proba_dist(self, tag_type: str, tags: List[Label]):
        self.tags_proba_dist[tag_type] = tags

    def add_tag(self, tag_type: str, tag_value: str, confidence=1.0):
        self.set_label(tag_type, tag_value, confidence)

    def get_tag(self, label_type):
        if len(self.get_labels(label_type)) == 0:
            return Label("")
        return self.get_labels(label_type)[0]

    def get_tags_proba_dist(self, tag_type: str) -> List[Label]:
        if tag_type in self.tags_proba_dist:
            return self.tags_proba_dist[tag_type]
        return []

    def get_head(self):
        return self.sentence.get_token(self.head_id)

    def set_embedding(self, name: str, vector: torch.tensor):
        device = flair_device  # flair_device
        if (flair_embedding_storage_mode == "cpu") and len(self._embeddings.keys()) > 0:
            device = next(iter(self._embeddings.values())).device
        if device != vector.device:
            vector = vector.to(device)
        self._embeddings[name] = vector

    def to(self, device: str, pin_memory: bool = False):
        for name, vector in self._embeddings.items():
            if str(vector.device) != str(device):
                if pin_memory:
                    self._embeddings[name] = vector.to(
                        device, non_blocking=True
                    ).pin_memory()
                else:
                    self._embeddings[name] = vector.to(device, non_blocking=True)

    def clear_embeddings(self, embedding_names: List[str] = None):
        if embedding_names is None:
            self._embeddings: Dict = {}
        else:
            for name in embedding_names:
                if name in self._embeddings.keys():
                    del self._embeddings[name]

    def get_each_embedding(
        self, embedding_names: Optional[List[str]] = None
    ) -> torch.tensor:
        embeddings = []
        for embed in sorted(self._embeddings.keys()):
            if embedding_names and embed not in embedding_names:
                continue
            embed = self._embeddings[embed].to(flair_device)
            if (flair_embedding_storage_mode == "cpu") and embed.device != flair_device:
                embed = embed.to(flair_device)
            embeddings.append(embed)
        return embeddings

    def get_embedding(self, names: Optional[List[str]] = None) -> torch.tensor:
        embeddings = self.get_each_embedding(names)

        if embeddings:
            return torch.cat(embeddings, dim=0)

        return torch.tensor([], device=flair_device)

    @property
    def start_position(self) -> int:
        return self.start_pos

    @property
    def end_position(self) -> int:
        return self.end_pos

    @property
    def embedding(self):
        return self.get_embedding()

    def __str__(self) -> str:
        return (
            "Token: {} {}".format(self.idx, self.text)
            if self.idx is not None
            else "Token: {}".format(self.text)
        )

    def __repr__(self) -> str:
        return (
            "Token: {} {}".format(self.idx, self.text)
            if self.idx is not None
            else "Token: {}".format(self.text)
        )


class Span(DataPoint):
    """
    This class represents one textual span consisting of Tokens.
    """

    def __init__(self, tokens: List[Token]):

        super().__init__()

        self.tokens = tokens
        self.start_pos = None
        self.end_pos = None

        if tokens:
            self.start_pos = tokens[0].start_position
            self.end_pos = tokens[len(tokens) - 1].end_position

    @property
    def text(self) -> str:
        return " ".join([t.text for t in self.tokens])

    def to_original_text(self) -> str:
        pos = self.tokens[0].start_pos
        if pos is None:
            return " ".join([t.text for t in self.tokens])
        str = ""
        for t in self.tokens:
            while t.start_pos != pos:
                str += " "
                pos += 1

            str += t.text
            pos += len(t.text)

        return str

    def to_dict(self):
        return {
            "text": self.to_original_text(),
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "labels": self.labels,
        }

    def __str__(self) -> str:
        ids = ",".join([str(t.idx) for t in self.tokens])
        label_string = " ".join([str(label) for label in self.labels])
        labels = f"   [− Labels: {label_string}]" if self.labels is not None else ""
        return 'Span [{}]: "{}"{}'.format(ids, self.text, labels)

    def __repr__(self) -> str:
        ids = ",".join([str(t.idx) for t in self.tokens])
        return (
            '<{}-span ({}): "{}">'.format(self.tag, ids, self.text)
            if self.tag is not None
            else '<span ({}): "{}">'.format(ids, self.text)
        )

    def __getitem__(self, idx: int) -> Token:
        return self.tokens[idx]

    def __iter__(self):
        return iter(self.tokens)

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def tag(self):
        return self.labels[0].value

    @property
    def score(self):
        return self.labels[0].score


class Tokenizer(ABC):
    r"""An abstract class representing a :class:`Tokenizer`.

    Tokenizers are used to represent algorithms and models to split plain text into
    individual tokens / words. All subclasses should overwrite :meth:`tokenize`, which
    splits the given plain text into tokens. Moreover, subclasses may overwrite
    :meth:`name`, returning a unique identifier representing the tokenizer's
    configuration.
    """

    @abstractmethod
    def tokenize(self, text: str) -> List[Token]:
        raise NotImplementedError()

    @property
    def name(self) -> str:
        return self.__class__.__name__


class TokenizerWrapper(Tokenizer):
    """
    Helper class to wrap tokenizer functions to the class-based tokenizer interface.
    """

    def __init__(self, tokenizer_func: Callable[[str], List[Token]]):
        super(TokenizerWrapper, self).__init__()
        self.tokenizer_func = tokenizer_func

    def tokenize(self, text: str) -> List[Token]:
        return self.tokenizer_func(text)

    @property
    def name(self) -> str:
        return self.__class__.__name__ + "_" + self.tokenizer_func.__name__


class SegtokTokenizer(Tokenizer):
    """
    Tokenizer using segtok, a third party library dedicated to rules-based Indo-European languages.

    For further details see: https://github.com/fnl/segtok
    """

    def __init__(self):
        super(SegtokTokenizer, self).__init__()

    def tokenize(self, text: str) -> List[Token]:
        return SegtokTokenizer.run_tokenize(text)

    @staticmethod
    def run_tokenize(text: str) -> List[Token]:
        tokens: List[Token] = []
        words: List[str] = []

        sentences = split_single(text)
        for sentence in sentences:
            contractions = split_contractions(word_tokenizer(sentence))
            words.extend(contractions)

        words = list(filter(None, words))

        # determine offsets for whitespace_after field
        index = text.index
        current_offset = 0
        previous_word_offset = -1
        previous_token = None
        for word in words:
            try:
                word_offset = index(word, current_offset)
                start_position = word_offset
            except:
                word_offset = previous_word_offset + 1
                start_position = (
                    current_offset + 1 if current_offset > 0 else current_offset
                )

            if word:
                token = Token(
                    text=word, start_position=start_position, whitespace_after=True
                )
                tokens.append(token)

            if (previous_token is not None) and word_offset - 1 == previous_word_offset:
                previous_token.whitespace_after = False

            current_offset = word_offset + len(word)
            previous_word_offset = current_offset - 1
            previous_token = token

        return tokens


class SpaceTokenizer(Tokenizer):
    """
    Tokenizer based on space character only.
    """

    def __init__(self):
        super(SpaceTokenizer, self).__init__()

    def tokenize(self, text: str) -> List[Token]:
        return SpaceTokenizer.run_tokenize(text)

    @staticmethod
    def run_tokenize(text: str) -> List[Token]:
        tokens: List[Token] = []
        word = ""
        index = -1
        for index, char in enumerate(text):
            if char == " ":
                if len(word) > 0:
                    start_position = index - len(word)
                    tokens.append(
                        Token(
                            text=word,
                            start_position=start_position,
                            whitespace_after=True,
                        )
                    )

                word = ""
            else:
                word += char
        # increment for last token in sentence if not followed by whitespace
        index += 1
        if len(word) > 0:
            start_position = index - len(word)
            tokens.append(
                Token(text=word, start_position=start_position, whitespace_after=False)
            )

        return tokens


class JapaneseTokenizer(Tokenizer):
    """
    Tokenizer using konoha, a third party library which supports
    multiple Japanese tokenizer such as MeCab, Janome and SudachiPy.

    For further details see:
        https://github.com/himkt/konoha
    """

    def __init__(self, tokenizer: str, sudachi_mode: str = "A"):
        super(JapaneseTokenizer, self).__init__()

        available_tokenizers = ["mecab", "janome", "sudachi"]

        if tokenizer.lower() not in available_tokenizers:
            raise NotImplementedError(
                f"Currently, {tokenizer} is only supported. Supported tokenizers: {available_tokenizers}."
            )

        self.tokenizer = tokenizer
        self.sentence_tokenizer = konoha.SentenceTokenizer()
        self.word_tokenizer = konoha.WordTokenizer(tokenizer, mode=sudachi_mode)

    def tokenize(self, text: str) -> List[Token]:
        tokens: List[Token] = []
        words: List[str] = []

        sentences = self.sentence_tokenizer.tokenize(text)
        for sentence in sentences:
            konoha_tokens = self.word_tokenizer.tokenize(sentence)
            words.extend(list(map(str, konoha_tokens)))

        # determine offsets for whitespace_after field
        index = text.index
        current_offset = 0
        previous_word_offset = -1
        previous_token = None
        for word in words:
            try:
                word_offset = index(word, current_offset)
                start_position = word_offset
            except:
                word_offset = previous_word_offset + 1
                start_position = (
                    current_offset + 1 if current_offset > 0 else current_offset
                )

            token = Token(
                text=word, start_position=start_position, whitespace_after=True
            )
            tokens.append(token)

            if (previous_token is not None) and word_offset - 1 == previous_word_offset:
                previous_token.whitespace_after = False

            current_offset = word_offset + len(word)
            previous_word_offset = current_offset - 1
            previous_token = token

        return tokens

    @property
    def name(self) -> str:
        return self.__class__.__name__ + "_" + self.tokenizer


class Sentence(DataPoint):
    """
    A Sentence is a list of tokens and is used to represent a sentence or text fragment.
    """

    def __init__(
        self,
        text: Union[str, List[str]] = None,
        use_tokenizer: Union[bool, Tokenizer] = True,
        language_code: str = None,
        start_position: int = None,
    ):
        """
        Class to hold all meta related to a text (tokens, predictions, language code, ...)
        :param text: original string (sentence), or a list of string tokens (words)
        :param use_tokenizer: a custom tokenizer (default is :class:`SpaceTokenizer`)
            more advanced options are :class:`SegTokTokenizer` to use segtok or :class:`SpacyTokenizer`
            to use Spacy library if available). Check the implementations of abstract class Tokenizer or
            implement your own subclass (if you need it). If instead of providing a Tokenizer, this parameter
            is just set to True (deprecated), :class:`SegtokTokenizer` will be used.
        :param language_code: Language of the sentence
        :param start_position: Start char offset of the sentence in the superordinate document
        """
        super().__init__()

        self.tokens: List[Token] = []

        self._embeddings: Dict = {}

        self.language_code: str = language_code

        self.start_pos = start_position
        self.end_pos = (
            start_position + len(text) if start_position is not None else None
        )

        if isinstance(use_tokenizer, Tokenizer):
            tokenizer = use_tokenizer
        elif hasattr(use_tokenizer, "__call__"):
            # from flair.tokenization import TokenizerWrapper

            tokenizer = TokenizerWrapper(use_tokenizer)
        elif type(use_tokenizer) == bool:
            # from flair.tokenization import SegtokTokenizer, SpaceTokenizer

            tokenizer = SegtokTokenizer() if use_tokenizer else SpaceTokenizer()
        else:
            raise AssertionError(
                "Unexpected type of parameter 'use_tokenizer'. "
                + "Parameter should be bool, Callable[[str], List[Token]] (deprecated), Tokenizer"
            )

        # if text is passed, instantiate sentence with tokens (words)
        if text is not None:
            if isinstance(text, (list, tuple)):
                [
                    self.add_token(self._restore_windows_1252_characters(token))
                    for token in text
                ]
            else:
                text = self._restore_windows_1252_characters(text)
                [self.add_token(token) for token in tokenizer.tokenize(text)]

        # log a warning if the dataset is empty
        if text == "":
            logger.warning(
                "Warning: An empty Sentence was created! Are there empty strings in your dataset?"
            )

        self.tokenized = None

        # some sentences represent a document boundary (but most do not)
        self.is_document_boundary: bool = False

    def get_token(self, token_id: int) -> Token:
        for token in self.tokens:
            if token.idx == token_id:
                return token

    def add_token(self, token: Union[Token, str]):

        if type(token) is str:
            token = Token(token)

        token.text = token.text.replace("\u200c", "")
        token.text = token.text.replace("\u200b", "")
        token.text = token.text.replace("\ufe0f", "")
        token.text = token.text.replace("\ufeff", "")

        # data with zero-width characters cannot be handled
        if token.text.strip() == "":
            return

        self.tokens.append(token)

        # set token idx if not set
        token.sentence = self
        if token.idx is None:
            token.idx = len(self.tokens)

    def get_label_names(self):
        label_names = []
        for label in self.labels:
            label_names.append(label.value)
        return label_names

    def _add_spans_internal(self, spans: List[Span], label_type: str, min_score):

        current_span = []

        tags = defaultdict(lambda: 0.0)

        previous_tag_value: str = "O"
        for token in self:

            tag: Label = token.get_tag(label_type)
            tag_value = tag.value

            # non-set tags are OUT tags
            if tag_value == "" or tag_value == "O" or tag_value == "_":
                tag_value = "O-"

            # anything that is not a BIOES tag is a SINGLE tag
            if tag_value[0:2] not in ["B-", "I-", "O-", "E-", "S-"]:
                tag_value = "S-" + tag_value

            # anything that is not OUT is IN
            in_span = False
            if tag_value[0:2] not in ["O-"]:
                in_span = True

            # single and begin tags start a new span
            starts_new_span = False
            if tag_value[0:2] in ["B-", "S-"]:
                starts_new_span = True

            if (
                previous_tag_value[0:2] in ["S-"]
                and previous_tag_value[2:] != tag_value[2:]
                and in_span
            ):
                starts_new_span = True

            if (starts_new_span or not in_span) and len(current_span) > 0:
                scores = [t.get_labels(label_type)[0].score for t in current_span]
                span_score = sum(scores) / len(scores)
                if span_score > min_score:
                    span = Span(current_span)
                    span.add_label(
                        label_type=label_type,
                        value=sorted(
                            tags.items(), key=lambda k_v: k_v[1], reverse=True
                        )[0][0],
                        score=span_score,
                    )
                    spans.append(span)

                current_span = []
                tags = defaultdict(lambda: 0.0)

            if in_span:
                current_span.append(token)
                weight = 1.1 if starts_new_span else 1.0
                tags[tag_value[2:]] += weight

            # remember previous tag
            previous_tag_value = tag_value

        if len(current_span) > 0:
            scores = [t.get_labels(label_type)[0].score for t in current_span]
            span_score = sum(scores) / len(scores)
            if span_score > min_score:
                span = Span(current_span)
                span.add_label(
                    label_type=label_type,
                    value=sorted(tags.items(), key=lambda k_v: k_v[1], reverse=True)[0][
                        0
                    ],
                    score=span_score,
                )
                spans.append(span)

        return spans

    def get_spans(self, label_type: Optional[str] = None, min_score=-1) -> List[Span]:

        spans: List[Span] = []

        # if label type is explicitly specified, get spans for this label type
        if label_type:
            return self._add_spans_internal(spans, label_type, min_score)

        # else determine all label types in sentence and get all spans
        label_types = []
        for token in self:
            for annotation in token.annotation_layers.keys():
                if annotation not in label_types:
                    label_types.append(annotation)

        for label_type in label_types:
            self._add_spans_internal(spans, label_type, min_score)
        return spans

    @property
    def embedding(self):
        return self.get_embedding()

    def set_embedding(self, name: str, vector: torch.tensor):
        device = flair_device
        if (flair_embedding_storage_mode == "cpu") and len(self._embeddings.keys()) > 0:
            device = next(iter(self._embeddings.values())).device
        if device != vector.device:
            vector = vector.to(device)
        self._embeddings[name] = vector

    def get_embedding(self, names: Optional[List[str]] = None) -> torch.tensor:
        embeddings = []
        for embed in sorted(self._embeddings.keys()):
            if names and embed not in names:
                continue
            embedding = self._embeddings[embed]
            embeddings.append(embedding)

        if embeddings:
            return torch.cat(embeddings, dim=0)

        return torch.Tensor()

    def to(self, device: str, pin_memory: bool = False):

        # move sentence embeddings to device
        for name, vector in self._embeddings.items():
            if str(vector.device) != str(device):
                if pin_memory:
                    self._embeddings[name] = vector.to(
                        device, non_blocking=True
                    ).pin_memory()
                else:
                    self._embeddings[name] = vector.to(device, non_blocking=True)

        # move token embeddings to device
        for token in self:
            token.to(device, pin_memory)

    def clear_embeddings(self, embedding_names: List[str] = None):

        # clear sentence embeddings
        if embedding_names is None:
            self._embeddings: Dict = {}
        else:
            for name in embedding_names:
                if name in self._embeddings.keys():
                    del self._embeddings[name]

        # clear token embeddings
        for token in self:
            token.clear_embeddings(embedding_names)

    def to_tagged_string(self, main_tag=None) -> str:
        list = []
        for token in self.tokens:
            list.append(token.text)

            tags: List[str] = []
            for label_type in token.annotation_layers.keys():

                if main_tag is not None and main_tag != label_type:
                    continue

                if token.get_labels(label_type)[0].value == "O":
                    continue
                if token.get_labels(label_type)[0].value == "_":
                    continue

                tags.append(token.get_labels(label_type)[0].value)
            all_tags = "<" + "/".join(tags) + ">"
            if all_tags != "<>":
                list.append(all_tags)
        return " ".join(list)

    def to_tokenized_string(self) -> str:

        if self.tokenized is None:
            self.tokenized = " ".join([t.text for t in self.tokens])

        return self.tokenized

    def to_plain_string(self):
        plain = ""
        for token in self.tokens:
            plain += token.text
            if token.whitespace_after:
                plain += " "
        return plain.rstrip()

    def convert_tag_scheme(self, tag_type: str = "ner", target_scheme: str = "iob"):

        tags: List[Label] = []
        for token in self.tokens:
            tags.append(token.get_tag(tag_type))

        if target_scheme == "iob":
            iob2(tags)

        if target_scheme == "iobes":
            iob2(tags)
            tags = iob_iobes(tags)

        for index, tag in enumerate(tags):
            self.tokens[index].set_label(tag_type, tag)

    def infer_space_after(self):
        """
        Heuristics in case you wish to infer whitespace_after values for tokenized text. This is useful for some old NLP
        tasks (such as CoNLL-03 and CoNLL-2000) that provide only tokenized data with no info of original whitespacing.
        :return:
        """
        last_token = None
        quote_count: int = 0
        # infer whitespace after field

        for token in self.tokens:
            if token.text == '"':
                quote_count += 1
                if quote_count % 2 != 0:
                    token.whitespace_after = False
                elif last_token is not None:
                    last_token.whitespace_after = False

            if last_token is not None:

                if token.text in [".", ":", ",", ";", ")", "n't", "!", "?"]:
                    last_token.whitespace_after = False

                if token.text.startswith("'"):
                    last_token.whitespace_after = False

            if token.text in ["("]:
                token.whitespace_after = False

            last_token = token
        return self

    def to_original_text(self) -> str:
        if len(self.tokens) > 0 and (self.tokens[0].start_pos is None):
            return " ".join([t.text for t in self.tokens])
        str = ""
        pos = 0
        for t in self.tokens:
            while t.start_pos > pos:
                str += " "
                pos += 1

            str += t.text
            pos += len(t.text)

        return str

    def to_dict(self, tag_type: str = None):
        labels = []
        entities = []

        if tag_type:
            entities = [span.to_dict() for span in self.get_spans(tag_type)]
        if self.labels:
            labels = [l.to_dict() for l in self.labels]

        return {"text": self.to_original_text(), "labels": labels, "entities": entities}

    def __getitem__(self, idx: int) -> Token:
        return self.tokens[idx]

    def __iter__(self):
        return iter(self.tokens)

    def __len__(self) -> int:
        return len(self.tokens)

    def __repr__(self):
        tagged_string = self.to_tagged_string()
        tokenized_string = self.to_tokenized_string()

        # add Sentence labels to output if they exist
        sentence_labels = (
            f"  − Sentence-Labels: {self.annotation_layers}"
            if self.annotation_layers != {}
            else ""
        )

        # add Token labels to output if they exist
        token_labels = (
            f'  − Token-Labels: "{tagged_string}"'
            if tokenized_string != tagged_string
            else ""
        )

        return f'Sentence: "{tokenized_string}"   [− Tokens: {len(self)}{token_labels}{sentence_labels}]'

    def __copy__(self):
        s = Sentence()
        for token in self.tokens:
            nt = Token(token.text)
            for tag_type in token.tags:
                nt.add_label(
                    tag_type,
                    token.get_tag(tag_type).value,
                    token.get_tag(tag_type).score,
                )

            s.add_token(nt)
        return s

    def __str__(self) -> str:

        tagged_string = self.to_tagged_string()
        tokenized_string = self.to_tokenized_string()

        # add Sentence labels to output if they exist
        sentence_labels = (
            f"  − Sentence-Labels: {self.annotation_layers}"
            if self.annotation_layers != {}
            else ""
        )

        # add Token labels to output if they exist
        token_labels = (
            f'  − Token-Labels: "{tagged_string}"'
            if tokenized_string != tagged_string
            else ""
        )

        return f'Sentence: "{tokenized_string}"   [− Tokens: {len(self)}{token_labels}{sentence_labels}]'

    def get_language_code(self) -> str:
        if self.language_code is None:
            try:
                self.language_code = langdetect.detect(self.to_plain_string())
            except:
                self.language_code = "en"

        return self.language_code

    @staticmethod
    def _restore_windows_1252_characters(text: str) -> str:
        def to_windows_1252(match):
            try:
                return bytes([ord(match.group(0))]).decode("windows-1252")
            except UnicodeDecodeError:
                # No character at the corresponding code point: remove it
                return ""

        return re.sub(r"[\u0080-\u0099]", to_windows_1252, text)

    def next_sentence(self):
        """
        Get the next sentence in the document (works only if context is set through dataloader or elsewhere)
        :return: next Sentence in document if set, otherwise None
        """
        if "_next_sentence" in self.__dict__.keys():
            return self._next_sentence

        if "_position_in_dataset" in self.__dict__.keys():
            dataset = self._position_in_dataset[0]
            index = self._position_in_dataset[1] + 1
            if index < len(dataset):
                return dataset[index]

        return None

    def previous_sentence(self):
        """
        Get the previous sentence in the document (works only if context is set through dataloader or elsewhere)
        :return: previous Sentence in document if set, otherwise None
        """
        if "_previous_sentence" in self.__dict__.keys():
            return self._previous_sentence

        if "_position_in_dataset" in self.__dict__.keys():
            dataset = self._position_in_dataset[0]
            index = self._position_in_dataset[1] - 1
            if index >= 0:
                return dataset[index]

        return None

    def is_context_set(self) -> bool:
        """
        Return True or False depending on whether context is set (for instance in dataloader or elsewhere)
        :return: True if context is set, else False
        """
        return (
            "_previous_sentence" in self.__dict__.keys()
            or "_position_in_dataset" in self.__dict__.keys()
        )


class Result(object):
    def __init__(
        self, main_score: float, log_header: str, log_line: str, detailed_results: str
    ):
        self.main_score: float = main_score
        self.log_header: str = log_header
        self.log_line: str = log_line
        self.detailed_results: str = detailed_results


class Metric(object):
    def __init__(self, name, beta=1):
        self.name = name
        self.beta = beta

        self._tps = defaultdict(int)
        self._fps = defaultdict(int)
        self._tns = defaultdict(int)
        self._fns = defaultdict(int)

    def add_tp(self, class_name):
        self._tps[class_name] += 1

    def add_tn(self, class_name):
        self._tns[class_name] += 1

    def add_fp(self, class_name):
        self._fps[class_name] += 1

    def add_fn(self, class_name):
        self._fns[class_name] += 1

    def get_tp(self, class_name=None):
        if class_name is None:
            return sum([self._tps[class_name] for class_name in self.get_classes()])
        return self._tps[class_name]

    def get_tn(self, class_name=None):
        if class_name is None:
            return sum([self._tns[class_name] for class_name in self.get_classes()])
        return self._tns[class_name]

    def get_fp(self, class_name=None):
        if class_name is None:
            return sum([self._fps[class_name] for class_name in self.get_classes()])
        return self._fps[class_name]

    def get_fn(self, class_name=None):
        if class_name is None:
            return sum([self._fns[class_name] for class_name in self.get_classes()])
        return self._fns[class_name]

    def precision(self, class_name=None):
        if self.get_tp(class_name) + self.get_fp(class_name) > 0:
            return self.get_tp(class_name) / (
                self.get_tp(class_name) + self.get_fp(class_name)
            )
        return 0.0

    def recall(self, class_name=None):
        if self.get_tp(class_name) + self.get_fn(class_name) > 0:
            return self.get_tp(class_name) / (
                self.get_tp(class_name) + self.get_fn(class_name)
            )
        return 0.0

    def f_score(self, class_name=None):
        if self.precision(class_name) + self.recall(class_name) > 0:
            return (
                (1 + self.beta * self.beta)
                * (self.precision(class_name) * self.recall(class_name))
                / (
                    self.precision(class_name) * self.beta * self.beta
                    + self.recall(class_name)
                )
            )
        return 0.0

    def accuracy(self, class_name=None):
        if (
            self.get_tp(class_name)
            + self.get_fp(class_name)
            + self.get_fn(class_name)
            + self.get_tn(class_name)
            > 0
        ):
            return (self.get_tp(class_name) + self.get_tn(class_name)) / (
                self.get_tp(class_name)
                + self.get_fp(class_name)
                + self.get_fn(class_name)
                + self.get_tn(class_name)
            )
        return 0.0

    def micro_avg_f_score(self):
        return self.f_score(None)

    def macro_avg_f_score(self):
        class_f_scores = [self.f_score(class_name) for class_name in self.get_classes()]
        if len(class_f_scores) == 0:
            return 0.0
        macro_f_score = sum(class_f_scores) / len(class_f_scores)
        return macro_f_score

    def micro_avg_accuracy(self):
        return self.accuracy(None)

    def macro_avg_accuracy(self):
        class_accuracy = [
            self.accuracy(class_name) for class_name in self.get_classes()
        ]

        if len(class_accuracy) > 0:
            return sum(class_accuracy) / len(class_accuracy)

        return 0.0

    def get_classes(self) -> List:
        all_classes = set(
            itertools.chain(
                *[
                    list(keys)
                    for keys in [
                        self._tps.keys(),
                        self._fps.keys(),
                        self._tns.keys(),
                        self._fns.keys(),
                    ]
                ]
            )
        )
        all_classes = [
            class_name for class_name in all_classes if class_name is not None
        ]
        all_classes.sort()
        return all_classes

    def to_tsv(self):
        return "{}\t{}\t{}\t{}".format(
            self.precision(), self.recall(), self.accuracy(), self.micro_avg_f_score()
        )

    @staticmethod
    def tsv_header(prefix=None):
        if prefix:
            return "{0}_PRECISION\t{0}_RECALL\t{0}_ACCURACY\t{0}_F-SCORE".format(prefix)

        return "PRECISION\tRECALL\tACCURACY\tF-SCORE"

    @staticmethod
    def to_empty_tsv():
        return "\t_\t_\t_\t_"

    def __str__(self):
        all_classes = self.get_classes()
        all_classes = [None] + all_classes
        all_lines = [
            "{0:<10}\ttp: {1} - fp: {2} - fn: {3} - tn: {4} - precision: {5:.4f} - recall: {6:.4f} - accuracy: {7:.4f} - f1-score: {8:.4f}".format(
                self.name if class_name is None else class_name,
                self.get_tp(class_name),
                self.get_fp(class_name),
                self.get_fn(class_name),
                self.get_tn(class_name),
                self.precision(class_name),
                self.recall(class_name),
                self.accuracy(class_name),
                self.f_score(class_name),
            )
            for class_name in all_classes
        ]
        return "\n".join(all_lines)


class FlairDataset(Dataset):
    @abstractmethod
    def is_in_memory(self) -> bool:
        pass


class SentenceDataset(FlairDataset):
    """
    A simple Dataset object to wrap a List of Sentence
    """

    def __init__(self, sentences: Union[Sentence, List[Sentence]]):
        """
        Instantiate SentenceDataset
        :param sentences: Sentence or List of Sentence that make up SentenceDataset
        """
        # cast to list if necessary
        if type(sentences) == Sentence:
            sentences = [sentences]
        self.sentences: List[Sentence] = sentences

    def is_in_memory(self) -> bool:
        return True

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index: int = 0) -> Sentence:
        return self.sentences[index]


class DataLoader(torch.utils.data.dataloader.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=8,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):

        # in certain cases, multi-CPU data loading makes no sense and slows
        # everything down. For this reason, we detect if a dataset is in-memory:
        # if so, num_workers is set to 0 for faster processing
        flair_dataset = dataset
        while True:
            if type(flair_dataset) is Subset:
                flair_dataset = flair_dataset.dataset
            elif type(flair_dataset) is ConcatDataset:
                flair_dataset = flair_dataset.datasets[0]
            else:
                break

        if type(flair_dataset) is list:
            num_workers = 0
        elif isinstance(flair_dataset, FlairDataset) and flair_dataset.is_in_memory():
            num_workers = 0

        super(DataLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=list,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


class FlairModel(torch.nn.Module):
    """Abstract base class for all downstream task models in Flair, such as SequenceTagger and TextClassifier.
    Every new type of model must implement these methods."""

    @abstractmethod
    def forward_loss(
        self, data_points: Union[List[DataPoint], DataPoint]
    ) -> torch.tensor:
        """Performs a forward pass and returns a loss tensor for backpropagation. Implement this to enable training."""

    @abstractmethod
    def evaluate(
        self,
        sentences: Union[List[DataPoint], Dataset],
        out_path: Path = None,
        embedding_storage_mode: str = "none",
    ) -> Tuple[Result, float]:
        """Evaluates the model. Returns a Result object containing evaluation
        results and a loss value. Implement this to enable evaluation.
        :param data_loader: DataLoader that iterates over dataset to be evaluated
        :param out_path: Optional output path to store predictions
        :param embedding_storage_mode: One of 'none', 'cpu' or 'gpu'. 'none' means all embeddings are deleted and
        freshly recomputed, 'cpu' means all embeddings are stored on CPU, or 'gpu' means all embeddings are stored on GPU
        :return: Returns a Tuple consisting of a Result object and a loss float value
        """

    @abstractmethod
    def _get_state_dict(self):
        """Returns the state dictionary for this model. Implementing this enables the save() and save_checkpoint()
        functionality."""

    @staticmethod
    @abstractmethod
    def _init_model_with_state_dict(state) -> torch.nn.Module:
        """Initialize the model from a state dictionary. Implementing this enables the load() and load_checkpoint()
        functionality."""

    @staticmethod
    @abstractmethod
    def _fetch_model(model_name) -> str:
        return model_name

    def save(self, model_file: Union[str, Path]):
        """
        Saves the current model to the provided file.
        :param model_file: the model file
        """
        model_state = self._get_state_dict()

        torch.save(model_state, str(model_file), pickle_protocol=4)

    @staticmethod
    def _load_big_file(f: str) -> mmap.mmap:
        """
        Workaround for loading a big pickle file. Files over 2GB cause pickle errors on certin Mac and Windows distributions.
        :param f:
        :return:
        """
        logger.info(f"loading file {f}")
        with open(f, "rb") as f_in:
            # mmap seems to be much more memory efficient
            bf = mmap.mmap(f_in.fileno(), 0, access=mmap.ACCESS_READ)
            f_in.close()
        return bf

    @classmethod
    def load(cls, model: Union[str, Path]):
        """
        Loads the model from the given file.
        :param model: the model file
        :return: the loaded text classifier model
        """
        model_file = cls._fetch_model(str(model))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # load_big_file is a workaround by https://github.com/highway11git to load models on some Mac/Windows setups
            # see https://github.com/zalandoresearch/flair/issues/351
            f = cls._load_big_file(str(model_file))
            state = torch.load(f, map_location="cpu")

        m = cls._init_model_with_state_dict(state)

        m.eval()
        m.to(flair_device)

        return m


class LockedDropout(torch.nn.Module):
    """
    Implementation of locked (or variational) dropout. Randomly drops out entire parameters in embedding space.
    """

    def __init__(self, dropout_rate=0.5, batch_first=True, inplace=False):
        super(LockedDropout, self).__init__()
        self.dropout_rate = dropout_rate
        self.batch_first = batch_first
        self.inplace = inplace

    def forward(self, x):
        if not self.training or not self.dropout_rate:
            return x

        if not self.batch_first:
            m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.dropout_rate)
        else:
            m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - self.dropout_rate)

        mask = torch.autograd.Variable(m, requires_grad=False) / (1 - self.dropout_rate)
        mask = mask.expand_as(x)
        return mask * x

    def extra_repr(self):
        inplace_str = ", inplace" if self.inplace else ""
        return "p={}{}".format(self.dropout_rate, inplace_str)


class WordDropout(torch.nn.Module):
    """
    Implementation of word dropout. Randomly drops out entire words (or characters) in embedding space.
    """

    def __init__(self, dropout_rate=0.05, inplace=False):
        super(WordDropout, self).__init__()
        self.dropout_rate = dropout_rate
        self.inplace = inplace

    def forward(self, x):
        if not self.training or not self.dropout_rate:
            return x

        m = x.data.new(x.size(0), x.size(1), 1).bernoulli_(1 - self.dropout_rate)

        mask = torch.autograd.Variable(m, requires_grad=False)
        return mask * x

    def extra_repr(self):
        inplace_str = ", inplace" if self.inplace else ""
        return "p={}{}".format(self.dropout_rate, inplace_str)


class Embeddings(torch.nn.Module):
    """Abstract base class for all embeddings. Every new type of embedding must implement these methods."""

    def __init__(self):
        """Set some attributes that would otherwise result in errors. Overwrite these in your embedding class."""
        if not hasattr(self, "name"):
            self.name: str = "unnamed_embedding"
        if not hasattr(self, "static_embeddings"):
            # if the embeddings for a sentence are the same in each epoch, set this to True for improved efficiency
            self.static_embeddings = False
        super().__init__()

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""

    @property
    @abstractmethod
    def embedding_type(self) -> str:
        pass

    def embed(self, sentences: Union[Sentence, List[Sentence]]) -> List[Sentence]:
        """Add embeddings to all words in a list of sentences. If embeddings are already added, updates only if embeddings
        are non-static."""

        # if only one sentence is passed, convert to list of sentence
        sentences_list: List[Sentence] = []
        if type(sentences) is Sentence:  #  or (type(sentences) is Image)
            sentences_list = [sentences]

        everything_embedded: bool = True

        if self.embedding_type == "word-level":
            for sentence in sentences:
                for token in sentence.tokens:
                    if self.name not in token._embeddings.keys():
                        everything_embedded = False
                        break
        else:
            for sentence in sentences:
                if self.name not in sentence._embeddings.keys():
                    everything_embedded = False
                    break

        if not everything_embedded or not self.static_embeddings:
            self._add_embeddings_internal(sentences_list)

        return sentences_list

    @abstractmethod
    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        """Private method for adding embeddings to all words in a list of sentences."""

    def get_names(self) -> List[str]:
        """Returns a list of embedding names. In most cases, it is just a list with one item, namely the name of
        this embedding. But in some cases, the embedding is made up by different embeddings (StackedEmbedding).
        Then, the list contains the names of all embeddings in the stack."""
        return [self.name]

    def get_named_embeddings_dict(self) -> Dict:
        return {self.name: self}


class TokenEmbeddings(Embeddings):
    """Abstract base class for all token-level embeddings. Ever new type of word embedding must implement these methods."""

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""

    @property
    def embedding_type(self) -> str:
        return "word-level"

    @staticmethod
    def get_instance_parameters(locals: dict) -> dict:
        class_definition = locals.get("__class__")
        instance_parameters = set(
            inspect.getfullargspec(class_definition.__init__).args
        )
        instance_parameters.difference_update(set(["self"]))
        instance_parameters.update(set(["__class__"]))
        instance_parameters = {
            class_attribute: attribute_value
            for class_attribute, attribute_value in locals.items()
            if class_attribute in instance_parameters
        }
        return instance_parameters


class StackedEmbeddings(TokenEmbeddings):
    """A stack of embeddings, used if you need to combine several different embedding types."""

    def __init__(self, embeddings: List[TokenEmbeddings]):
        """The constructor takes a list of embeddings to be combined."""
        super().__init__()

        self.embeddings = embeddings

        # IMPORTANT: add embeddings as torch modules
        for i, embedding in enumerate(embeddings):
            embedding.name = f"{str(i)}-{embedding.name}"
            self.add_module(f"list_embedding_{str(i)}", embedding)

        self.name: str = "Stack"
        self.static_embeddings: bool = True

        self.__embedding_type: str = embeddings[0].embedding_type

        self.__embedding_length: int = 0
        for embedding in embeddings:
            self.__embedding_length += embedding.embedding_length

    def embed(
        self, sentences: Union[Sentence, List[Sentence]], static_embeddings: bool = True
    ):
        # if only one sentence is passed, convert to list of sentence
        if type(sentences) is Sentence:
            sentences = [sentences]

        for embedding in self.embeddings:
            embedding.embed(sentences)

    @property
    def embedding_type(self) -> str:
        return self.__embedding_type

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        for embedding in self.embeddings:
            embedding._add_embeddings_internal(sentences)

        return sentences

    def __str__(self):
        return f'StackedEmbeddings [{",".join([str(e) for e in self.embeddings])}]'

    def get_names(self) -> List[str]:
        """Returns a list of embedding names. In most cases, it is just a list with one item, namely the name of
        this embedding. But in some cases, the embedding is made up by different embeddings (StackedEmbedding).
        Then, the list contains the names of all embeddings in the stack."""
        names = []
        for embedding in self.embeddings:
            names.extend(embedding.get_names())
        return names

    def get_named_embeddings_dict(self) -> Dict:

        named_embeddings_dict = {}
        for embedding in self.embeddings:
            named_embeddings_dict.update(embedding.get_named_embeddings_dict())

        return named_embeddings_dict


class SequenceTagger(FlairModel):
    def __init__(
        self,
        hidden_size: int,
        embeddings: TokenEmbeddings,
        tag_dictionary: Dictionary,
        tag_type: str,
        use_crf: bool = True,
        use_rnn: bool = True,
        rnn_layers: int = 1,
        dropout: float = 0.0,
        word_dropout: float = 0.05,
        locked_dropout: float = 0.5,
        reproject_embeddings: Union[bool, int] = True,
        train_initial_hidden_state: bool = False,
        rnn_type: str = "LSTM",
        pickle_module: str = "pickle",
        beta: float = 1.0,
        loss_weights: Dict[str, float] = None,
    ):
        """
        Initializes a SequenceTagger
        :param hidden_size: number of hidden states in RNN
        :param embeddings: word embeddings used in tagger
        :param tag_dictionary: dictionary of tags you want to predict
        :param tag_type: string identifier for tag type
        :param use_crf: if True use CRF decoder, else project directly to tag space
        :param use_rnn: if True use RNN layer, otherwise use word embeddings directly
        :param rnn_layers: number of RNN layers
        :param dropout: dropout probability
        :param word_dropout: word dropout probability
        :param reproject_embeddings: if True, adds trainable linear map on top of embedding layer. If False, no map.
        If you set this to an integer, you can control the dimensionality of the reprojection layer
        :param locked_dropout: locked dropout probability
        :param train_initial_hidden_state: if True, trains initial hidden state of RNN
        :param beta: Parameter for F-beta score for evaluation and training annealing
        :param loss_weights: Dictionary of weights for classes (tags) for the loss function
        (if any tag's weight is unspecified it will default to 1.0)

        """

        super(SequenceTagger, self).__init__()
        self.use_rnn = use_rnn
        self.hidden_size = hidden_size
        self.use_crf: bool = use_crf
        self.rnn_layers: int = rnn_layers

        self.trained_epochs: int = 0

        self.embeddings = embeddings

        # set the dictionaries
        self.tag_dictionary: Dictionary = tag_dictionary
        # if we use a CRF, we must add special START and STOP tags to the dictionary
        if use_crf:
            self.tag_dictionary.add_item(START_TAG)
            self.tag_dictionary.add_item(STOP_TAG)

        self.tag_type: str = tag_type
        self.tagset_size: int = len(tag_dictionary)

        self.beta = beta

        self.weight_dict = loss_weights
        # Initialize the weight tensor
        if loss_weights is not None:
            n_classes = len(self.tag_dictionary)
            weight_list = [1.0 for i in range(n_classes)]
            for i, tag in enumerate(self.tag_dictionary.get_items()):
                if tag in loss_weights.keys():
                    weight_list[i] = loss_weights[tag]
            self.loss_weights = torch.FloatTensor(weight_list).to(flair_device)
        else:
            self.loss_weights = None

        # initialize the network architecture
        self.nlayers: int = rnn_layers
        self.hidden_word = None

        # dropouts
        self.use_dropout: float = dropout
        self.use_word_dropout: float = word_dropout
        self.use_locked_dropout: float = locked_dropout

        self.pickle_module = pickle_module

        if dropout > 0.0:
            self.dropout = torch.nn.Dropout(dropout)

        if word_dropout > 0.0:
            self.word_dropout = WordDropout(word_dropout)

        if locked_dropout > 0.0:
            self.locked_dropout = LockedDropout(locked_dropout)

        embedding_dim: int = self.embeddings.embedding_length
        rnn_input_dim: int = embedding_dim

        # optional reprojection layer on top of word embeddings
        self.reproject_embeddings = reproject_embeddings
        if self.reproject_embeddings:
            if type(self.reproject_embeddings) == int:
                rnn_input_dim = self.reproject_embeddings

            self.embedding2nn = torch.nn.Linear(embedding_dim, rnn_input_dim)

        self.train_initial_hidden_state = train_initial_hidden_state
        self.bidirectional = True
        self.rnn_type = rnn_type

        # bidirectional LSTM on top of embedding layer
        if self.use_rnn:
            num_directions = 2 if self.bidirectional else 1

            if self.rnn_type in ["LSTM", "GRU"]:

                self.rnn = getattr(torch.nn, self.rnn_type)(
                    rnn_input_dim,
                    hidden_size,
                    num_layers=self.nlayers,
                    dropout=0.0 if self.nlayers == 1 else 0.5,
                    bidirectional=True,
                    batch_first=True,
                )
                # Create initial hidden state and initialize it
                if self.train_initial_hidden_state:
                    self.hs_initializer = torch.nn.init.xavier_normal_

                    self.lstm_init_h = Parameter(
                        torch.randn(self.nlayers * num_directions, self.hidden_size),
                        requires_grad=True,
                    )

                    self.lstm_init_c = Parameter(
                        torch.randn(self.nlayers * num_directions, self.hidden_size),
                        requires_grad=True,
                    )

                    # TODO: Decide how to initialize the hidden state variables
                    # self.hs_initializer(self.lstm_init_h)
                    # self.hs_initializer(self.lstm_init_c)

            # final linear map to tag space
            self.linear = torch.nn.Linear(
                hidden_size * num_directions, len(tag_dictionary)
            )
        else:
            self.linear = torch.nn.Linear(
                self.embeddings.embedding_length, len(tag_dictionary)
            )

        if self.use_crf:
            self.transitions = torch.nn.Parameter(
                torch.randn(self.tagset_size, self.tagset_size)
            )

            self.transitions.detach()[
                self.tag_dictionary.get_idx_for_item(START_TAG), :
            ] = -10000

            self.transitions.detach()[
                :, self.tag_dictionary.get_idx_for_item(STOP_TAG)
            ] = -10000

        self.to(flair_device)

    def _get_state_dict(self):
        model_state = {
            "state_dict": self.state_dict(),
            "embeddings": self.embeddings,
            "hidden_size": self.hidden_size,
            "train_initial_hidden_state": self.train_initial_hidden_state,
            "tag_dictionary": self.tag_dictionary,
            "tag_type": self.tag_type,
            "use_crf": self.use_crf,
            "use_rnn": self.use_rnn,
            "rnn_layers": self.rnn_layers,
            "use_dropout": self.use_dropout,
            "use_word_dropout": self.use_word_dropout,
            "use_locked_dropout": self.use_locked_dropout,
            "rnn_type": self.rnn_type,
            "beta": self.beta,
            "weight_dict": self.weight_dict,
            "reproject_embeddings": self.reproject_embeddings,
        }
        return model_state

    @staticmethod
    def _init_model_with_state_dict(state):

        rnn_type = "LSTM" if "rnn_type" not in state.keys() else state["rnn_type"]
        use_dropout = 0.0 if "use_dropout" not in state.keys() else state["use_dropout"]
        use_word_dropout = (
            0.0 if "use_word_dropout" not in state.keys() else state["use_word_dropout"]
        )
        use_locked_dropout = (
            0.0
            if "use_locked_dropout" not in state.keys()
            else state["use_locked_dropout"]
        )
        train_initial_hidden_state = (
            False
            if "train_initial_hidden_state" not in state.keys()
            else state["train_initial_hidden_state"]
        )
        beta = 1.0 if "beta" not in state.keys() else state["beta"]
        weights = None if "weight_dict" not in state.keys() else state["weight_dict"]
        reproject_embeddings = (
            True
            if "reproject_embeddings" not in state.keys()
            else state["reproject_embeddings"]
        )
        if "reproject_to" in state.keys():
            reproject_embeddings = state["reproject_to"]

        model = SequenceTagger(
            hidden_size=state["hidden_size"],
            embeddings=state["embeddings"],
            tag_dictionary=state["tag_dictionary"],
            tag_type=state["tag_type"],
            use_crf=state["use_crf"],
            use_rnn=state["use_rnn"],
            rnn_layers=state["rnn_layers"],
            dropout=use_dropout,
            word_dropout=use_word_dropout,
            locked_dropout=use_locked_dropout,
            train_initial_hidden_state=train_initial_hidden_state,
            rnn_type=rnn_type,
            beta=beta,
            loss_weights=weights,
            reproject_embeddings=reproject_embeddings,
        )
        model.load_state_dict(state["state_dict"])
        return model

    @staticmethod
    def _store_embeddings(sentences: List[Sentence], storage_mode: str):

        # if memory mode option 'none' delete everything
        if storage_mode == "none":
            for sentence in sentences:
                sentence.clear_embeddings()

        # else delete only dynamic embeddings (otherwise autograd will keep everything in memory)
        else:
            # find out which ones are dynamic embeddings
            delete_keys = []
            if type(sentences[0]) == Sentence:
                for name, vector in sentences[0][0]._embeddings.items():
                    if sentences[0][0]._embeddings[name].requires_grad:
                        delete_keys.append(name)

            # find out which ones are dynamic embeddings
            for sentence in sentences:
                sentence.clear_embeddings(delete_keys)

        # memory management - option 1: send everything to CPU (pin to memory if we train on GPU)
        if storage_mode == "cpu":
            pin_memory = False if str(flair_device) == "cpu" else True
            for sentence in sentences:
                sentence.to("cpu", pin_memory=pin_memory)

        # record current embedding storage mode to allow optimization (for instance in FlairEmbeddings class)

    def predict(
        self,
        sentences: Union[List[Sentence], Sentence],
        mini_batch_size=32,
        all_tag_prob: bool = False,
        verbose: bool = False,
        label_name: Optional[str] = None,
        return_loss=False,
        embedding_storage_mode="none",
    ):
        """
        Predict sequence tags for Named Entity Recognition task
        :param sentences: a Sentence or a List of Sentence
        :param mini_batch_size: size of the minibatch, usually bigger is more rapid but consume more memory,
        up to a point when it has no more effect.
        :param all_tag_prob: True to compute the score for each tag on each token,
        otherwise only the score of the best tag is returned
        :param verbose: set to True to display a progress bar
        :param return_loss: set to True to return loss
        :param label_name: set this to change the name of the label type that is predicted
        :param embedding_storage_mode: default is 'none' which is always best. Only set to 'cpu' or 'gpu' if
        you wish to not only predict, but also keep the generated embeddings in CPU or GPU memory respectively.
        'gpu' to store embeddings in GPU memory.
        """
        if label_name == None:
            label_name = self.tag_type

        with torch.no_grad():
            if not sentences:
                return sentences
            sentences_list: List[Sentence] = []
            if isinstance(sentences, Sentence):
                sentences_list = [sentences]

            # set context if not set already
            previous_sentence = None
            for sentence in sentences_list:
                if sentence.is_context_set():
                    continue
                sentence._previous_sentence = previous_sentence
                sentence._next_sentence = None
                if previous_sentence:
                    previous_sentence._next_sentence = sentence
                previous_sentence = sentence

            # reverse sort all sequences by their length
            rev_order_len_index = sorted(
                range(len(sentences_list)),
                key=lambda k: len(sentences_list[k]),
                reverse=True,
            )

            reordered_sentences: List[Sentence] = [
                sentences_list[index] for index in rev_order_len_index
            ]

            dataloader = DataLoader(
                dataset=SentenceDataset(reordered_sentences), batch_size=mini_batch_size
            )

            if self.use_crf:
                transitions = self.transitions.detach().cpu().numpy()
            else:
                transitions = None

            # progress bar for verbosity
            if verbose:
                dataloader = tqdm(dataloader)

            overall_loss = 0
            batch_no = 0
            for batch in dataloader:

                batch_no += 1

                if verbose:
                    dataloader.set_description(f"Inferencing on batch {batch_no}")

                batch = self._filter_empty_sentences(batch)
                # stop if all sentences are empty
                if not batch:
                    continue

                feature = self.forward(batch)

                if return_loss:
                    overall_loss += self._calculate_loss(feature, batch)

                tags, all_tags = self._obtain_labels(
                    feature=feature,
                    batch_sentences=batch,
                    transitions=transitions,
                    get_all_tags=all_tag_prob,
                )

                for (sentence, sent_tags) in zip(batch, tags):
                    for (token, tag) in zip(sentence.tokens, sent_tags):
                        token.add_tag_label(label_name, tag)

                # all_tags will be empty if all_tag_prob is set to False, so the for loop will be avoided
                for (sentence, sent_all_tags) in zip(batch, all_tags):
                    for (token, token_all_tags) in zip(sentence.tokens, sent_all_tags):
                        token.add_tags_proba_dist(label_name, token_all_tags)

                # clearing token embeddings to save memory
                self._store_embeddings(batch, storage_mode=embedding_storage_mode)

            if return_loss:
                return overall_loss / batch_no

    def _requires_span_F1_evaluation(self) -> bool:
        span_F1 = False
        for item in self.tag_dictionary.get_items():
            if item.startswith("B-"):
                span_F1 = True
        return span_F1

    def _evaluate_with_span_F1(
        self, data_loader, embedding_storage_mode, mini_batch_size, out_path
    ):
        eval_loss = 0

        batch_no: int = 0

        metric = Metric("Evaluation", beta=self.beta)

        lines: List[str] = []

        y_true = []
        y_pred = []

        for batch in data_loader:

            # predict for batch
            loss = self.predict(
                batch,
                embedding_storage_mode=embedding_storage_mode,
                mini_batch_size=mini_batch_size,
                label_name="predicted",
                return_loss=True,
            )
            eval_loss += loss
            batch_no += 1

            for sentence in batch:

                # make list of gold tags
                gold_spans = sentence.get_spans(self.tag_type)
                gold_tags = [(span.tag, repr(span)) for span in gold_spans]

                # make list of predicted tags
                predicted_spans = sentence.get_spans("predicted")
                predicted_tags = [(span.tag, repr(span)) for span in predicted_spans]

                # check for true positives, false positives and false negatives
                for tag, prediction in predicted_tags:
                    if (tag, prediction) in gold_tags:
                        metric.add_tp(tag)
                    else:
                        metric.add_fp(tag)

                for tag, gold in gold_tags:
                    if (tag, gold) not in predicted_tags:
                        metric.add_fn(tag)

                tags_gold = []
                tags_pred = []

                # also write to file in BIO format to use old conlleval script
                if out_path:
                    for token in sentence:
                        # check if in gold spans
                        gold_tag = "O"
                        for span in gold_spans:
                            if token in span:
                                gold_tag = (
                                    "B-" + span.tag
                                    if token == span[0]
                                    else "I-" + span.tag
                                )
                        tags_gold.append(gold_tag)

                        predicted_tag = "O"
                        # check if in predicted spans
                        for span in predicted_spans:
                            if token in span:
                                predicted_tag = (
                                    "B-" + span.tag
                                    if token == span[0]
                                    else "I-" + span.tag
                                )
                        tags_pred.append(predicted_tag)

                        lines.append(f"{token.text} {gold_tag} {predicted_tag}\n")
                    lines.append("\n")

                y_true.append(tags_gold)
                y_pred.append(tags_pred)

        if out_path:
            with open(Path(out_path), "w", encoding="utf-8") as outfile:
                outfile.write("".join(lines))

        eval_loss /= batch_no

        detailed_result = (
            "\nResults:"
            f"\n- F1-score (micro) {metric.micro_avg_f_score():.4f}"
            f"\n- F1-score (macro) {metric.macro_avg_f_score():.4f}"
            "\n\nBy class:"
        )

        for class_name in metric.get_classes():
            detailed_result += (
                f"\n{class_name:<10} tp: {metric.get_tp(class_name)} - fp: {metric.get_fp(class_name)} - "
                f"fn: {metric.get_fn(class_name)} - precision: "
                f"{metric.precision(class_name):.4f} - recall: {metric.recall(class_name):.4f} - "
                f"f1-score: "
                f"{metric.f_score(class_name):.4f}"
            )

        result = Result(
            main_score=metric.micro_avg_f_score(),
            log_line=f"{metric.precision():.4f}\t{metric.recall():.4f}\t{metric.micro_avg_f_score():.4f}",
            log_header="PRECISION\tRECALL\tF1",
            detailed_results=detailed_result,
        )

        return result, eval_loss

    def evaluate(
        self,
        sentences: Union[List[Sentence], Dataset],
        out_path: Union[str, Path] = None,
        embedding_storage_mode: str = "none",
        mini_batch_size: int = 32,
        num_workers: int = 8,
        wsd_evaluation: bool = False,
    ) -> Tuple[Result, float]:

        # read Dataset into data loader (if list of sentences passed, make Dataset first)
        if not isinstance(sentences, Dataset):
            sentences = SentenceDataset(sentences)
        data_loader = DataLoader(
            sentences, batch_size=mini_batch_size, num_workers=num_workers
        )

        # if span F1 needs to be used, use separate eval method
        if self._requires_span_F1_evaluation() and not wsd_evaluation:
            return self._evaluate_with_span_F1(
                data_loader, embedding_storage_mode, mini_batch_size, out_path
            )

        # else, use scikit-learn to evaluate
        y_true = []
        y_pred = []
        labels = Dictionary(add_unk=False)

        eval_loss = 0
        batch_no: int = 0

        lines: List[str] = []

        for batch in tqdm(data_loader):  # NOTE: edited

            # predict for batch
            loss = self.predict(
                batch,
                embedding_storage_mode=embedding_storage_mode,
                mini_batch_size=mini_batch_size,
                label_name="predicted",
                return_loss=True,
            )
            eval_loss += loss
            batch_no += 1

            for sentence in batch:

                for token in sentence:
                    # add gold tag
                    gold_tag = token.get_tag(self.tag_type).value
                    y_true.append(labels.add_item(gold_tag))

                    # add predicted tag
                    if wsd_evaluation:
                        if gold_tag == "O":
                            predicted_tag = "O"
                        else:
                            predicted_tag = token.get_tag("predicted").value
                    else:
                        predicted_tag = token.get_tag("predicted").value

                    y_pred.append(labels.add_item(predicted_tag))

                    # for file output
                    lines.append(f"{token.text} {gold_tag} {predicted_tag}\n")

                lines.append("\n")

        if out_path:
            with open(Path(out_path), "w", encoding="utf-8") as outfile:
                outfile.write("".join(lines))

        eval_loss /= batch_no

        # make "classification report"
        target_names = []
        labels_to_report = []
        all_labels = []
        all_indices = []
        for i in range(len(labels)):
            label = labels.get_item_for_index(i)
            all_labels.append(label)
            all_indices.append(i)
            if label == "_" or label == "":
                continue
            target_names.append(label)
            labels_to_report.append(i)

        # report over all in case there are no labels
        if not labels_to_report:
            target_names = all_labels
            labels_to_report = all_indices

        classification_report = metrics.classification_report(
            y_true,
            y_pred,
            digits=4,
            target_names=target_names,
            zero_division=1,
            labels=labels_to_report,
        )

        # get scores
        micro_f_score = round(
            metrics.fbeta_score(
                y_true, y_pred, beta=self.beta, average="micro", labels=labels_to_report
            ),
            4,
        )
        macro_f_score = round(
            metrics.fbeta_score(
                y_true, y_pred, beta=self.beta, average="macro", labels=labels_to_report
            ),
            4,
        )
        accuracy_score = round(metrics.accuracy_score(y_true, y_pred), 4)

        detailed_result = (
            "\nResults:"
            f"\n- F-score (micro): {micro_f_score}"
            f"\n- F-score (macro): {macro_f_score}"
            f"\n- Accuracy (incl. no class): {accuracy_score}"
            "\n\nBy class:\n" + classification_report
        )

        # line for log file
        log_header = "ACCURACY"
        log_line = f"\t{accuracy_score}"

        result = Result(
            main_score=micro_f_score,
            log_line=log_line,
            log_header=log_header,
            detailed_results=detailed_result,
        )
        return result, eval_loss

    def forward_loss(self, data_points: List[Sentence], sort=True) -> torch.tensor:
        features = self.forward(data_points)
        return self._calculate_loss(features, data_points)

    def forward(self, sentences: List[Sentence]):

        self.embeddings.embed(sentences)

        names = self.embeddings.get_names()

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        longest_token_sequence_in_batch: int = max(lengths)

        pre_allocated_zero_tensor = torch.zeros(
            self.embeddings.embedding_length * longest_token_sequence_in_batch,
            dtype=torch.float,
            device=flair_device,
        )

        all_embs = list()
        for sentence in sentences:
            all_embs += [
                emb for token in sentence for emb in token.get_each_embedding(names)
            ]
            nb_padding_tokens = longest_token_sequence_in_batch - len(sentence)

            if nb_padding_tokens > 0:
                t = pre_allocated_zero_tensor[
                    : self.embeddings.embedding_length * nb_padding_tokens
                ]
                all_embs.append(t)

        sentence_tensor = torch.cat(all_embs).view(
            [
                len(sentences),
                longest_token_sequence_in_batch,
                self.embeddings.embedding_length,
            ]
        )

        # --------------------------------------------------------------------
        # FF PART
        # --------------------------------------------------------------------
        if self.use_dropout > 0.0:
            sentence_tensor = self.dropout(sentence_tensor)
        if self.use_word_dropout > 0.0:
            sentence_tensor = self.word_dropout(sentence_tensor)
        if self.use_locked_dropout > 0.0:
            sentence_tensor = self.locked_dropout(sentence_tensor)

        if self.reproject_embeddings:
            sentence_tensor = self.embedding2nn(sentence_tensor)

        if self.use_rnn:
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                sentence_tensor, lengths, enforce_sorted=False, batch_first=True
            )

            # if initial hidden state is trainable, use this state
            if self.train_initial_hidden_state:
                initial_hidden_state = [
                    self.lstm_init_h.unsqueeze(1).repeat(1, len(sentences), 1),
                    self.lstm_init_c.unsqueeze(1).repeat(1, len(sentences), 1),
                ]
                rnn_output, hidden = self.rnn(packed, initial_hidden_state)
            else:
                rnn_output, hidden = self.rnn(packed)

            sentence_tensor, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
                rnn_output, batch_first=True
            )

            if self.use_dropout > 0.0:
                sentence_tensor = self.dropout(sentence_tensor)
            # word dropout only before LSTM - TODO: more experimentation needed
            # if self.use_word_dropout > 0.0:
            #     sentence_tensor = self.word_dropout(sentence_tensor)
            if self.use_locked_dropout > 0.0:
                sentence_tensor = self.locked_dropout(sentence_tensor)

        features = self.linear(sentence_tensor)

        return features

    def _score_sentence(self, feats, tags, lens_):

        start = torch.tensor(
            [self.tag_dictionary.get_idx_for_item(START_TAG)], device=flair_device
        )
        start = start[None, :].repeat(tags.shape[0], 1)

        stop = torch.tensor(
            [self.tag_dictionary.get_idx_for_item(STOP_TAG)], device=flair_device
        )
        stop = stop[None, :].repeat(tags.shape[0], 1)

        pad_start_tags = torch.cat([start, tags], 1)
        pad_stop_tags = torch.cat([tags, stop], 1)

        for i in range(len(lens_)):
            pad_stop_tags[i, lens_[i] :] = self.tag_dictionary.get_idx_for_item(
                STOP_TAG
            )

        score = torch.FloatTensor(feats.shape[0]).to(flair_device)

        for i in range(feats.shape[0]):
            r = torch.LongTensor(range(lens_[i])).to(flair_device)

            score[i] = torch.sum(
                self.transitions[
                    pad_stop_tags[i, : lens_[i] + 1], pad_start_tags[i, : lens_[i] + 1]
                ]
            ) + torch.sum(feats[i, r, tags[i, : lens_[i]]])

        return score

    @staticmethod
    def _pad_tensors(tensor_list):
        ml = max([x.shape[0] for x in tensor_list])
        shape = [len(tensor_list), ml] + list(tensor_list[0].shape[1:])
        template = torch.zeros(*shape, dtype=torch.long, device=flair_device)
        lens_ = [x.shape[0] for x in tensor_list]
        for i, tensor in enumerate(tensor_list):
            template[i, : lens_[i]] = tensor

        return template, lens_

    def _calculate_loss(
        self, features: torch.tensor, sentences: List[Sentence]
    ) -> float:

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]

        tag_list: List = []
        for s_id, sentence in enumerate(sentences):
            # get the tags in this sentence
            tag_idx: List[int] = [
                self.tag_dictionary.get_idx_for_item(token.get_tag(self.tag_type).value)
                for token in sentence
            ]
            # add tags as tensor
            tag = torch.tensor(tag_idx, device=flair_device)
            tag_list.append(tag)

        if self.use_crf:
            # pad tags if using batch-CRF decoder
            tags, _ = self._pad_tensors(tag_list)

            forward_score = self._forward_alg(features, lengths)
            gold_score = self._score_sentence(features, tags, lengths)

            score = forward_score - gold_score

            return score.mean()

        else:
            score = 0
            for sentence_feats, sentence_tags, sentence_length in zip(
                features, tag_list, lengths
            ):
                sentence_feats = sentence_feats[:sentence_length]
                score += torch.nn.functional.cross_entropy(
                    sentence_feats, sentence_tags, weight=self.loss_weights
                )
            score /= len(features)
            return score

    def _obtain_labels(
        self,
        feature: torch.Tensor,
        batch_sentences: List[Sentence],
        transitions: Optional[np.ndarray],
        get_all_tags: bool,
    ) -> Tuple[List[List[Label]], List[List[List[Label]]]]:
        """
        Returns a tuple of two lists:
         - The first list corresponds to the most likely `Label` per token in each sentence.
         - The second list contains a probability distribution over all `Labels` for each token
           in a sentence for all sentences.
        """

        lengths: List[int] = [len(sentence.tokens) for sentence in batch_sentences]

        tags = []
        all_tags = []
        feature = feature.cpu()
        if self.use_crf:
            feature = feature.numpy()
        else:
            for index, length in enumerate(lengths):
                feature[index, length:] = 0
            softmax_batch = F.softmax(feature, dim=2).cpu()
            scores_batch, prediction_batch = torch.max(softmax_batch, dim=2)
            feature = zip(softmax_batch, scores_batch, prediction_batch)

        for feats, length in zip(feature, lengths):
            if self.use_crf:
                confidences, tag_seq, scores = self._viterbi_decode(
                    feats=feats[:length],
                    transitions=transitions,
                    all_scores=get_all_tags,
                )
            else:
                softmax, score, prediction = feats
                confidences = score[:length].tolist()
                tag_seq = prediction[:length].tolist()
                scores = softmax[:length].tolist()

            tags.append(
                [
                    Label(self.tag_dictionary.get_item_for_index(tag), conf)
                    for conf, tag in zip(confidences, tag_seq)
                ]
            )

            if get_all_tags:
                all_tags.append(
                    [
                        [
                            Label(
                                self.tag_dictionary.get_item_for_index(score_id), score
                            )
                            for score_id, score in enumerate(score_dist)
                        ]
                        for score_dist in scores
                    ]
                )

        return tags, all_tags

    @staticmethod
    def _softmax(x, axis):
        # reduce raw values to avoid NaN during exp
        x_norm = x - x.max(axis=axis, keepdims=True)
        y = np.exp(x_norm)
        return y / y.sum(axis=axis, keepdims=True)

    def _viterbi_decode(
        self, feats: np.ndarray, transitions: np.ndarray, all_scores: bool
    ):
        id_start = self.tag_dictionary.get_idx_for_item(START_TAG)
        id_stop = self.tag_dictionary.get_idx_for_item(STOP_TAG)

        backpointers = np.empty(shape=(feats.shape[0], self.tagset_size), dtype=np.int_)
        backscores = np.empty(
            shape=(feats.shape[0], self.tagset_size), dtype=np.float32
        )

        init_vvars = np.expand_dims(
            np.repeat(-10000.0, self.tagset_size), axis=0
        ).astype(np.float32)
        init_vvars[0][id_start] = 0

        forward_var = init_vvars
        for index, feat in enumerate(feats):
            # broadcasting will do the job of reshaping and is more efficient than calling repeat
            next_tag_var = forward_var + transitions
            bptrs_t = next_tag_var.argmax(axis=1)
            viterbivars_t = next_tag_var[np.arange(bptrs_t.shape[0]), bptrs_t]
            forward_var = viterbivars_t + feat
            backscores[index] = forward_var
            forward_var = forward_var[np.newaxis, :]
            backpointers[index] = bptrs_t

        terminal_var = forward_var.squeeze() + transitions[id_stop]
        terminal_var[id_stop] = -10000.0
        terminal_var[id_start] = -10000.0
        best_tag_id = terminal_var.argmax()

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        start = best_path.pop()
        assert start == id_start
        best_path.reverse()

        best_scores_softmax = self._softmax(backscores, axis=1)
        best_scores_np = np.max(best_scores_softmax, axis=1)

        # default value
        all_scores_np = np.zeros(0, dtype=np.float64)
        if all_scores:
            all_scores_np = best_scores_softmax
            for index, (tag_id, tag_scores) in enumerate(zip(best_path, all_scores_np)):
                if type(tag_id) != int and tag_id.item() != tag_scores.argmax():
                    swap_index_score = tag_scores.argmax()
                    (
                        all_scores_np[index][tag_id.item()],
                        all_scores_np[index][swap_index_score],
                    ) = (
                        all_scores_np[index][swap_index_score],
                        all_scores_np[index][tag_id.item()],
                    )
                elif type(tag_id) == int and tag_id != tag_scores.argmax():
                    swap_index_score = tag_scores.argmax()
                    (
                        all_scores_np[index][tag_id],
                        all_scores_np[index][swap_index_score],
                    ) = (
                        all_scores_np[index][swap_index_score],
                        all_scores_np[index][tag_id],
                    )

        return best_scores_np.tolist(), best_path, all_scores_np.tolist()

    @staticmethod
    def _log_sum_exp_batch(vecs):
        maxi = torch.max(vecs, 1)[0]
        maxi_bc = maxi[:, None].repeat(1, vecs.shape[1])
        recti_ = torch.log(torch.sum(torch.exp(vecs - maxi_bc), 1))
        return maxi + recti_

    def _forward_alg(self, feats, lens_):

        init_alphas = torch.FloatTensor(self.tagset_size).fill_(-10000.0)
        init_alphas[self.tag_dictionary.get_idx_for_item(START_TAG)] = 0.0

        forward_var = torch.zeros(
            feats.shape[0],
            feats.shape[1] + 1,
            feats.shape[2],
            dtype=torch.float,
            device=flair_device,
        )

        forward_var[:, 0, :] = init_alphas[None, :].repeat(feats.shape[0], 1)

        transitions = self.transitions.view(
            1, self.transitions.shape[0], self.transitions.shape[1]
        ).repeat(feats.shape[0], 1, 1)

        for i in range(feats.shape[1]):
            emit_score = feats[:, i, :]

            tag_var = (
                emit_score[:, :, None].repeat(1, 1, transitions.shape[2])
                + transitions
                + forward_var[:, i, :][:, :, None]
                .repeat(1, 1, transitions.shape[2])
                .transpose(2, 1)
            )

            max_tag_var, _ = torch.max(tag_var, dim=2)

            tag_var = tag_var - max_tag_var[:, :, None].repeat(
                1, 1, transitions.shape[2]
            )

            agg_ = torch.log(torch.sum(torch.exp(tag_var), dim=2))

            cloned = forward_var.clone()
            cloned[:, i + 1, :] = max_tag_var + agg_

            forward_var = cloned

        forward_var = forward_var[range(forward_var.shape[0]), lens_, :]

        terminal_var = forward_var + self.transitions[
            self.tag_dictionary.get_idx_for_item(STOP_TAG)
        ][None, :].repeat(forward_var.shape[0], 1)

        alpha = self._log_sum_exp_batch(terminal_var)

        return alpha

    @staticmethod
    def _filter_empty_sentences(sentences: List[Sentence]) -> List[Sentence]:
        filtered_sentences = [sentence for sentence in sentences if sentence.tokens]
        if len(sentences) != len(filtered_sentences):
            logger.warning(
                f"Ignore {len(sentences) - len(filtered_sentences)} sentence(s) with no tokens."
            )
        return filtered_sentences

    @staticmethod
    def _filter_empty_string(texts: List[str]) -> List[str]:
        filtered_texts = [text for text in texts if text]
        if len(texts) != len(filtered_texts):
            logger.warning(
                f"Ignore {len(texts) - len(filtered_texts)} string(s) with no tokens."
            )
        return filtered_texts

    @staticmethod
    def _fetch_model(model_name) -> str:

        hu_path: str = "https://nlp.informatik.hu-berlin.de/resources/models"

        model_map = {
            # English NER models
            "ner": "/".join([hu_path, "ner", "en-ner-conll03-v0.4.pt"]),
            "ner-pooled": "/".join(
                [hu_path, "ner-pooled", "en-ner-conll03-pooled-v0.5.pt"]
            ),
            "ner-fast": "/".join([hu_path, "ner-fast", "en-ner-fast-conll03-v0.4.pt"]),
            "ner-ontonotes": "/".join(
                [hu_path, "ner-ontonotes", "en-ner-ontonotes-v0.4.pt"]
            ),
            "ner-ontonotes-fast": "/".join(
                [hu_path, "ner-ontonotes-fast", "en-ner-ontonotes-fast-v0.4.pt"]
            ),
            # Multilingual NER models
            "ner-multi": "/".join([hu_path, "multi-ner", "quadner-large.pt"]),
            "multi-ner": "/".join([hu_path, "multi-ner", "quadner-large.pt"]),
            "ner-multi-fast": "/".join(
                [hu_path, "multi-ner-fast", "ner-multi-fast.pt"]
            ),
            # English POS models
            "upos": "/".join([hu_path, "upos", "en-pos-ontonotes-v0.4.pt"]),
            "upos-fast": "/".join(
                [hu_path, "upos-fast", "en-upos-ontonotes-fast-v0.4.pt"]
            ),
            "pos": "/".join([hu_path, "pos", "en-pos-ontonotes-v0.5.pt"]),
            "pos-fast": "/".join(
                [hu_path, "pos-fast", "en-pos-ontonotes-fast-v0.5.pt"]
            ),
            # Multilingual POS models
            "pos-multi": "/".join([hu_path, "multi-pos", "pos-multi-v0.1.pt"]),
            "multi-pos": "/".join([hu_path, "multi-pos", "pos-multi-v0.1.pt"]),
            "pos-multi-fast": "/".join(
                [hu_path, "multi-pos-fast", "pos-multi-fast.pt"]
            ),
            "multi-pos-fast": "/".join(
                [hu_path, "multi-pos-fast", "pos-multi-fast.pt"]
            ),
            # English SRL models
            "frame": "/".join([hu_path, "frame", "en-frame-ontonotes-v0.4.pt"]),
            "frame-fast": "/".join(
                [hu_path, "frame-fast", "en-frame-ontonotes-fast-v0.4.pt"]
            ),
            # English chunking models
            "chunk": "/".join([hu_path, "chunk", "en-chunk-conll2000-v0.4.pt"]),
            "chunk-fast": "/".join(
                [hu_path, "chunk-fast", "en-chunk-conll2000-fast-v0.4.pt"]
            ),
            # Danish models
            "da-pos": "/".join([hu_path, "da-pos", "da-pos-v0.1.pt"]),
            "da-ner": "/".join([hu_path, "NER-danish", "da-ner-v0.1.pt"]),
            # German models
            "de-pos": "/".join([hu_path, "de-pos", "de-pos-ud-hdt-v0.5.pt"]),
            "de-pos-tweets": "/".join(
                [hu_path, "de-pos-tweets", "de-pos-twitter-v0.1.pt"]
            ),
            "de-ner": "/".join([hu_path, "de-ner", "de-ner-conll03-v0.4.pt"]),
            "de-ner-germeval": "/".join(
                [hu_path, "de-ner-germeval", "de-ner-germeval-0.4.1.pt"]
            ),
            "de-ler": "/".join([hu_path, "de-ner-legal", "de-ner-legal.pt"]),
            "de-ner-legal": "/".join([hu_path, "de-ner-legal", "de-ner-legal.pt"]),
            # French models
            "fr-ner": "/".join([hu_path, "fr-ner", "fr-ner-wikiner-0.4.pt"]),
            # Dutch models
            "nl-ner": "/".join([hu_path, "nl-ner", "nl-ner-bert-conll02-v0.6.pt"]),
            "nl-ner-rnn": "/".join([hu_path, "nl-ner-rnn", "nl-ner-conll02-v0.5.pt"]),
            # Malayalam models
            "ml-pos": "https://raw.githubusercontent.com/qburst/models-repository/master/FlairMalayalamModels/malayalam-xpos-model.pt",
            "ml-upos": "https://raw.githubusercontent.com/qburst/models-repository/master/FlairMalayalamModels/malayalam-upos-model.pt",
            # Portuguese models
            "pt-pos-clinical": "/".join(
                [
                    hu_path,
                    "pt-pos-clinical",
                    "pucpr-flair-clinical-pos-tagging-best-model.pt",
                ]
            ),
            # Keyphase models
            "keyphrase": "/".join([hu_path, "keyphrase", "keyphrase-en-scibert.pt"]),
            "negation-speculation": "/".join(
                [hu_path, "negation-speculation", "negation-speculation-model.pt"]
            ),
            # Biomedical models
            "hunflair-paper-cellline": "/".join(
                [
                    hu_path,
                    "hunflair_smallish_models",
                    "cellline",
                    "hunflair-celline-v1.0.pt",
                ]
            ),
            "hunflair-paper-chemical": "/".join(
                [
                    hu_path,
                    "hunflair_smallish_models",
                    "chemical",
                    "hunflair-chemical-v1.0.pt",
                ]
            ),
            "hunflair-paper-disease": "/".join(
                [
                    hu_path,
                    "hunflair_smallish_models",
                    "disease",
                    "hunflair-disease-v1.0.pt",
                ]
            ),
            "hunflair-paper-gene": "/".join(
                [hu_path, "hunflair_smallish_models", "gene", "hunflair-gene-v1.0.pt"]
            ),
            "hunflair-paper-species": "/".join(
                [
                    hu_path,
                    "hunflair_smallish_models",
                    "species",
                    "hunflair-species-v1.0.pt",
                ]
            ),
            "hunflair-cellline": "/".join(
                [
                    hu_path,
                    "hunflair_smallish_models",
                    "cellline",
                    "hunflair-celline-v1.0.pt",
                ]
            ),
            "hunflair-chemical": "/".join(
                [
                    hu_path,
                    "hunflair_allcorpus_models",
                    "huner-chemical",
                    "hunflair-chemical-full-v1.0.pt",
                ]
            ),
            "hunflair-disease": "/".join(
                [
                    hu_path,
                    "hunflair_allcorpus_models",
                    "huner-disease",
                    "hunflair-disease-full-v1.0.pt",
                ]
            ),
            "hunflair-gene": "/".join(
                [
                    hu_path,
                    "hunflair_allcorpus_models",
                    "huner-gene",
                    "hunflair-gene-full-v1.0.pt",
                ]
            ),
            "hunflair-species": "/".join(
                [
                    hu_path,
                    "hunflair_allcorpus_models",
                    "huner-species",
                    "hunflair-species-full-v1.1.pt",
                ]
            ),
        }

        cache_dir = Path("models")
        if model_name in model_map:
            model_name = cached_path(model_map[model_name], cache_dir=cache_dir)

        # the historical German taggers by the @redewiegergabe project
        if model_name == "de-historic-indirect":
            model_file = (
                Path(flair_cache_root) / cache_dir / "indirect" / "final-model.pt"
            )
            if not model_file.exists():
                cached_path(
                    "http://www.redewiedergabe.de/models/indirect.zip",
                    cache_dir=cache_dir,
                )
                unzip_file(
                    Path(flair_cache_root) / cache_dir / "indirect.zip",
                    Path(flair_cache_root) / cache_dir,
                )
            model_name = str(
                Path(flair_cache_root) / cache_dir / "indirect" / "final-model.pt"
            )

        if model_name == "de-historic-direct":
            model_file = (
                Path(flair_cache_root) / cache_dir / "direct" / "final-model.pt"
            )
            if not model_file.exists():
                cached_path(
                    "http://www.redewiedergabe.de/models/direct.zip",
                    cache_dir=cache_dir,
                )
                unzip_file(
                    Path(flair_cache_root) / cache_dir / "direct.zip",
                    Path(flair_cache_root) / cache_dir,
                )
            model_name = str(
                Path(flair_cache_root) / cache_dir / "direct" / "final-model.pt"
            )

        if model_name == "de-historic-reported":
            model_file = (
                Path(flair_cache_root) / cache_dir / "reported" / "final-model.pt"
            )
            if not model_file.exists():
                cached_path(
                    "http://www.redewiedergabe.de/models/reported.zip",
                    cache_dir=cache_dir,
                )
                unzip_file(
                    Path(flair_cache_root) / cache_dir / "reported.zip",
                    Path(flair_cache_root) / cache_dir,
                )
            model_name = str(
                Path(flair_cache_root) / cache_dir / "reported" / "final-model.pt"
            )

        if model_name == "de-historic-free-indirect":
            model_file = (
                Path(flair_cache_root) / cache_dir / "freeIndirect" / "final-model.pt"
            )
            if not model_file.exists():
                cached_path(
                    "http://www.redewiedergabe.de/models/freeIndirect.zip",
                    cache_dir=cache_dir,
                )
                unzip_file(
                    Path(flair_cache_root) / cache_dir / "freeIndirect.zip",
                    Path(flair_cache_root) / cache_dir,
                )
            model_name = str(
                Path(flair_cache_root) / cache_dir / "freeIndirect" / "final-model.pt"
            )

        # Fallback to Hugging Face model hub
        if not Path(model_name).exists() and not model_name.startswith("http"):
            # e.g. stefan-it/flair-ner-conll03 is a valid namespace
            # and  stefan-it/flair-ner-conll03@main supports specifying a commit/branch name
            hf_model_name = "pytorch_model.bin"
            revision = "main"

            if "@" in model_name:
                model_name_splitted = model_name.split("@")
                revision = model_name_splitted[-1]
                model_name = model_name_splitted[0]

            # Lazy import
            # from transformers import file_utils

            url = file_utils.hf_bucket_url(
                model_id=model_name, revision=revision, filename=hf_model_name
            )
            model_name = file_utils.cached_path(
                url_or_filename=url, cache_dir=flair_cache_root
            )

        return model_name

    def get_transition_matrix(self):
        data = []
        for to_idx, row in enumerate(self.transitions):
            for from_idx, column in enumerate(row):
                row = [
                    self.tag_dictionary.get_item_for_index(from_idx),
                    self.tag_dictionary.get_item_for_index(to_idx),
                    column.item(),
                ]
                data.append(row)
            data.append(["----"])
        print(tabulate(data, headers=["FROM", "TO", "SCORE"]))

    def __str__(self):
        return (
            super(FlairModel, self).__str__().rstrip(")")
            + f"  (beta): {self.beta}\n"
            + f"  (weights): {self.weight_dict}\n"
            + f"  (weight_tensor) {self.loss_weights}\n)"
        )
