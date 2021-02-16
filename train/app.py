import os
from pathlib import Path
from typing import Optional

import hydra
from flair.datasets import ColumnCorpus

from flair.trainers import ModelTrainer
from omegaconf import DictConfig
from flair.embeddings import FlairEmbeddings, StackedEmbeddings, WordEmbeddings
from flair.models import SequenceTagger
# from ja_elmo import ELMoEmbeddings


def make_bio_conll_corpus(
    data_folder: Path,
    train_file="train.bio.retokenize",
    dev_file="dev.bio.retokenize",
    test_file="test.bio.retokenize",
) -> Optional[ColumnCorpus]:
    """(flairで使うtokenizerでretokenize済みの)conll2003形式データセットのロード"""
    train_path = data_folder / train_file
    dev_path = data_folder / dev_file
    test_path = data_folder / test_file
    if train_path.exists() and dev_path.exists() and test_path.exists():
        columns = {0: "text", 1: "ner"}
        corpus = ColumnCorpus(
            data_folder,
            columns,
            train_file=train_file,
            dev_file=dev_file,
            test_file=test_file,
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
        # WordEmbeddings('ja'), # 695MB
        # WordEmbeddings('ja-crawl'), # 1.2GB
        FlairEmbeddings("ja-forward"), FlairEmbeddings("ja-backward"),  # 335MB * 2
    ]
    embeddings = StackedEmbeddings(embeddings=embedding_types)
    # embeddings = ELMoEmbeddings('ja')

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
    corpus = make_bio_conll_corpus(data_folder)

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
        monitor_train=cfg.training.debug,
        monitor_test=cfg.training.debug,
        num_workers=cfg.training.workers,
        embeddings_storage_mode=cfg.training.embeddings_storage_mode,
    )

    # tagger = SequenceTagger.load(model_dir / "best-model.pt")
    # qtagger = torch.quantization.quantize_dynamic(
    #     tagger, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8
    # )
    # with open(model_dir / "quantized-best-model.pkl", "wb") as fp:
    #     pickle.dump(qtagger, fp)


if __name__ == "__main__":
    main()
