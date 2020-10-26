import pickle
import json
import jsonlines
from flair.models import SequenceTagger
from flair.data import Sentence  # , build_japanese_tokenizer
import MeCab
import textspan
from itertools import accumulate
from typing import List, Tuple, Dict, Optional

# import torch

wakati = MeCab.Tagger("-Owakati")


def span_to_dict(span: List[Tuple[int, int]]) -> Dict:
    d = span.to_dict()
    d["labels"] = [l.to_dict() for l in d["labels"]]
    return d


def tag_single_text(tagger: SequenceTagger, text: str) -> Dict:
    sentence = Sentence(wakati.parse(text))  # , use_tokenizer=tokenizer)
    tagger.predict(sentence)
    spans = sentence.get_spans("ner")
    spans = [span_to_dict(span) for span in spans]
    sentence_tokenized = sentence.to_plain_string()
    return {"text": text, "tokenized_text": sentence_tokenized, "spans": spans}


def tag_texts(
    tagger: SequenceTagger, texts: List[str], batch_size: int = 32
) -> List[Dict]:
    sentences = [Sentence(wakati.parse(text_i)) for text_i in texts]
    tagger.predict(sentences, mini_batch_size=batch_size)

    results = []
    for text_i, sentence in zip(texts, sentences):
        spans = [span_to_dict(span) for span in sentence.get_spans("ner")]
        sentence_tokenized = sentence.to_plain_string()
        results.append(
            {"text": text_i, "tokenized_text": sentence_tokenized, "spans": spans}
        )
    return results


def align_spans_from_single_response(
    tagger: SequenceTagger,
    text: str,
    spans_gold: List[Tuple[int, int]],
    labels_gold: List[str],
) -> Dict:
    response = tag_single_text(tagger, text)
    # tokenized_spans, tokenized_text, original_text -> original_spans
    text = response["text"]
    tokenized_text = response["tokenized_text"]
    labels_pred = [d["labels"] for d in response["spans"]]
    tokenized_spans_pred = [(d["start_pos"], d["end_pos"]) for d in response["spans"]]
    spans_pred = textspan.align_spans(tokenized_spans_pred, tokenized_text, text)
    spans_pred = [components[0] for components in spans_pred if len(components) > 0]

    # original_spans, original_text, tokenized_text -> tokenized_spans
    labels_gold = [[{"value": label, "confidence": 1.0}] for label in labels_gold]
    tokenized_spans_gold = textspan.align_spans(spans_gold, text, tokenized_text)
    tokenized_spans_gold = [
        (components[0][0], components[-1][1])
        for components in tokenized_spans_gold
        if len(components) > 0
    ]

    return {
        "text": text,
        "tokenized_text": tokenized_text,
        "string_spans": {
            "pred": [
                {"text": text[s:e], "start_pos": s, "end_pos": e, "labels": ld}
                for (s, e), ld in zip(spans_pred, labels_pred)
            ],
            "gold": [
                {"text": text[s:e], "start_pos": s, "end_pos": e, "labels": ld}
                for (s, e), ld in zip(spans_gold, labels_gold)
            ],
        },
        "token_spans": {
            "pred": response["spans"],
            "gold": [
                {
                    "text": tokenized_text[s:e],
                    "start_pos": s,
                    "end_pos": e,
                    "labels": ld,
                }
                for (s, e), ld in zip(tokenized_spans_gold, labels_gold)
            ],
        },
    }


def tag_and_align_spans(
    tagger: SequenceTagger,
    text: str,
    spans_gold: List[Tuple[int, int]],
    labels_gold: List[str],
    rolling_step: int = 150,
    batch_size: int = 32,
) -> Dict:
    """1. Do NER, 2. Align tokenized spans and original spans for pred and gold data"""
    # Do NER
    if rolling_step <= 0 or len(text) <= rolling_step:
        return align_spans_from_single_response(tagger, text, spans_gold, labels_gold)
    else:
        texts = [text[i : i + rolling_step] for i in range(0, len(text), rolling_step)]
        responses = tag_texts(tagger, texts, batch_size)
        # 分割テキスト上スパンから元テキスト上のスパンを復元
        spans_pred = []
        labels_pred = []
        tokenized_texts = []
        # 元スパンをテキスト分割後のスパン位置に分割
        # TODO: いくつかは境界に位置して落ちてしまうのでそれらを残す
        text_offsets = [0] + list(accumulate([len(text_i) for text_i in texts]))[:-1]
        spans_labels_golds_split = [
            [
                (span[0] - text_offset, span[1] - text_offset, label)
                for span, label in zip(spans_gold, labels_gold)
                if span[0] - text_offset >= 0 and span[1] - text_offset < rolling_step
            ]
            for text_offset in text_offsets
        ]
        tokenized_spans_pred = []
        tokenized_spans_gold = []
        labels_gold = []
        for response, text_offset, spans_labels_gold_i in zip(
            responses, text_offsets, spans_labels_golds_split
        ):
            # tokenized_spans, tokenized_text, original_text -> original_spans
            text_i = response["text"]
            tokenized_text_i = response["tokenized_text"]
            labels_pred_i = [d["labels"] for d in response["spans"]]
            tokenized_spans_pred_i = [
                (d["start_pos"], d["end_pos"]) for d in response["spans"]
            ]
            spans_pred_i = textspan.align_spans(
                tokenized_spans_pred_i, tokenized_text_i, text_i
            )
            spans_pred_i = [
                components[0] for components in spans_pred_i if len(components) > 0
            ]
            # split -> org
            spans_pred_org_i = [
                (s + text_offset, e + text_offset) for (s, e) in spans_pred_i
            ]
            spans_pred.extend(spans_pred_org_i)
            labels_pred.extend(labels_pred_i)
            tokenized_spans_pred.append(tokenized_spans_pred_i)
            tokenized_texts.append(tokenized_text_i)

            # original_spans, original_text, tokenized_text -> tokenized_spans
            spans_gold_i = [(s, e) for s, e, _ in spans_labels_gold_i]
            labels_gold_i = [
                [{"value": label, "confidence": 1.0}]
                for _, _, label in spans_labels_gold_i
            ]
            tokenized_spans_gold_i = textspan.align_spans(
                spans_gold_i, text_i, tokenized_text_i
            )
            tokenized_spans_gold_i = [
                (components[0][0], components[-1][1])
                for components in tokenized_spans_gold_i
                if len(components) > 0
            ]
            tokenized_spans_gold.extend(tokenized_spans_gold_i)
            labels_gold.extend(labels_gold_i)

        # NOTE: 各splitにおいて、 delimiter 1文字ぶんだけ余分に加算されることを加味したenumerate
        tokenized_offsets = [0] + list(
            accumulate(
                [len(tokenized_i) + i for i, tokenized_i in enumerate(tokenized_texts)]
            )
        )[:-1]
        tokenized_spans_pred = [
            (s + tokenized_offset, e + tokenized_offset)
            for spans, tokenized_offset in zip(tokenized_spans_pred, tokenized_offsets)
            for s, e in spans
        ]
        tokenized_text = " ".join(tokenized_texts)
        print(tokenized_offsets)
        print(tokenized_text)
        print(tokenized_spans_pred)
        print(labels_pred)
        return {
            "text": text,
            "tokenized_text": tokenized_text,
            "string_spans": {
                "pred": [
                    {"text": text[s:e], "start_pos": s, "end_pos": e, "labels": ld}
                    for (s, e), ld in zip(spans_pred, labels_pred)
                ],
                "gold": [
                    {"text": text[s:e], "start_pos": s, "end_pos": e, "labels": ld}
                    for (s, e), ld in zip(spans_gold, labels_gold)
                ],
            },
            "token_spans": {
                "pred": [
                    {
                        "text": tokenized_text[s:e],
                        "start_pos": s,
                        "end_pos": e,
                        "labels": ld,
                    }
                    for (s, e), ld in zip(tokenized_spans_pred, labels_pred)
                ],
                "gold": [
                    {
                        "text": tokenized_text[s:e],
                        "start_pos": s,
                        "end_pos": e,
                        "labels": ld,
                    }
                    for (s, e), ld in zip(tokenized_spans_gold, labels_gold)
                ],
            },
        }


def jp_dumps(s: str) -> Dict:
    return json.dumps(s, ensure_ascii=False)


def load_model(modelpath: str) -> Optional[SequenceTagger]:
    if modelpath.endswith("pt"):
        tagger = SequenceTagger.load(modelpath)
        # if quantize:
        #     tagger = torch.quantization.quantize_dynamic(tagger, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8)
    elif modelpath.endswith("pkl"):
        with open(modelpath, "rb") as fp:
            tagger = pickle.load(fp)
    else:
        tagger = None
    return tagger


if __name__ == "__main__":
    modelpath = "./model/best-model.pkl"
    # quantize = False
    batch_size = 32
    rolling_step = 150
    tagger = load_model(modelpath)
    if tagger is not None:
        outputs = []
        with jsonlines.open("/app/data/test.jsonl") as reader:
            for text, entd in reader.iter():
                triples = sorted(set(map(tuple, entd["entities"])), key=lambda x: x[0])
                labels_gold = [label for _, _, label in triples]
                spans_gold = [(s, e) for s, e, _ in triples if s > 0 and e > 0]
                result = tag_and_align_spans(
                    tagger, text, spans_gold, labels_gold, rolling_step, batch_size
                )
                outputs.append(result)

        with jsonlines.open(
            "/app/data/predict.jsonl", mode="w", dumps=jp_dumps
        ) as writer:
            writer.write_all(outputs)
    else:
        print(f"Failed to load model: {modelpath}")
