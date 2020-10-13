from fastapi import FastAPI
from pydantic import BaseModel
import json
from flair.models import SequenceTagger
from flair.data import Sentence  #, build_japanese_tokenizer

import MeCab
textspan

# import sys
# sys.setrecursionlimit(100)

# Set up the schema & input validation for the inputs.
class Case(BaseModel):
    text: str

# tokenizer = build_japanese_tokenizer(tokenizer="MeCab")
wakati = MeCab.Tagger("-Owakati")
tagger = SequenceTagger.load("./model/best-model.pt")
app = FastAPI()


def span_to_dict(span):
    d = span.to_dict()
    d['labels'] = [l.to_dict() for l in d['labels']]
    return d


def tag_text(tagger: SequenceTagger, sentence: str):

    """
    A small function to tag the incoming string.
    ------------------------
    Params:
    tagger: The loaded model object.
    sentence: A string to tag.
    ------------------------
    Output:
    A list of dicts containing text, position spans and labels with confidences.
    ["{\"text\": \"株式会社XYZ\", \"start_pos\": 0, \"end_pos\": 7, \"labels\": [{\"value\": \"COMPANY\", \"confidence\": 0.99}]}"]
    """
    text = sentence
    sentence = Sentence(wakati.parse(sentence))  #, use_tokenizer=tokenizer)
    tagger.predict(sentence)
    tokenized_spans_response = [span_to_dict(span) for span in sentence.get_spans('ner')]

    # get string_spans from tokenized_spans
    tokenized_text = sentence.to_plain_string()
    labels_pred = [d['labels'] for d in tokenized_spans_response]
    tokenized_spans = [(d['start_pos'], d['end_pos']) for d in tokenized_spans_response]
    string_spans = textspan.align_spans(tokenized_spans, tokenized_text, text)
    string_spans = [components[0] for components in string_spans]
    string_spans_response = [{'text': text[s:e], 'start_pos': s, 'end_pos': e, 'labels': ld} for (s,e), ld in zip(string_spans, labels_pred)]
    return {
        'text': text,
        'tokenized_text': sentence_tokenized,
        'string_spans': string_spans_response,
        'tokenized_spans': tokenized_spans_response,
    }


@app.post("/ner")
async def tag_text_endpoint(case: Case):
    """Takes the text request and returns a record with the span & labels with confidences."""
    return json.dumps(tag_text(tagger, case.text), ensure_ascii=False)
