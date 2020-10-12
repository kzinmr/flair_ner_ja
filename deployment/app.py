from fastapi import FastAPI
from pydantic import BaseModel
import json
from flair.models import SequenceTagger
from flair.data import Sentence  #, build_japanese_tokenizer

import MeCab

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


# Define the prediction function.
def tag_text(tagger: SequenceTagger, sentence: str):

    """
    A small function to tag the incoming string.
    ------------------------
    Params:
    tagger: The loaded model object.
    sentence: A string to tag.
    ------------------------
    Output:
    A list of tuples containing labels & probabilities.
    """
    sentence = Sentence(' '.join(wakati.parse(sentence).split()))  #, use_tokenizer=tokenizer)
    tagger.predict(sentence)
    spans = sentence.get_spans('ner')
    spans = [json.dumps(span_to_dict(span), ensure_ascii=False) for span in spans]
    return spans


@app.post("/ner")
async def tag_text_endpoint(case: Case):
    """Takes the text request and returns a record with the labels & associated probabilities."""

    # Use the pretrained model to classify the incoming text in the request.
    tagged_text = tag_text(tagger, case.text)

    return tagged_text
