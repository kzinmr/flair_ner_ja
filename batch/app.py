import json
import jsonlines
from flair.models import SequenceTagger
from flair.data import Sentence  #, build_japanese_tokenizer

import MeCab


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
    sentence = Sentence(' '.join(wakati.parse(sentence).split()))  #, use_tokenizer=tokenizer)
    print(sentence)
    tagger.predict(sentence)
    spans = sentence.get_spans('ner')
    spans = [span_to_dict(span) for span in spans]
    print(spans)
    return spans


def jp_dumps(s):
    return json.dumps(s, ensure_ascii=False)

if __name__=='__main__':
    wakati = MeCab.Tagger("-Owakati")
    modelpath = "./model/best-model.pt"
    try:
        tagger = SequenceTagger.load(modelpath)
        outputs = []
        with jsonlines.open('/app/data/test.jsonl') as reader:
            for sentence, entd in reader.iter():
                spans = tag_text(tagger, sentence)
                outputs.append({'text': sentence, 'spans': spans, 'entities': entd['entities']})

        with jsonlines.open('/app/data/predict.jsonl', mode='w', dumps=jp_dumps) as writer:
            writer.write_all(outputs)
    except FileNotFoundError:
        print(f'No file of {modelpath}')