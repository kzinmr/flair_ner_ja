import pickle
import json
import jsonlines
from flair.models import SequenceTagger
from flair.data import Sentence  #, build_japanese_tokenizer
import torch
import MeCab
import textspan


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
    print(sentence)
    tagger.predict(sentence)
    spans = sentence.get_spans('ner')
    spans = [span_to_dict(span) for span in spans]
    print(spans)
    sentence_tokenized = sentence.to_plain_string()
    return {'text': text, 'tokenized_text': sentence_tokenized, 'spans': spans}

def tag_and_align_spans(tagger, text, spans_gold, labels_gold):
    """ 1. Do NER, 2. Align tokenized spans and original spans for pred and gold data
    """
    # Do NER
    response = tag_text(tagger, text)

    # tokenized_spans, tokenized_text, original_text -> original_spans
    tokenized_text = response['tokenized_text']
    labels_pred = [d['labels'] for d in response['spans']]
    tokenized_spans_pred = [(d['start_pos'], d['end_pos']) for d in response['spans']]
    spans_pred = textspan.align_spans(tokenized_spans_pred, tokenized_text, text)
    spans_pred = [components[0] for components in spans_pred if len(components) > 0]

    # original_spans, original_text, tokenized_text -> tokenized_spans
    labels_gold = [[{'value': label, 'confidence': 1.0}] for label in labels_gold]
    tokenized_spans_gold = textspan.align_spans(spans_gold, text, tokenized_text)
    tokenized_spans_gold = [(components[0][0], components[-1][1]) for components in tokenized_spans_gold if len(components) > 0]

    return {
        'text': text,
        'tokenized_text': tokenized_text,
        'string_spans': {
            'pred': [{'text': text[s:e], 'start_pos': s, 'end_pos': e, 'labels': ld} for (s,e), ld in zip(spans_pred, labels_pred)],
            'gold': [{'text': text[s:e], 'start_pos': s, 'end_pos': e, 'labels': ld} for (s,e), ld in zip(spans_gold, labels_gold)]
        },
        'token_spans': {
            'pred': response['spans'],
            'gold': [{'text': tokenized_text[s:e], 'start_pos': s, 'end_pos': e, 'labels': ld} for (s,e), ld in zip(tokenized_spans_gold, labels_gold)]
        }
    }


def jp_dumps(s):
    return json.dumps(s, ensure_ascii=False)

def load_model(modelpath, quantize=False):
    if modelpath.endswith('pt'):
        tagger = SequenceTagger.load(modelpath)
        if quantize:
            tagger = torch.quantization.quantize_dynamic(tagger, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8)
    elif modelpath.endswith('pkl'):
        with open(modelpath, 'rb') as fp:
            tagger = pickle.load(fp)
    else:
        tagger = None
    return tagger

if __name__=='__main__':
    modelpath = "./model/best-model.pkl"
    quantize = False
    tagger = load_model(modelpath, quantize)
    if tagger is not None:
        wakati = MeCab.Tagger("-Owakati")
        outputs = []
        with jsonlines.open('/app/data/test.jsonl') as reader:
            for text, entd in reader.iter():
                triples = sorted(set(map(tuple, entd['entities'])), key=lambda x: x[0])
                labels_gold = [label for _, _, label in triples]
                spans_gold = [(s, e) for s, e, _ in triples if s > 0 and e > 0]
                result = tag_and_align_spans(tagger, text, spans_gold, labels_gold)
                outputs.append(result)

        with jsonlines.open('/app/data/predict.jsonl', mode='w', dumps=jp_dumps) as writer:
            writer.write_all(outputs)
    else:
        print(f'Failed to load model: {modelpath}')