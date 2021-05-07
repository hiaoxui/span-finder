from argparse import ArgumentParser
from typing import *

from flask import Flask
from flask import request

from sftp import SpanPredictor, Span

parser = ArgumentParser()
parser.add_argument('model', metavar='MODEL_PATH', type=str)
parser.add_argument('-p', metavar='PORT', type=int, default=7749)
parser.add_argument('-d', metavar='DEVICE', type=int, default=-1)
args = parser.parse_args()

template = open('tools/demo/flask_template.html').read()
predictor = SpanPredictor.from_path(args.model, cuda_device=args.d)
app = Flask(__name__)
default_sentence = '因为 आरजू です vegan , هي купил soja .'


def visualized_prediction(inputs: List[str], prediction: Span, prefix=''):
    spans = list()
    span2event = [[] for _ in inputs]
    for event_idx, event in enumerate(prediction):
        for arg_idx, arg in enumerate(event):
            for token_idx in range(arg.start_idx, arg.end_idx+1):
                span2event[token_idx].append((event_idx, arg_idx))

    for token_idx, token in enumerate(inputs):
        class_labels = ' '.join(
            ['token'] + [f'{prefix}-arg-{event_idx}-{arg_idx}' for event_idx, arg_idx in span2event[token_idx]]
        )
        spans.append(f'<span id="{prefix}-token-{token_idx}" class="{class_labels}" style="background-color">{token} </span>\n')

    for event_idx, event in enumerate(prediction):
        spans[event.start_idx] = (
            f'<span class="highlight bottom blue" '
            f' onmouseenter="highlight_args({event_idx}, \'{prefix}\')" onmouseleave="cancel_highlight(\'{prefix}\')">'
            '<span class="highlight__content" align="center">'
            f'<span class="event" id="{prefix}-event-{event_idx}">'
            + spans[event.start_idx]
        )
        spans[event.end_idx] += f'</span></span><span class="highlight__label"><center>{event.label}</center></span>'
        arg_tips = []
        for arg_idx, arg in enumerate(event):
            arg_tips.append(f'<span class="{prefix}-arg-{event_idx}-{arg_idx}">{arg.label}</span>')
        if len(arg_tips) > 0:
            arg_tips = '<br>'.join(arg_tips)
            spans[event.end_idx] += f'<span class="highlight__tooltip">{arg_tips}</span>\n'
        spans[event.end_idx] += '\n</span>'
    return(
            '<div class="passage model__content__summary highlight-container highlight-container--bottom-labels">\n' +
            '\n'.join(spans) + '\n</div>'
    )


def structured_prediction(inputs, prediction):
    ret = list()
    for event in prediction:
        event_text, event_label = ' '.join(inputs[event.start_idx: event.end_idx+1]), event.label
        ret.append(f'<li class="list-group-item list-group-item-info">'
                   f'<strong>{event_label}</strong>: {event_text}</li>')
        for arg in event:
            arg_text = ' '.join(inputs[arg.start_idx: arg.end_idx+1])
            ret.append(
                f'<li class="list-group-item">&nbsp;&nbsp;&nbsp;&nbsp;<strong>{arg.label}</strong>: {arg_text}</li>'
            )
    content = '\n'.join(ret)
    return '\n<ul class="list-group">\n' + content + '\n</ul>'


@app.route('/')
def sftp():
    ret = template
    tokens = request.args.get('sentence')
    if tokens is not None:
        ret = ret.replace('DEFAULT_SENTENCE', tokens)
        sentences = tokens.split('\n')
        model_outputs = predictor.predict_batch_sentences(sentences, max_tokens=512)
        vis_pred, str_pred = list(), list()
        for sent_idx, output in enumerate(model_outputs):
            vis_pred.append(visualized_prediction(output.sentence, output.span, f'sent{sent_idx}'))
            str_pred.append(structured_prediction(output.sentence, output.span))
        ret = ret.replace('VISUALIZED_PREDICTION', '<hr>'.join(vis_pred))
        ret = ret.replace('STRUCTURED_PREDICTION', '<hr>'.join(str_pred))
    else:
        ret = ret.replace('DEFAULT_SENTENCE', default_sentence)
        ret = ret.replace('VISUALIZED_PREDICTION', '')
        ret = ret.replace('STRUCTURED_PREDICTION', '')
    return ret


app.run(port=args.p)
