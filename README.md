# span-finder
Parse sentences by finding &amp; labeling spans

## Installation

Environment:
- python >= 3.7
- pip

To install the dependencies, execute

``` shell script
pip install -r requirements.txt
pip uninstall -y dataclasses
```

Then install SFTP (Span Finding - Transductive Parsing) package:

``` shell script
python setup.py install
```

## Prediction

If you use SpanFinder only for inference, please read [this example](scripts/predict_span.py).

## Demo

A demo (combined with Patrick's coref model) is [here](https://nlp.jhu.edu/demos/lome).

## Pre-Trained Models

A model pretrained for framenet parsing can be found [here](https://gqin.top/sftp-fn).
