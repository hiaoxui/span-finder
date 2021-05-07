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

## Paper

Please cite this paper for reference:

```bibtex
@inproceedings{xia-etal-2021-lome,
    title = "{LOME}: Large Ontology Multilingual Extraction",
    author = "Xia, Patrick  and
      Qin, Guanghui  and
      Vashishtha, Siddharth  and
      Chen, Yunmo  and
      Chen, Tongfei  and
      May, Chandler  and
      Harman, Craig  and
      Rawlins, Kyle  and
      White, Aaron Steven  and
      Van Durme, Benjamin",
    booktitle = "Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations",
    year = "2021",
    url = "https://www.aclweb.org/anthology/2021.eacl-demos.19",
    pages = "149--159",
}
```
