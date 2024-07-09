# span-finder
Parse sentences by finding &amp; labeling spans

## Installation

Environment:
- python 3.8
- python-pip

Suppose you are using Anaconda, you can create such an environment

```shell
conda create -n spanfinder python=3.8
conda activate spanfinder
```

If you have CUDA devices, run the following line
```shell
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

To install the dependencies, execute

``` shell
pip3 install -r requirements.txt
```


Optionally, you may install the package via
``` shell
python3 setup.py install
```
and import span-finder with `import sftp`.

## Demo

A demo (combined with Patrick's coref model) is [here](https://nlp.jhu.edu/demos/lome).

## Inference Only

If you use SpanFinder only for inference, we provide a checkpoint that was trained on FrameNet v1.7.
[This example](scripts/predict_span.py) shows the basic API of SpanFinder.
Note that the script will incur a checkpoint download everytime, so the best way is to
download [the checkpoint](https://gqin.top/sftp-fn) to local (~1.7GiB), or better extract it out, 
and then point the `-m` argument to the archived file or extracted folder.


## Training

For training, you may need to read [an overall document](docs/overall.md),
[the doc for data](docs/data.md), and [the doc for training](docs/training.md).

## Paper

Welcome to cite our work if you found it useful:

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
