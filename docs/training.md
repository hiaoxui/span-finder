# Training Span Finder


## Training script

Span-finder is built upon AllenNLP framework. To get the training started, set up a few environment variables and run

```bash
export DATA_PATH=path/to/framenet/folder
export ONTOLOGY_PATH=path/to/ontology
export ENCODER=xlm-roberta-large
export CUDA=-1
allennlp train -s path/to/checkpoint/folder --include-package sftp config/fn.jsonnet
```

Please refer to the [data doc](doc/data.md) for data explanations.

Note that the `CUDA` can be -1, which means CPU; can be CUDA number (0, 1, 2, 3...) or a list of CUDA numbers.
If a list is passed, multi-device training will be triggered.
But note that AllenNLP can be unstable with multi-device training.

## Metrics explanation

By default, the following metrics will be used

- em: (includes emp, emr, emf) Exact matching metric. A span is exactly matched iff its parent, boundaries, and label are all correctly predicted. Note that if a parent is not correctly predicted, all its children will be treated as false negative. In another word, errors are propagated.
- sm: (includes smp, smr, smf) Span matching metric. Similar to EM but will not check the labels. If you observe high EM but low SM, then the typing system is not properly working.
- finder: (includes finder-p, finder-r, finder-f) A metric to measure how well the model can find spans. Different from SM, in this metric, gold parent will be provided, so the errors will not be propagated.
- typing_acc: Span typing accuracy with gold parent and gold span boundaries.


Optional metrics that might be useful for SRL-style tasks. Put the following line

`metrics: [{type: "srl", check_type: true}],` 

to the span model in the config file to turn on this feature. You will see the following two metrics:

- trigger: (include trigger-p, trigger-r, trigger-f) It measures how well the system can find the event triggers (or frames in FrameNet). If `check_type` is True, it also checks the event label.
- role: (include role-p, role-r, role-f) It measures how well the system can find roles. Note if the event/trigger is not found, all its children will be treated as false negative. If `check_type` is True, it also checks the role label.

## Typing loss factor

The loss comes from two sources: SpanFinding and SpanTyping modules.
SpanFinder uses CRF and use probability as loss, but SpanTyping uses cross entropy.
They're of different scale so we have to re-scale them.
The formula is:

`loss = finding_loss + typing_loss_factor * typing_loss`

Empirically Guanghui finds the optimal `typing_loss_factor` for FrameNet system is 750.

In theory, we should put the two losses to the same space. Guanghui is looking into this, and this might be solved in SpanFinder 0.0.2.

## Optimizer

A custom optimizer `transformer` is used for span finder. 
It allows you to specify special learning rate for transformer encoder and fix the parameters of certain modules.
Empirically, fix embedding (so only fine-tune the encoder and pooler) and train with lr=1e-5 yields best results for FrameNet.
For usage and more details, see its class doc.
