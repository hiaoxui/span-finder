# Training Span Finder

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

## Ontology Constraint

In some cases, certain spans can also be attached to specific spans.
E.g., in SRL tasks, event can only be attached to the VirtualRoot, and arguments can only be attached to the events.
The constraints of FrameNet is harsher, where each frame have some specific frame elements.

These constraints can be abstracted as a boolean square matrix whose columns and rows are span labels including VIRTUAL_ROOT. 
Say it's `M`, label2 can be label1's child iff `M[label1, label2]` if True.

You can specify ontology constraint for SpanFinder with the `ontology_path` argument in the SpanModel class.
The format of this file is simple. Each line is one row of the `M` matrix:

```parent_label child_label_1 child_label_2```

which means child1 and child2 can be attached to the parent. 
Both `parent_label` and `child_label` are strings, and the space between them should be `\t` not ` `.
If a parent_label is missing from the file, by default all children be attachable.
If this file is not provided, all labels can be attached to all labels.

An example of this file can be found at CLSP grid:

```/home/gqin2/data/framenet/ontology.tsv```

## Typing loss factor

(This section might be updated soon -- Guanghui)

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
