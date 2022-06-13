# Overall

## Parse tree structure

Span-finder treats the parse as a tree. 
Every node in the tree corresponds to a span in the text, and is associated with a type.

For the shallow semantic parsing problem (e.g. FrameNet), the tree starts with a special node of type
`@@VIRTUAL_ROOT@@`, and corresponds to a dummy span in the text (the begin-of-sentence token).
The children of virtual root form the second layer of the tree, and their types are *frames* (or "events" in SRL),
whose corresponding spans are *lexical units*.
Each frame may also have some children, and they are frame elements, whose corresponding spans are their trigger spans.

Not that although span-finder can work with arbitrarily numbers of layers theoretically,
we restrict it to predict at most 3 layers for FrameNet parsing.
For other tasks that may involve fewer or more layers, you may change that setting in the configuration file.


## Ontology

Ontology defines what types can be the children of what types. 
For example, a frame can be a child of virtual root, while a frame element can only be the child of its corresponding
frames. To inform the model this, an ontology file should be specified.

Each line in the ontology defines a rule for a type. The first line should be like
```
@@VIRTUAL_ROOT@@<TAB>frame1<TAB>frame2<TAB>frame3...
```
and it defines what types can be used as "frames".
The following lines should be like
```
frame1<TAB>fe1<TAB>fe2<TAB>fe3...
frame2<TAB>fe2<TAB>fe4<TAB>fe5...
```
Do not forget to replace `<TAB>` with `\t`.
You may find this file in [the provided dataset](https://gqin.top/fn-data).

The ontology file is only needed for training. After training, the ontology will be stored in the checkpoint
and will be automatically used during inference.
