## Mapping

If a file is passed to the predictor,
the predicted spans will be converted into a new ontology.
The file format should be

`<original parent label>\t<original label>\t<new label>`

If the predicted span is labeled as `<original label>`,
and its parent is labeled as `<orignal parent label>`,
it will be re-labeled as `<new label>`.
If no rules match, the span and all of its descendents will be ignored.

The `<original parent label>` is optional.
If the parent label is `@@VIRTUAL_ROOT@@`, then this rule matches the first layer of spans.
In semantic parsing, it matches events.
If the parent label is `*`, it means it can match anything.
