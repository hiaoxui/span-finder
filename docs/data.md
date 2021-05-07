# Data Format

You can pass SpanFinder any formats of data, as long as you implement a dataset reader inherited from SpanReader. We also provide a Concrete dataset reader. Besides them, SpanFinder comes with its own JSON data format, which enables richer features for training and modeling.

The minimal example of the JSON is

```JSON
{
  "meta": {
    "fully_annotated": true
  },
  "tokens": ["Bob", "attacks", "the", "building", "."],
  "annotations": [
    {
      "span": [1, 1],
      "label": "Attack",
      "children": [
        {
          "span": [0, 0],
          "label": "Assailant",
          "children": []
        },
        {
          "span": [2, 3],
          "label": "Victim",
          "children": []
        }
      ]
    },
    {
      "span": [3, 3],
      "label": "Buildings",
      "children": [
        {
          "span": [3, 3],
          "label": "Building",
          "children": []
        }
      ]
    }
  ]
}
```

You can have nested spans with unlimited depth.

## Meta-info for Semantic Role Labeling (SRL)

```JSON
{
  "ontology": {
    "event": ["Violence-Attack"],
    "argument": ["Agent", "Patient"],
    "link": [[0, 0], [0, 1]]
  },
  "ontology_mapping": {
    "event": {
      "Attack": ["Violence-Attack", 0.8]
    },
    "argument": {
      "Assault": ["Agent", 0.95],
      "Victim": ["patient", 0.9]
    }
  }
}
```

TODO: Guanghui needs to doc this.
