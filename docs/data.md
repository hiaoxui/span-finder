# Data Format

You can pass SpanFinder any formats of data, as long as you implement a dataset reader inherited from SpanReader.

By default,
SpanFinder uses its own JSON data format, which enables richer features for training and modeling.

The minimal example of the JSON is

```JSON
{
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

An example data file for FrameNet can be downloaded [here](https://gqin.top/fn-data).
