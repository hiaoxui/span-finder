from typing import *

import torch

from .span import Span


def _tensor2span_batch(
        span_boundary: torch.Tensor,
        span_labels: torch.Tensor,
        parent_indices: torch.Tensor,
        num_spans: torch.Tensor,
        label_confidence: torch.Tensor,
        idx2label: Dict[int, str],
        label_ignore: List[int],
) -> Span:
    spans = list()
    for (start_idx, end_idx), parent_idx, label, label_conf in \
            list(zip(span_boundary, parent_indices, span_labels, label_confidence))[:int(num_spans)]:
        if label not in label_ignore:
            span = Span(int(start_idx), int(end_idx), idx2label[int(label)], True, confidence=float(label_conf))
            if int(parent_idx) < len(spans):
                spans[int(parent_idx)].add_child(span)
            spans.append(span)
    return spans[0]


def tensor2span(
        span_boundary: torch.Tensor,
        span_labels: torch.Tensor,
        parent_indices: torch.Tensor,
        num_spans: torch.Tensor,
        label_confidence: torch.Tensor,
        idx2label: Dict[int, str],
        label_ignore: Optional[List[int]] = None,
) -> List[Span]:
    """
    Generate spans in dict from vectors. Refer to the model part for the meaning of these variables.
    If idx_ignore is provided, some labels will be ignored.
    :return:
    """
    label_ignore = label_ignore or []
    if span_boundary.device.type != 'cpu':
        span_boundary = span_boundary.to(device='cpu')
        parent_indices = parent_indices.to(device='cpu')
        span_labels = span_labels.to(device='cpu')
        num_spans = num_spans.to(device='cpu')
        label_confidence = label_confidence.to(device='cpu')

    ret = list()
    for args in zip(
            span_boundary.unbind(0), span_labels.unbind(0), parent_indices.unbind(0), num_spans.unbind(0),
            label_confidence.unbind(0),
    ):
        ret.append(_tensor2span_batch(*args, label_ignore=label_ignore, idx2label=idx2label))

    return ret
