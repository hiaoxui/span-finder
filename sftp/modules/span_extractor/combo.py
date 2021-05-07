from typing import *

import torch
from allennlp.modules.span_extractors import SpanExtractor


@SpanExtractor.register('combo')
class ComboSpanExtractor(SpanExtractor):
    def __init__(self, input_dim: int, sub_extractors: List[SpanExtractor]):
        super().__init__()
        self.sub_extractors = sub_extractors
        for i, sub in enumerate(sub_extractors):
            self.add_module(f'SpanExtractor-{i+1}', sub)
        self.input_dim = input_dim

    def get_input_dim(self) -> int:
        return self.input_dim

    def get_output_dim(self) -> int:
        return sum([sub.get_output_dim() for sub in self.sub_extractors])

    def forward(
            self,
            sequence_tensor: torch.FloatTensor,
            span_indices: torch.LongTensor,
            sequence_mask: torch.BoolTensor = None,
            span_indices_mask: torch.BoolTensor = None,
    ):
        outputs = [
            sub(
                sequence_tensor=sequence_tensor,
                span_indices=span_indices,
                span_indices_mask=span_indices_mask
            ) for sub in self.sub_extractors
        ]
        return torch.cat(outputs, dim=2)
