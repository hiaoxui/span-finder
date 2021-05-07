from abc import ABC
from typing import *

import torch
from allennlp.common import Registrable
from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN, Vocabulary
from allennlp.training.metrics import CategoricalAccuracy


class SpanTyping(Registrable, torch.nn.Module, ABC):
    """
    Models the probability p(child_label | child_span, parent_span, parent_label).
    """
    def __init__(
            self,
            n_label: int,
            label_to_ignore: Optional[List[int]] = None,
    ):
        """
        :param label_to_ignore: Label indexes in this list will be ignored.
            Usually this should include NULL, PADDING and UNKNOWN.
        """
        super().__init__()
        self.label_to_ignore = label_to_ignore or list()
        self.acc_metric = CategoricalAccuracy()
        self.onto = torch.ones([n_label, n_label], dtype=torch.bool)
        self.register_buffer('ontology', self.onto)

    def load_ontology(self, path: str, vocab: Vocabulary):
        unk_id = vocab.get_token_index(DEFAULT_OOV_TOKEN, 'span_label')
        for line in open(path).readlines():
            entities = [vocab.get_token_index(ent, 'span_label') for ent in line.replace('\n', '').split('\t')]
            parent, children = entities[0], entities[1:]
            if parent == unk_id:
                continue
            self.onto[parent, :] = False
            children = list(filter(lambda x: x != unk_id, children))
            self.onto[parent, children] = True
        self.register_buffer('ontology', self.onto)

    def forward(
            self,
            span_vec: torch.Tensor,
            parent_at_span: torch.Tensor,
            span_labels: Optional[torch.Tensor],
            prediction_only: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Inputs: All features for typing a child span.
        Output: The loss of typing and predictions.
        :param span_vec: Shape [batch, span, token_dim]
        :param parent_at_span: Shape [batch, span]
        :param span_labels: Shape [batch, span]
        :param prediction_only: If True, no loss returned & metric will not be updated
        :return:
            loss: Loss for label prediction. (absent of pred_only = True)
            prediction: Predicted labels.
        """
        raise NotImplementedError

    def get_metric(self, reset):
        return{
            "typing_acc": self.acc_metric.get_metric(reset) * 100
        }
