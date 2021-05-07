from abc import ABC, abstractmethod
from typing import *

import torch
from allennlp.common import Registrable
from allennlp.modules.span_extractors import SpanExtractor


class SpanFinder(Registrable, ABC, torch.nn.Module):
    """
    Model the probability p(child_span | parent_span [, parent_label])
    It's optional to model parent_label, since in some cases we may want the parameters to be shared across
    different tasks, where we may have similar span semantics but different label space.
    """
    def __init__(
            self,
            no_label: bool = True,
    ):
        """
        :param no_label: If True, will not use input labels as features and use all 0 vector instead.
        """
        super().__init__()
        self._no_label = no_label

    @abstractmethod
    def forward(
            self,
            token_vec: torch.Tensor,
            token_mask: torch.Tensor,
            span_vec: torch.Tensor,
            span_mask: Optional[torch.Tensor] = None,  # Do not need to provide
            span_labels: Optional[torch.Tensor] = None,  # Do not need to provide
            parent_indices: Optional[torch.Tensor] = None,  # Do not need to provide
            parent_mask: Optional[torch.Tensor] = None,
            bio_seqs: Optional[torch.Tensor] = None,
            prediction: bool = False,
            **extra
    ) -> Dict[str, torch.Tensor]:
        """
        Return training loss and predictions.
        :param token_vec: Vector representation of tokens. Shape [batch, token ,token_dim]
        :param token_mask: True for non-padding tokens.
        :param span_vec: Vector representation of spans. Shape [batch, span, token_dim]
        :param span_mask: True for non-padding spans. Shape [batch, span]
        :param span_labels: The labels of spans. Shape [batch, span]
        :param parent_indices: Parent indices of spans. Shape [batch, span]
        :param parent_mask: True for parent spans. Shape [batch, span]
        :param prediction: If True, no loss will be return & no metrics will be updated.
        :param bio_seqs: BIO sequences. Shape [batch, parent, token, 3]
        :return:
            loss: Training loss
            prediction: Shape [batch, span]. True for positive predictions.
        """
        raise NotImplementedError

    @abstractmethod
    def inference_forward_handler(
            self,
            token_vec: torch.Tensor,
            token_mask: torch.Tensor,
            span_extractor: SpanExtractor,
            **auxiliaries,
    ) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None]:
        """
        Pre-process some information and return a callable module for p(child_span | parent_span [,parent_label])
        :param token_vec: Vector representation of tokens. Shape [batch, token ,token_dim]
        :param token_mask: True for non-padding tokens.
        :param span_extractor: The same module in model.
        :param auxiliaries: Environment variables. You can pass extra environment variables
            since the extras will be ignored.
        :return:
            A callable function in a closure.
            The arguments for the callable object are:
                - span_boundary: Shape [batch, span, 2]
                - span_labels: Shape [batch, span]
                - parent_mask: Shape [batch, span]
                - parent_indices: Shape [batch, span]
                - cursor: Shape [batch]
            No return values. Everything should be done inplace.
            Note the span indexing space has different meaning from training process. We don't have gold span list,
            so span here refers to the predicted spans.
        """
        raise NotImplementedError

    @abstractmethod
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        raise NotImplementedError
