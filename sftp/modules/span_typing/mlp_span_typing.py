from typing import *

import torch
from torch.nn import CrossEntropyLoss, KLDivLoss, LogSoftmax

from .span_typing import SpanTyping


@SpanTyping.register('mlp')
class MLPSpanTyping(SpanTyping):
    """
    An MLP implementation for Span Typing.
    """
    def __init__(
            self,
            input_dim: int,
            hidden_dims: List[int],
            label_emb: torch.nn.Embedding,
            n_category: int,
            label_to_ignore: Optional[List[int]] = None
    ):
        """
        :param input_dim: dim(parent_span) + dim(child_span) + dim(label_dim)
        :param hidden_dims: The dim of hidden layers of MLP.
        :param n_category: #labels
        :param label_emb: Embeds labels to vectors.
        """
        super().__init__(label_emb.num_embeddings, label_to_ignore, )
        self.MLPs: List[torch.nn.Linear] = list()
        for i_mlp, output_dim in enumerate(hidden_dims + [n_category]):
            mlp = torch.nn.Linear(input_dim, output_dim, bias=True)
            self.MLPs.append(mlp)
            self.add_module(f'MLP-{i_mlp}', mlp)
            input_dim = output_dim

        # Embeds labels as features.
        self.label_emb = label_emb

    def forward(
            self,
            span_vec: torch.Tensor,
            parent_at_span: torch.Tensor,
            span_labels: Optional[torch.Tensor],
            prediction_only: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Inputs: All features for typing a child span.
        Process: Update the metric.
        Output: The loss of typing and predictions.
        :return:
            loss: Loss for label prediction.
            prediction: Predicted labels.
        """
        is_soft = span_labels.dtype != torch.int64
        # Shape [batch, span, label_dim]
        label_vec = span_labels @ self.label_emb.weight if is_soft else self.label_emb(span_labels)
        n_batch, n_span, _ = label_vec.shape
        n_label, _ = self.ontology.shape
        # Shape [batch, span, label_dim]
        parent_label_features = label_vec.gather(1, parent_at_span.unsqueeze(2).expand_as(label_vec))
        # Shape [batch, span, token_dim]
        parent_span_features = span_vec.gather(1, parent_at_span.unsqueeze(2).expand_as(span_vec))
        # Shape [batch, span, token_dim]
        child_span_features = span_vec

        features = torch.cat([parent_label_features, parent_span_features, child_span_features], dim=2)
        # Shape [batch, span, label]
        for mlp in self.MLPs[:-1]:
            features = torch.relu(mlp(features))
        logits = self.MLPs[-1](features)

        logits_for_prediction = logits.clone()

        if not is_soft:
            # Shape [batch, span]
            parent_labels = span_labels.gather(1, parent_at_span)
            onto_mask = self.ontology.unsqueeze(0).expand(n_batch, -1, -1).gather(
                1, parent_labels.unsqueeze(2).expand(-1, -1, n_label)
            )
            logits_for_prediction[~onto_mask] = float('-inf')

        label_dist = torch.softmax(logits_for_prediction, 2)
        label_confidence, predictions = label_dist.max(2)
        ret = {'prediction': predictions, 'label_confidence': label_confidence, 'distribution': label_dist}
        if prediction_only:
            return ret

        span_labels = span_labels.clone()

        if is_soft:
            self.acc_metric(logits_for_prediction, span_labels.max(2)[1], ~span_labels.sum(2).isclose(torch.tensor(0.)))
            ret['loss'] = KLDivLoss(reduction='sum')(LogSoftmax(dim=2)(logits), span_labels)
        else:
            for label_idx in self.label_to_ignore:
                span_labels[span_labels == label_idx] = -100
            self.acc_metric(logits_for_prediction, span_labels, span_labels != -100)
            ret['loss'] = CrossEntropyLoss(reduction='sum')(logits.flatten(0, 1), span_labels.flatten())

        return ret
