from typing import *

import torch
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.span_extractors import SpanExtractor
from allennlp.training.metrics import FBetaMeasure

from ..smooth_crf import SmoothCRF
from .span_finder import SpanFinder
from ...utils import num2mask, mask2idx, BIO


@SpanFinder.register("bio")
class BIOSpanFinder(SpanFinder):
    """
    Train BIO representations for span finding.
    """
    def __init__(
            self,
            bio_encoder: Seq2SeqEncoder,
            label_emb: torch.nn.Embedding,
            no_label: bool = True,
    ):
        super().__init__(no_label)
        self.bio_encoder = bio_encoder
        self.label_emb = label_emb

        self.classifier = torch.nn.Linear(bio_encoder.get_output_dim(), 3)
        self.crf = SmoothCRF(3)

        self.fb_measure = FBetaMeasure(1., 'micro', [BIO.index('B'), BIO.index('I')])

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
        See doc of SpanFinder.
        Possible extra variables:
            smoothing_factor
        :return:
            - loss
            - prediction
        """
        ret = dict()
        is_soft = span_labels.dtype != torch.int64

        distinct_parent_indices, num_parents = mask2idx(parent_mask)
        n_batch, n_parent = distinct_parent_indices.shape
        n_token = token_vec.shape[1]
        # Shape [batch, parent, token_dim]
        parent_span_features = span_vec.gather(
            1, distinct_parent_indices.unsqueeze(2).expand(-1, -1, span_vec.shape[2])
        )
        label_features = span_labels @ self.label_emb.weight if is_soft else self.label_emb(span_labels)
        if self._no_label:
            label_features = label_features.zero_()
        # Shape [batch, span, label_dim]
        parent_label_features = label_features.gather(
            1, distinct_parent_indices.unsqueeze(2).expand(-1, -1, label_features.shape[2])
        )
        # Shape [batch, parent, token, token_dim*2]
        encoder_inputs = torch.cat([
            parent_span_features.unsqueeze(2).expand(-1, -1, n_token, -1),
            token_vec.unsqueeze(1).expand(-1, n_parent, -1, -1),
            parent_label_features.unsqueeze(2).expand(-1, -1, n_token, -1),
        ], dim=3)
        encoder_inputs = encoder_inputs.reshape(n_batch * n_parent, n_token, -1)

        # Shape [batch, parent]. Considers batches may have fewer seqs.
        seq_mask = num2mask(num_parents)
        # Shape [batch, parent, token]. Also considers batches may have fewer tokens.
        token_mask = seq_mask.unsqueeze(2).expand(-1, -1, n_token) & token_mask.unsqueeze(1).expand(-1, n_parent, -1)

        class_in = self.bio_encoder(encoder_inputs, token_mask.flatten(0, 1))
        class_out = self.classifier(class_in).reshape(n_batch, n_parent, n_token, 3)

        if not prediction:
            # For training
            # We use `seq_mask` here because seq with length 0 is not acceptable.
            ret['loss'] = -self.crf(class_out[seq_mask], bio_seqs[seq_mask], token_mask[seq_mask])
            self.fb_measure(class_out[seq_mask], bio_seqs[seq_mask].max(2).indices, token_mask[seq_mask])
        else:
            # For prediction
            features_for_decode = class_out.clone().detach()
            decoded = self.crf.viterbi_tags(features_for_decode.flatten(0, 1), token_mask.flatten(0, 1))
            pred_tag = torch.tensor(
                [path + [BIO.index('O')] * (n_token - len(path)) for path, _ in decoded]
            )
            pred_tag = pred_tag.reshape(n_batch, n_parent, n_token)
            ret['prediction'] = pred_tag

        return ret

    @staticmethod
    def bio2boundary(seqs) -> Tuple[torch.Tensor, torch.Tensor]:
        def recursive_construct_spans(seqs_):
            """
            Helper function for bio2boundary
            Recursively convert seqs of integers to boundary indices.
            Return boundary indices and corresponding lens
            """
            if isinstance(seqs_, torch.Tensor):
                if seqs_.device.type == 'cuda':
                    seqs_ = seqs_.to(device='cpu')
                seqs_ = seqs_.tolist()
            if isinstance(seqs_[0], int):
                seqs_ = [BIO[i] for i in seqs_]
                span_boundary_list = bio_tags_to_spans(seqs_)
                return torch.tensor([item[1] for item in span_boundary_list]), len(span_boundary_list)
            span_boundary = list()
            lens_ = list()
            for seq in seqs_:
                one_bou, one_len = recursive_construct_spans(seq)
                span_boundary.append(one_bou)
                lens_.append(one_len)
            if isinstance(lens_[0], int):
                lens_ = torch.tensor(lens_)
            else:
                lens_ = torch.stack(lens_)
            return span_boundary, lens_

        boundary_list, lens = recursive_construct_spans(seqs)
        max_span = int(lens.max())
        boundary = torch.zeros((*lens.shape, max_span, 2), dtype=torch.long)

        def recursive_copy(list_var, tensor_var):
            if len(list_var) == 0:
                return
            if isinstance(list_var, torch.Tensor):
                tensor_var[:len(list_var)] = list_var
                return
            assert len(list_var) == len(tensor_var)
            for list_var_, tensor_var_ in zip(list_var, tensor_var):
                recursive_copy(list_var_, tensor_var_)

        recursive_copy(boundary_list, boundary)

        return boundary, lens

    def inference_forward_handler(
            self,
            token_vec: torch.Tensor,
            token_mask: torch.Tensor,
            span_extractor: SpanExtractor,
            **auxiliaries,
    ) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None]:
        """
        Refer to the doc of the SpanFinder for definition of this function.
        """

        def handler(
                span_boundary: torch.Tensor,
                span_labels: torch.Tensor,
                parent_mask: torch.Tensor,
                parent_indices: torch.Tensor,
                cursor: torch.tensor,
        ):
            """
            Refer to the doc of the SpanFinder for definition of this function.
            """
            max_decoding_span = span_boundary.shape[1]
            # Shape [batch, span, token_dim]
            span_vec = span_extractor(token_vec, span_boundary)
            # Shape [batch, parent]
            parent_indices_at_span, _ = mask2idx(parent_mask)
            pred_bio = self(
                token_vec, token_mask, span_vec, None, span_labels, None, parent_mask, prediction=True
            )['prediction']
            # Shape [batch, parent, span, 2]; Shape [batch, parent]
            pred_boundary, pred_num = self.bio2boundary(pred_bio)
            if pred_boundary.device != span_boundary.device:
                pred_boundary = pred_boundary.to(device=span_boundary.device)
                pred_num = pred_num.to(device=span_boundary.device)
            # Shape [batch, parent, span]
            pred_mask = num2mask(pred_num)

            # Parent Loop
            for pred_boundary_parent, pred_mask_parent, parent_indices_parent \
                    in zip(pred_boundary.unbind(1), pred_mask.unbind(1), parent_indices_at_span.unbind(1)):
                for pred_boundary_step, step_mask in zip(pred_boundary_parent.unbind(1), pred_mask_parent.unbind(1)):
                    step_mask &= cursor < max_decoding_span
                    parent_indices[step_mask] = parent_indices[step_mask].scatter(
                        1,
                        cursor[step_mask].unsqueeze(1),
                        parent_indices_parent[step_mask].unsqueeze(1)
                    )
                    span_boundary[step_mask] = span_boundary[step_mask].scatter(
                        1,
                        cursor[step_mask].reshape(-1, 1, 1).expand(-1, -1, 2),
                        pred_boundary_step[step_mask].unsqueeze(1)
                    )
                    cursor[step_mask] += 1

        return handler

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        score = self.fb_measure.get_metric(reset)
        if reset:
            return {
                'finder_p': score['precision'] * 100,
                'finder_r': score['recall'] * 100,
                'finder_f': score['fscore'] * 100,
            }
        else:
            return {'finder_f': score['fscore'] * 100}
