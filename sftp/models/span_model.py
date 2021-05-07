import os
from typing import *

import torch
from allennlp.common.from_params import Params, T, pop_and_construct_arg
from allennlp.data.vocabulary import Vocabulary, DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import Seq2SeqEncoder
from allennlp.modules.span_extractors import SpanExtractor
from allennlp.training.metrics import Metric

from ..metrics import ExactMatch
from ..modules import SpanFinder, SpanTyping
from ..utils import num2mask, VIRTUAL_ROOT, Span, tensor2span


@Model.register("span")
class SpanModel(Model):
    """
    Identify/Find spans; link them as a tree; label them.
    """
    default_predictor = 'span'

    def __init__(
            self,
            vocab: Vocabulary,

            # Modules
            word_embedding: TextFieldEmbedder,
            span_extractor: SpanExtractor,
            span_finder: SpanFinder,
            span_typing: SpanTyping,

            # Config
            typing_loss_factor: float = 1.,
            max_recursion_depth: int = -1,
            max_decoding_spans: int = -1,
            debug: bool = False,

            # Ontology Constraints
            ontology_path: Optional[str] = None,

            # Metrics
            metrics: Optional[List[Metric]] = None,
    ) -> None:
        """
        Note for jsonnet file: it doesn't strictly follow the init examples of every module for that we override
        the from_params method.
        You can either check the SpanModel.from_params or the example jsonnet file.
        :param vocab: No need to specify.
        ## Modules
        :param word_embedding: Refer to the module doc.
        :param span_extractor: Refer to the module doc.
        :param span_finder: Refer to the module doc.
        :param span_typing: Refer to the module doc.
        ## Configs
        :param typing_loss_factor: loss = span_finder_loss + span_typing_loss * typing_loss_factor
        :param max_recursion_depth: Maximum tree depth for inference. E.g., 1 for shallow event typing, 2 for SRL,
            -1 (unlimited) for dependency parsing.
        :param max_decoding_spans: Maximum spans for inference. -1 for unlimited.
        :param debug: Useless now.
        """
        self._pad_idx = vocab.get_token_index(DEFAULT_PADDING_TOKEN, 'token')
        self._null_idx = vocab.get_token_index(DEFAULT_OOV_TOKEN, 'span_label')
        super().__init__(vocab)

        self.word_embedding = word_embedding
        self._span_finder = span_finder
        self._span_extractor = span_extractor
        self._span_typing = span_typing

        self.metrics = [ExactMatch(True), ExactMatch(False)]
        if metrics is not None:
            self.metrics.extend(metrics)

        if ontology_path is not None and os.path.exists(ontology_path):
            self._span_typing.load_ontology(ontology_path, self.vocab)

        self._max_decoding_spans = max_decoding_spans
        self._typing_loss_factor = typing_loss_factor
        self._max_recursion_depth = max_recursion_depth
        self.debug = debug

    def forward(
            self,
            tokens: Dict[str, Dict[str, torch.Tensor]],

            span_boundary: Optional[torch.Tensor] = None,
            span_labels: Optional[torch.Tensor] = None,
            parent_indices: Optional[torch.Tensor] = None,
            parent_mask: Optional[torch.Tensor] = None,

            bio_seqs: Optional[torch.Tensor] = None,
            raw_inputs: Optional[dict] = None,
            meta: Optional[dict] = None,

            **extra
    ) -> Dict[str, torch.Tensor]:
        """
        For training, provide all blow.
        For inference, it's enough to only provide words.

        :param tokens: Indexed input sentence. Shape: [batch, token]

        :param span_boundary: Start and end indices for every span. Note this includes both parent and
            non-parent spans. Shape: [batch, span, 2]. For the last dim, [0] is start idx and [1] is end idx.
        :param span_labels: Indexed label for spans, including parent and non-parent ones. Shape: [batch, span]
        :param parent_indices: The parent span idx of every span. Shape: [batch, span]
        :param parent_mask: True if this span is a parent. Shape: [batch, span]

        :param bio_seqs: Shape [batch, parent, token, 3]
        :param raw_inputs

        :param meta: Meta information. Will be copied to the outputs.

        :return:
            - loss: training loss
            - prediction: Predicted spans
            - meta: Meta info copied from input
            - inputs: Input sentences and spans (if exist)
        """
        ret = {'inputs': raw_inputs, 'meta': meta or dict()}

        is_eval = span_labels is not None and not self.training  # evaluation on dev set
        is_test = span_labels is None  # test on test set
        # Shape [batch]
        num_spans = (span_labels != -1).sum(1) if span_labels is not None else None
        num_words = tokens['pieces']['mask'].sum(1)
        # Shape [batch, word, token_dim]
        token_vec = self.word_embedding(tokens)

        if span_labels is not None:
            # Revise the padding value from -1 to 0
            span_labels[span_labels == -1] = 0

        # Calculate Loss
        if self.training or is_eval:
            # Shape [batch, word, token_dim]
            span_vec = self._span_extractor(token_vec, span_boundary)
            finder_rst = self._span_finder(
                token_vec, num2mask(num_words), span_vec, num2mask(num_spans), span_labels, parent_indices,
                parent_mask, bio_seqs
            )
            typing_rst = self._span_typing(span_vec, parent_indices, span_labels)
            ret['loss'] = finder_rst['loss'] + typing_rst['loss'] * self._typing_loss_factor

        # Decoding
        if is_eval or is_test:
            pred_span_boundary, pred_span_labels, pred_parent_indices, pred_cursor, pred_label_confidence \
                = self.inference(num_words, token_vec, **extra)
            prediction = self.post_process_pred(
                pred_span_boundary, pred_span_labels, pred_parent_indices, pred_cursor, pred_label_confidence
            )
            for pred, raw_in in zip(prediction, raw_inputs):
                pred.re_index(raw_in['offsets'], True, True, True)
                pred.remove_overlapping()
            ret['prediction'] = prediction
            if 'spans' in raw_inputs[0]:
                for pred, raw_in in zip(prediction, raw_inputs):
                    gold = raw_in['spans']
                    for metric in self.metrics:
                        metric(pred, gold)

        return ret

    def inference(
            self,
            num_words: torch.Tensor,
            token_vec: torch.Tensor,
            **auxiliaries
    ):
        n_batch = num_words.shape[0]
        # The decoding results are preserved in the following tensors starting with `pred`
        # During inference, we completely ignore the arguments defaulted None in the forward method.
        # The span indexing space is shift to the decoding span space. (since we do not have gold span now)
        # boundary indices of every predicted span
        pred_span_boundary = num_words.new_zeros([n_batch, self._max_decoding_spans, 2])
        # labels (and corresponding confidence) for predicted spans
        pred_span_labels = num_words.new_full(
            [n_batch, self._max_decoding_spans], self.vocab.get_token_index(VIRTUAL_ROOT, 'span_label')
        )
        pred_label_confidence = num_words.new_zeros([n_batch, self._max_decoding_spans])
        # label masked as True will be treated as parent in the next round
        pred_parent_mask = num_words.new_zeros([n_batch, self._max_decoding_spans], dtype=torch.bool)
        pred_parent_mask[:, 0] = True
        # parent index (in the span indexing space) for every span
        pred_parent_indices = num_words.new_zeros([n_batch, self._max_decoding_spans])
        # what index have we reached for every batch?
        pred_cursor = num_words.new_ones([n_batch])

        # Pass environment variables to handler. Extra variables will be ignored.
        # So pass the union of variables that are needed by different modules.
        span_find_handler = self._span_finder.inference_forward_handler(
            token_vec, num2mask(num_words), self._span_extractor, **auxiliaries
        )

        # Every step here is one layer of the tree. It deals with all the parents for the last layer
        # so there might be 0 to multiple parents for a batch for a single step.
        for _ in range(self._max_recursion_depth):
            cursor_before_find = pred_cursor.clone()
            span_find_handler(
                pred_span_boundary, pred_span_labels, pred_parent_mask, pred_parent_indices, pred_cursor
            )
            # Labels of old spans are re-predicted. It doesn't matter since their results shouldn't change
            # in theory.
            span_typing_ret = self._span_typing(
                self._span_extractor(token_vec, pred_span_boundary), pred_parent_indices, pred_span_labels, True
            )
            pred_span_labels = span_typing_ret['prediction']
            pred_label_confidence = span_typing_ret['label_confidence']
            pred_span_labels[:, 0] = self.vocab.get_token_index(VIRTUAL_ROOT, 'span_label')
            pred_parent_mask = (
                    num2mask(cursor_before_find, self._max_decoding_spans) ^ num2mask(pred_cursor,
                                                                                      self._max_decoding_spans)
            )

            # Break the inference loop if 1) all batches reach max span limit OR 2) no parent is predicted
            # at last step OR 3) max recursion limit is reached (for loop condition)
            if (pred_cursor == self._max_decoding_spans).all() or pred_parent_mask.sum() == 0:
                break

        return pred_span_boundary, pred_span_labels, pred_parent_indices, pred_cursor, pred_label_confidence

    def one_step_prediction(
            self,
            tokens: Dict[str, Dict[str, torch.Tensor]],
            parent_boundary: torch.Tensor,
            parent_labels: torch.Tensor,
    ):
        """
        Single step prediction. Given parent span boundary indices, return the corresponding children spans
            and their labels.
        Restriction: Each sentence contain exactly 1 parent.
        For efficient multi-layer prediction, i.e. given a root, predict the whole tree,
            refer to the `forward' method.
        :param tokens: See forward.
        :param parent_boundary: Pairs of (start_idx, end_idx) for parents. Shape [batch, 2]
        :param parent_labels: Labels for parents. Shape [batch]
            Note: If `no_label' is on in span_finder module, this will be ignored.
        :return:
            children_boundary: (start_idx, end_idx) for every child span. Padded with (0, 0).
                Shape [batch, children, 2]
            children_labels: Label for every child span. Padded with null_idx. Shape [batch, children]
            num_children: The number of children predicted for parent/batch. Shape [batch]
                Tips: You can use num2mask method to convert this to bool tensor mask.
        """
        num_words = tokens['pieces']['mask'].sum(1)
        # Shape [batch, word, token_dim]
        token_vec = self.word_embedding(tokens)
        n_batch = token_vec.shape[0]

        # The following variables assumes the parent is the 0-th span, and we let the model
        # to extend the span list.
        pred_span_boundary = num_words.new_zeros([n_batch, self._max_decoding_spans, 2])
        pred_span_boundary[:, 0] = parent_boundary
        pred_span_labels = num_words.new_full([n_batch, self._max_decoding_spans], self._null_idx)
        pred_span_labels[:, 0] = parent_labels
        pred_parent_mask = num_words.new_zeros(pred_span_labels.shape, dtype=torch.bool)
        pred_parent_mask[:, 0] = True
        pred_parent_indices = num_words.new_zeros([n_batch, self._max_decoding_spans])
        # We start from idx 1 since 0 is the parents.
        pred_cursor = num_words.new_ones([n_batch])

        span_find_handler = self._span_finder.inference_forward_handler(
            token_vec, num2mask(num_words), self._span_extractor
        )
        span_find_handler(
            pred_span_boundary, pred_span_labels, pred_parent_mask, pred_parent_indices, pred_cursor
        )
        typing_out = self._span_typing(
            self._span_extractor(token_vec, pred_span_boundary), pred_parent_indices, pred_span_labels, True
        )
        pred_span_labels = typing_out['prediction']

        # Now remove the parent
        num_children = pred_cursor - 1
        max_children = int(num_children.max())
        children_boundary = pred_span_boundary[:, 1:max_children + 1]
        children_labels = pred_span_labels[:, 1:max_children + 1]
        children_distribution = typing_out['distribution'][:, 1:max_children + 1]
        return children_boundary, children_labels, num_children, children_distribution

    def post_process_pred(
            self, span_boundary, span_labels, parent_indices, num_spans, label_confidence
    ) -> List[Span]:
        pred_spans = tensor2span(
            span_boundary, span_labels, parent_indices, num_spans, label_confidence,
            self.vocab.get_index_to_token_vocabulary('span_label'),
            label_ignore=[self._null_idx],
        )
        return pred_spans

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        ret = dict()
        if reset:
            for metric in self.metrics:
                ret.update(metric.get_metric(reset))
        ret.update(self._span_finder.get_metrics(reset))
        ret.update(self._span_typing.get_metric(reset))
        return ret

    @classmethod
    def from_params(
            cls: Type[T],
            params: Params,
            constructor_to_call: Callable[..., T] = None,
            constructor_to_inspect: Callable[..., T] = None,
            **extras,
    ) -> T:
        """
        Specify the dependency between modules. E.g. the input dim of a module might depend on the output dim
        of another module.
        """
        vocab = extras['vocab']
        word_embedding = pop_and_construct_arg('SpanModel', 'word_embedding', TextFieldEmbedder, None, params, **extras)
        label_dim, token_emb_dim = params.pop('label_dim'), word_embedding.get_output_dim()
        span_extractor = pop_and_construct_arg(
            'SpanModel', 'span_extractor', SpanExtractor, None, params, input_dim=token_emb_dim, **extras
        )
        label_embedding = torch.nn.Embedding(vocab.get_vocab_size('span_label'), label_dim)
        extras['label_emb'] = label_embedding

        if params.get('span_finder').get('type') == 'bio':
            bio_encoder = Seq2SeqEncoder.from_params(
                params['span_finder'].pop('bio_encoder'),
                input_size=span_extractor.get_output_dim() + token_emb_dim + label_dim,
                input_dim=span_extractor.get_output_dim() + token_emb_dim + label_dim,
                **extras
            )
            extras['span_finder'] = SpanFinder.from_params(
                params.pop('span_finder'), bio_encoder=bio_encoder, **extras
            )
        else:
            extras['span_finder'] = pop_and_construct_arg(
                'SpanModel', 'span_finder', SpanFinder, None, params, **extras
            )
            extras['span_finder'].label_emb = label_embedding

        if params.get('span_typing').get('type') == 'mlp':
            extras['span_typing'] = SpanTyping.from_params(
                params.pop('span_typing'),
                input_dim=span_extractor.get_output_dim() * 2 + label_dim,
                n_category=vocab.get_vocab_size('span_label'),
                label_to_ignore=[
                    vocab.get_token_index(lti, 'span_label')
                    for lti in [DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN]
                ],
                **extras
            )
        else:
            extras['span_typing'] = pop_and_construct_arg(
                'SpanModel', 'span_typing', SpanTyping, None, params, **extras
            )
            extras['span_typing'].label_emb = label_embedding

        return super().from_params(
            params,
            word_embedding=word_embedding,
            span_extractor=span_extractor,
            **extras
        )
