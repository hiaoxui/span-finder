import logging
from abc import ABC
from typing import *

import numpy as np
from allennlp.common.util import END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans
from allennlp.data.fields import *
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer, Token

from ..utils import Span, BIOSmoothing, apply_bio_smoothing

logger = logging.getLogger(__name__)


@DatasetReader.register('span')
class SpanReader(DatasetReader, ABC):
    def __init__(
            self,
            pretrained_model: str,
            max_length: int = 512,
            ignore_label: bool = False,
            debug: bool = False,
            **extras
    ) -> None:
        """
        :param pretrained_model: The name of the pretrained model. E.g. xlm-roberta-large
        :param max_length: Sequences longer than this limit will be truncated.
        :param ignore_label: If True, label on spans will be anonymized.
        :param debug: True to turn on debugging mode.
        :param span_proposals: Needed for "enumeration" scheme, but not needed for "BIO".
            If True, it will try to enumerate candidate spans in the sentence, which will then be fed into
            a binary classifier (EnumSpanFinder).
            Note: It might take time to propose spans. And better to use SpacyTokenizer if you want to call
            constituency parser or dependency parser.
        :param maximum_negative_spans: Necessary for EnumSpanFinder.
        :param extras: Args to DatasetReader.
        """
        super().__init__(**extras)
        self.word_indexer = {
            'pieces': PretrainedTransformerIndexer(pretrained_model, namespace='pieces')
        }

        self._pretrained_model_name = pretrained_model
        self.debug = debug
        self.ignore_label = ignore_label

        self._pretrained_tokenizer = PretrainedTransformerTokenizer(pretrained_model)
        self.max_length = max_length
        self.n_span_removed = 0

    def retokenize(
            self, sentence: List[str], truncate: bool = True
    ) -> Tuple[List[str], List[Optional[Tuple[int, int]]]]:
        pieces, offsets = self._pretrained_tokenizer.intra_word_tokenize(sentence)
        pieces = list(map(str, pieces))
        if truncate:
            pieces = pieces[:self.max_length]
            pieces[-1] = END_SYMBOL
        return pieces, offsets

    def prepare_inputs(
            self,
            sentence: List[str],
            spans: Optional[Union[List[Span], Span]] = None,
            truncate: bool = True,
            label_type: str = 'string',
    ) -> Dict[str, Field]:
        """
        Prepare inputs and auxiliary variables for span model.
        :param sentence: A list of tokens. Do not pass in any special tokens, like BOS or EOS.
            Necessary for both training and testing.
        :param spans: Optional. For training, spans passed in will be considered as positive examples; the spans
            that are automatically proposed and not in the positive set will be considered as negative examples.
            Necessary for training.
        :param truncate: If True, sequence will be truncated if it's longer than `self.max_training_length`
        :param label_type: One of [string, list].

        :return: Dict of AllenNLP fields. For detailed of explanation of every field, refer to the comments
            below. For the shape of every field, check the module doc.
                Fields list:
                    - words
                    - span_labels
                    - span_boundary
                    - parent_indices
                    - parent_mask
                    - bio_seqs
                    - raw_sentence
                    - raw_spans
                    - proposed_spans
        """
        fields = dict()

        pieces, offsets = self.retokenize(sentence, truncate)
        fields['tokens'] = TextField(list(map(Token, pieces)), self.word_indexer)
        raw_inputs = {'sentence': sentence, "pieces": pieces, 'offsets': offsets}
        fields['raw_inputs'] = MetadataField(raw_inputs)

        if spans is None:
            return fields

        vr = spans if isinstance(spans, Span) else Span.virtual_root(spans)
        self.n_span_removed = vr.remove_overlapping()
        raw_inputs['spans'] = vr

        vr = vr.re_index(offsets)
        if truncate:
            vr.truncate(self.max_length)
        if self.ignore_label:
            vr.ignore_labels()

        # (start_idx, end_idx) pairs. Left and right inclusive.
        # The first span is the Virtual Root node. Shape [span, 2]
        span_boundary = list()
        # label on span. Shape [span]
        span_labels = list()
        # parent idx (span indexing space). Shape [span]
        span_parent_indices = list()
        # True for parents. Shape [span]
        parent_mask = [False] * vr.n_nodes
        # Key: parent idx (span indexing space). Value: child span idx
        flatten_spans = list(vr.bfs())
        for span_idx, span in enumerate(vr.bfs()):
            if span.is_parent:
                parent_mask[span_idx] = True
            # 0 is the virtual root
            parent_idx = flatten_spans.index(span.parent) if span.parent else 0
            span_parent_indices.append(parent_idx)
            span_boundary.append(span.boundary)
            span_labels.append(span.label)

        bio_tag_list: List[List[str]] = list()
        bio_configs: List[List[BIOSmoothing]] = list()
        # Shape: [#parent, #token, 3]
        bio_seqs: List[np.ndarray] = list()
        # Parent index for every BIO seq
        for parent_idx, parent in filter(lambda node: node[1].is_parent, enumerate(flatten_spans)):
            bio_tags = ['O'] * len(pieces)
            bio_tag_list.append(bio_tags)
            bio_smooth: List[BIOSmoothing] = [parent.child_smooth.clone() for _ in pieces]
            bio_configs.append(bio_smooth)
            for child in parent:
                assert all(bio_tags[bio_idx] == 'O' for bio_idx in range(child.start_idx, child.end_idx + 1))
                if child.smooth_weight is not None:
                    for i in range(child.start_idx, child.end_idx+1):
                        bio_smooth[i].weight = child.smooth_weight
                bio_tags[child.start_idx] = 'B'
                for word_idx in range(child.start_idx + 1, child.end_idx + 1):
                    bio_tags[word_idx] = 'I'
            bio_seqs.append(apply_bio_smoothing(bio_smooth, bio_tags))

        fields['span_boundary'] = ArrayField(
            np.array(span_boundary), padding_value=0, dtype=np.int64
        )
        fields['parent_indices'] = ArrayField(np.array(span_parent_indices), 0, np.int64)
        if label_type == 'string':
            fields['span_labels'] = ListField([LabelField(label, 'span_label') for label in span_labels])
        elif label_type == 'list':
            fields['span_labels'] = ArrayField(np.array(span_labels))
        else:
            raise NotImplementedError
        fields['parent_mask'] = ArrayField(np.array(parent_mask), False, np.bool)
        fields['bio_seqs'] = ArrayField(np.stack(bio_seqs))

        self._sanity_check(
            flatten_spans, pieces, bio_tag_list, parent_mask, span_boundary, span_labels, span_parent_indices
        )

        return fields

    @staticmethod
    def _sanity_check(
            flatten_spans, words, bio_tag_list, parent_mask, span_boundary, span_labels, parent_indices, verbose=False
    ):
        # For debugging use.
        assert len(parent_mask) == len(span_boundary) == len(span_labels) == len(parent_indices)
        for (parent_idx, parent_span), bio_tags in zip(
                filter(lambda x: x[1].is_parent, enumerate(flatten_spans)), bio_tag_list
        ):
            assert parent_mask[parent_idx]
            parent_s, parent_e = span_boundary[parent_idx]
            if verbose:
                print('Parent: ', span_labels[parent_idx], 'Text: ', ' '.join(words[parent_s:parent_e+1]))
                print(f'It contains {len(parent_span)} children.')
            for child in parent_span:
                child_idx = flatten_spans.index(child)
                assert parent_indices[child_idx] == flatten_spans.index(parent_span)
                if verbose:
                    child_s, child_e = span_boundary[child_idx]
                    print('   ', span_labels[child_idx], 'Text', words[child_s:child_e+1])

            if verbose:
                print(f'Child derived from BIO tags:')
                for _, (start, end) in bio_tags_to_spans(bio_tags):
                    print(words[start:end+1])
