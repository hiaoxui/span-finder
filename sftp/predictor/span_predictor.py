import os
from time import time
from typing import *
import json

import numpy as np
import torch
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.data.samplers import MaxTokensBatchSampler
from allennlp.data.tokenizers import SpacyTokenizer
from allennlp.models import Model
from allennlp.nn import util as nn_util
from allennlp.predictors import Predictor
from concrete import (
    MentionArgument, SituationMentionSet, SituationMention, TokenRefSequence,
    EntityMention, EntityMentionSet, Entity, EntitySet, AnnotationMetadata, Communication
)
from concrete.util import CommunicationReader, AnalyticUUIDGeneratorFactory, CommunicationWriterZip
from concrete.validate import validate_communication
from tqdm import tqdm

from ..data_reader import concrete_doc, concrete_doc_tokenized
from ..utils import Span, re_index_span, VIRTUAL_ROOT


class PredictionReturn(NamedTuple):
    span: Union[Span, dict, Communication]
    sentence: List[str]
    meta: Dict[str, Any]


class ForceDecodingReturn(NamedTuple):
    span: np.ndarray
    label: List[str]
    distribution: np.ndarray


@Predictor.register('span')
class SpanPredictor(Predictor):
    @staticmethod
    def format_convert(
            sentence: Union[List[str], List[List[str]]],
            prediction: Union[Span, List[Span]],
            output_format: str
    ):
        if output_format == 'span':
            return prediction
        elif output_format == 'json':
            if isinstance(prediction, list):
                return [SpanPredictor.format_convert(sent, pred, 'json') for sent, pred in zip(sentence, prediction)]
            return prediction.to_json()
        elif output_format == 'concrete':
            if isinstance(prediction, Span):
                sentence, prediction = [sentence], [prediction]
            return concrete_doc_tokenized(sentence, prediction)

    def predict_concrete(
            self,
            concrete_path: str,
            output_path: Optional[str] = None,
            max_tokens: int = 2048,
            ontology_mapping: Optional[Dict[str, str]] = None,
    ):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        writer = CommunicationWriterZip(output_path)

        for comm, fn in CommunicationReader(concrete_path):
            assert len(comm.sectionList) == 1
            concrete_sentences = comm.sectionList[0].sentenceList
            json_sentences = list()
            for con_sent in concrete_sentences:
                json_sentences.append(
                    [t.text for t in con_sent.tokenization.tokenList.tokenList]
                )
            predictions = self.predict_batch_sentences(json_sentences, max_tokens, ontology_mapping=ontology_mapping)

            # Merge predictions into concrete
            aug = AnalyticUUIDGeneratorFactory(comm).create()
            situation_mention_set = SituationMentionSet(next(aug), AnnotationMetadata('Span Finder', time()), list())
            comm.situationMentionSetList = [situation_mention_set]
            situation_mention_set.mentionList = sm_list = list()
            entity_mention_set = EntityMentionSet(next(aug), AnnotationMetadata('Span Finder', time()), list())
            comm.entityMentionSetList = [entity_mention_set]
            entity_mention_set.mentionList = em_list = list()
            entity_set = EntitySet(
                next(aug), AnnotationMetadata('Span Finder', time()), list(), None, entity_mention_set.uuid
            )
            comm.entitySetList = [entity_set]

            em_dict = dict()
            for con_sent, pred in zip(concrete_sentences, predictions):
                for event in pred.span:
                    def raw_text_span(start_idx, end_idx, **_):
                        si_char = con_sent.tokenization.tokenList.tokenList[start_idx].textSpan.start
                        ei_char = con_sent.tokenization.tokenList.tokenList[end_idx].textSpan.ending
                        return comm.text[si_char:ei_char]
                    sm = SituationMention(
                        next(aug),
                        text=raw_text_span(event.start_idx, event.end_idx),
                        situationKind=event.label,
                        situationType='EVENT',
                        confidence=event.confidence,
                        argumentList=list(),
                        tokens=TokenRefSequence(
                            tokenIndexList=list(range(event.start_idx, event.end_idx+1)),
                            tokenizationId=con_sent.tokenization.uuid
                        )
                    )

                    for arg in event:
                        em = em_dict.get((arg.start_idx, arg.end_idx + 1))
                        if em is None:
                            em = EntityMention(
                                next(aug),
                                tokens=TokenRefSequence(
                                    tokenIndexList=list(range(arg.start_idx, arg.end_idx+1)),
                                    tokenizationId=con_sent.tokenization.uuid,
                                ),
                                text=raw_text_span(arg.start_idx, arg.end_idx)
                            )
                            em_list.append(em)
                            entity_set.entityList.append(Entity(next(aug), id=em.text, mentionIdList=[em.uuid]))
                            em_dict[(arg.start_idx, arg.end_idx+1)] = em
                        sm.argumentList.append(MentionArgument(
                            role=arg.label,
                            entityMentionId=em.uuid,
                            confidence=arg.confidence
                        ))
                    sm_list.append(sm)
            validate_communication(comm)
            writer.write(comm, fn)
        writer.close()

    def predict_sentence(
            self,
            sentence: Union[str, List[str]],
            ontology_mapping: Optional[Dict[str, str]] = None,
            output_format: str = 'span',
    ) -> PredictionReturn:
        """
        Predict spans on a single sentence (no batch). If not tokenized, will tokenize it with SpacyTokenizer.
        :param sentence: If tokenized, should be a list of tokens in string. If not, should be a string.
        :param ontology_mapping:
        :param output_format: span, json or concrete.
        """
        prediction = self.predict_json(self._prepare_sentence(sentence))
        prediction['prediction'] = self.format_convert(
            prediction['sentence'],
            Span.from_json(prediction['prediction']).map_ontology(ontology_mapping),
            output_format
        )
        return PredictionReturn(prediction['prediction'], prediction['sentence'], prediction.get('meta', dict()))

    def predict_batch_sentences(
            self,
            sentences: List[Union[List[str], str]],
            max_tokens: int = 512,
            ontology_mapping: Optional[Dict[str, str]] = None,
            output_format: str = 'span',
            progress: bool = False,
    ) -> List[PredictionReturn]:
        """
        Predict spans on a batch of sentences. If not tokenized, will tokenize it with SpacyTokenizer.
        :param sentences: A list of sentences. Refer to `predict_sentence`.
        :param max_tokens: Maximum tokens in a batch.
        :param ontology_mapping: If not None, will try to map the output from one ontology to another.
            If the predicted frame is not in the mapping, the prediction will be ignored.
        :param output_format: span, json or concrete.
        :param progress: If True, the progress bar will be displayed.
        :return: A list of predictions.
        """
        sentences = list(map(self._prepare_sentence, sentences))
        for i_sent, sent in enumerate(sentences):
            sent['meta'] = {"idx": i_sent}
        instances = list(map(self._json_to_instance, sentences))
        outputs = list()
        batches = list(MaxTokensBatchSampler(max_tokens, ["tokens"], 0.0).get_batch_indices(instances))
        for ins_indices in tqdm(batches, disable=not progress):
            batch_ins = list(
                SimpleDataLoader([instances[ins_idx] for ins_idx in ins_indices], len(ins_indices), vocab=self.vocab)
            )[0]
            batch_inputs = nn_util.move_to_device(batch_ins, device=self.cuda_device)
            batch_outputs = self._model(**batch_inputs)
            for meta, prediction, inputs in zip(
                batch_outputs['meta'], batch_outputs['prediction'], batch_outputs['inputs']
            ):
                prediction.map_ontology(ontology_mapping)
                prediction = self.format_convert(inputs['sentence'], prediction, output_format)
                outputs.append(PredictionReturn(prediction, inputs['sentence'], {"input_idx": meta['idx']}))

        outputs.sort(key=lambda x: x.meta['input_idx'])
        return outputs

    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        outputs = sanitize(outputs)
        return {
            'prediction': outputs['prediction'],
            'sentence': outputs['inputs']['sentence'],
            'meta': outputs.get('meta', {})
        }

    def __init__(
            self,
            model: Model,
            dataset_reader: DatasetReader,
            frozen: bool = True,
    ):
        super(SpanPredictor, self).__init__(model=model, dataset_reader=dataset_reader, frozen=frozen)
        self.spacy_tokenizer = SpacyTokenizer(language='en_core_web_sm')

    def economize(
            self,
            max_decoding_spans: Optional[int] = None,
            max_recursion_depth: Optional[int] = None,
    ):
        if max_decoding_spans:
            self._model._max_decoding_spans = max_decoding_spans
        if max_recursion_depth:
            self._model._max_recursion_depth = max_recursion_depth

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self._dataset_reader.text_to_instance(**json_dict)

    @staticmethod
    def to_nested(prediction: List[dict]):
        first_layer, idx2children = list(), dict()
        for idx, pred in enumerate(prediction):
            children = list()
            pred['children'] = idx2children[idx+1] = children
            if pred['parent'] == 0:
                first_layer.append(pred)
            else:
                idx2children[pred['parent']].append(pred)
            del pred['parent']
        return first_layer

    def _prepare_sentence(self, sentence: Union[str, List[str]]) -> Dict[str, List[str]]:
        if isinstance(sentence, str):
            while '  ' in sentence:
                sentence = sentence.replace('  ', ' ')
            sentence = sentence.replace(chr(65533), '')
            if sentence == '':
                sentence = [""]
            sentence = list(map(str, self.spacy_tokenizer.tokenize(sentence)))
        return {"tokens": sentence}

    @staticmethod
    def json_to_concrete(
            predictions: List[dict],
    ):
        sentences = list()
        for pred in predictions:
            tokenization, event = list(), list()
            sent = {'text': ' '.join(pred['inputs']), 'tokenization': tokenization, 'event': event}
            sentences.append(sent)
            start_idx = 0
            for token in pred['inputs']:
                tokenization.append((start_idx, len(token)-1+start_idx))
                start_idx += len(token) + 1
            for pred_event in pred['prediction']:
                arg_list = list()
                one_event = {'argument': arg_list}
                event.append(one_event)
                for key in ['start_idx', 'end_idx', 'label']:
                    one_event[key] = pred_event[key]
                for pred_arg in pred_event['children']:
                    arg_list.append({key: pred_arg[key] for key in ['start_idx', 'end_idx', 'label']})

        concrete_comm = concrete_doc(sentences)
        return concrete_comm

    def force_decode(
            self,
            sentence: List[str],
            parent_span: Tuple[int, int] = (-1, -1),
            parent_label: str = VIRTUAL_ROOT,
            child_spans: Optional[List[Tuple[int, int]]] = None,
    ) -> ForceDecodingReturn:
        """
        Force decoding. There are 2 modes:
        1. Given parent span and its label, find all it children (direct children, not including other descendents)
            and type them.
        2. Given parent span, parent label, and children spans, type all children.
        :param sentence: Tokens.
        :param parent_span: [start_idx, end_idx], both inclusive.
        :param parent_label: Parent label in string.
        :param child_spans: Optional. If provided, will turn to mode 2; else mode 1.
        :return:
            - span: children spans.
            - label: most probable labels of children.
            - distribution: distribution over children labels.
        """
        instance = self._dataset_reader.text_to_instance(self._prepare_sentence(sentence)['tokens'])
        model_input = nn_util.move_to_device(
            list(SimpleDataLoader([instance], 1, vocab=self.vocab))[0], device=self.cuda_device
        )
        offsets = instance.fields['raw_inputs'].metadata['offsets']

        with torch.no_grad():
            tokens = model_input['tokens']
            parent_span = re_index_span(parent_span, offsets)
            if parent_span[1] >= self._dataset_reader.max_length:
                return ForceDecodingReturn(
                    np.zeros([0, 2], dtype=np.int),
                    [],
                    np.zeros([0, self.vocab.get_vocab_size('span_label')], dtype=np.float64)
                )
            if child_spans is not None:
                token_vec = self._model.word_embedding(tokens)
                child_pieces = [re_index_span(bdr, offsets) for bdr in child_spans]
                child_pieces = list(filter(lambda x: x[1] < self._dataset_reader.max_length-1, child_pieces))
                span_tensor = torch.tensor(
                    [parent_span] + child_pieces, dtype=torch.int64, device=self.device
                ).unsqueeze(0)
                parent_indices = span_tensor.new_zeros(span_tensor.shape[0:2])
                span_labels = parent_indices.new_full(
                    parent_indices.shape, self._model.vocab.get_token_index(parent_label, 'span_label')
                )
                span_vec = self._model._span_extractor(token_vec, span_tensor)
                typing_out = self._model._span_typing(span_vec, parent_indices, span_labels)
                distribution = typing_out['distribution'][0, 1:].cpu().numpy()
                boundary = np.array(child_spans)
            else:
                parent_label_tensor = torch.tensor(
                    [self._model.vocab.get_token_index(parent_label, 'span_label')], device=self.device
                )
                parent_boundary_tensor = torch.tensor([parent_span], device=self.device)
                boundary, _, num_children, distribution = self._model.one_step_prediction(
                    tokens, parent_boundary_tensor, parent_label_tensor
                )
                boundary, distribution = boundary[0].cpu().tolist(), distribution[0].cpu().numpy()
                boundary = np.array([re_index_span(bdr, offsets, True) for bdr in boundary])

            labels = [
                self.vocab.get_token_from_index(label_idx, 'span_label') for label_idx in distribution.argmax(1)
            ]
            return ForceDecodingReturn(boundary, labels, distribution)

    @property
    def vocab(self):
        return self._model.vocab

    @property
    def device(self):
        return self.cuda_device if self.cuda_device > -1 else 'cpu'

    @staticmethod
    def read_ontology_mapping(file_path: str):
        """
        Read the ontology mapping file. The file format can be read in docs.
        """
        if file_path is None:
            return None
        if file_path.endswith('.json'):
            return json.load(open(file_path))
        mapping = dict()
        for line in open(file_path).readlines():
            parent_label, original_label, new_label = line.replace('\n', '').split('\t')
            if parent_label == '*':
                mapping[original_label] = new_label
            else:
                mapping[(parent_label, original_label)] = new_label
        return mapping
