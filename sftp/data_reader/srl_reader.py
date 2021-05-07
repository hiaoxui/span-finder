import json
import logging
import random
from typing import *

import numpy as np
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import MetadataField
from allennlp.data.instance import Instance

from .span_reader import SpanReader
from ..utils import Span, VIRTUAL_ROOT, BIOSmoothing

logger = logging.getLogger(__name__)


@DatasetReader.register('semantic_role_labeling')
class SRLDatasetReader(SpanReader):
    def __init__(
            self,
            min_negative: int = 5,
            negative_ratio: float = 1.,
            event_only: bool = False,
            event_smoothing_factor: float = 0.,
            arg_smoothing_factor: float = 0.,
            # For Ontology Mapping
            ontology_mapping_path: Optional[str] = None,
            min_weight: float = 1e-2,
            max_weight: float = 1.0,
            **extra
    ):
        super().__init__(**extra)
        self.min_negative = min_negative
        self.negative_ratio = negative_ratio
        self.event_only = event_only
        self.event_smooth_factor = event_smoothing_factor
        self.arg_smooth_factor = arg_smoothing_factor
        self.ontology_mapping = None
        if ontology_mapping_path is not None:
            self.ontology_mapping = json.load(open(ontology_mapping_path))
            for k1 in ['event', 'argument']:
                for k2, weights in self.ontology_mapping['mapping'][k1].items():
                    weights = np.array(weights)
                    weights[weights < min_weight] = 0.0
                    weights[weights > max_weight] = max_weight
                    self.ontology_mapping['mapping'][k1][k2] = weights
                self.ontology_mapping['mapping'][k1] = {
                    k2: weights for k2, weights in self.ontology_mapping['mapping'][k1].items() if weights.sum() > 1e-5
                }
            vr_label = [0.] * len(self.ontology_mapping['target']['label'])
            vr_label[self.ontology_mapping['target']['label'].index(VIRTUAL_ROOT)] = 1.0
            self.ontology_mapping['mapping']['event'][VIRTUAL_ROOT] = np.array(vr_label)

    def _read(self, file_path: str) -> Iterable[Instance]:
        all_lines = list(map(json.loads, open(file_path).readlines()))
        if self.debug:
            random.seed(1); random.shuffle(all_lines)
        for line in all_lines:
            ins = self.text_to_instance(**line)
            if ins is not None:
                yield ins
        if self.n_span_removed > 0:
            logger.warning(f'{self.n_span_removed} spans are removed.')
        self.n_span_removed = 0

    def apply_ontology_mapping(self, vr):
        new_events = list()
        event_map, arg_map = self.ontology_mapping['mapping']['event'], self.ontology_mapping['mapping']['argument']
        for event in vr:
            if event.label not in event_map: continue
            event.child_smooth.weight = event.smooth_weight = event_map[event.label].sum()
            event = event.map_ontology(event_map, False, False)
            new_events.append(event)
            new_children = list()
            for child in event:
                if child.label not in arg_map: continue
                child.child_smooth.weight = child.smooth_weight = arg_map[child.label].sum()
                child = child.map_ontology(arg_map, False, False)
                new_children.append(child)
            event.remove_child()
            for child in new_children: event.add_child(child)
        new_vr = Span.virtual_root(new_events)
        # For Virtual Root itself.
        new_vr.map_ontology(self.ontology_mapping['mapping']['event'], True, False)
        return new_vr

    def text_to_instance(self, tokens, annotations=None, meta=None) -> Optional[Instance]:
        meta = meta or {'fully_annotated': True}
        meta['fully_annotated'] = meta.get('fully_annotated', True)
        vr = None
        if annotations is not None:
            vr = annotations if isinstance(annotations, Span) else Span.from_json(annotations)
            vr = self.apply_ontology_mapping(vr) if self.ontology_mapping is not None else vr
            # if len(vr) == 0: return  # Ignore sentence with empty annotation
            if self.event_smooth_factor != 0.0:
                vr.child_smooth = BIOSmoothing(o_smooth=self.event_smooth_factor if meta['fully_annotated'] else -1)
            if self.arg_smooth_factor != 0.0:
                for event in vr:
                    event.child_smooth = BIOSmoothing(o_smooth=self.arg_smooth_factor)
            if self.event_only:
                for event in vr:
                    event.remove_child()
                    event.is_parent = False

        fields = self.prepare_inputs(tokens, vr, True, 'string' if self.ontology_mapping is None else 'list')
        fields['meta'] = MetadataField(meta)
        return Instance(fields)
