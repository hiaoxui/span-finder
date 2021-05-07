import logging
from collections import defaultdict
from typing import *
import os

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from concrete import SituationMention
from concrete.util import CommunicationReader

from .span_reader import SpanReader
from .srl_reader import SRLDatasetReader
from .concrete_srl import collect_concrete_srl
from ..utils import Span, BIOSmoothing

logger = logging.getLogger(__name__)


@DatasetReader.register('concrete')
class ConcreteDatasetReader(SRLDatasetReader):
    def __init__(
            self,
            event_only: bool = False,
            event_smoothing_factor: float = 0.,
            arg_smoothing_factor: float = 0.,
            **extra
    ):
        super().__init__(**extra)
        self.event_only = event_only
        self.event_only = event_only
        self.event_smooth_factor = event_smoothing_factor
        self.arg_smooth_factor = arg_smoothing_factor

    def _read(self, file_path: str) -> Iterable[Instance]:
        if os.path.isdir(file_path):
            for fn in os.listdir(file_path):
                yield from self._read(os.path.join(file_path, fn))
        all_files = CommunicationReader(file_path)
        for comm, fn in all_files:
            sentences = collect_concrete_srl(comm)
            for tokens, vr in sentences:
                yield self.text_to_instance(tokens, vr)
        logger.warning(f'{self.n_span_removed} spans were removed')
        self.n_span_removed = 0
