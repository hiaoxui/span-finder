import json
import logging
import os
from collections import defaultdict, namedtuple
from typing import *

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance

from .span_reader import SpanReader
from ..utils import Span

# logging.basicConfig(level=logging.DEBUG)

# for v in logging.Logger.manager.loggerDict.values():
# v.disabled = True

logger = logging.getLogger(__name__)

SpanTuple = namedtuple('Span', ['start', 'end'])


@DatasetReader.register('better')
class BetterDatasetReader(SpanReader):
    def __init__(
            self,
            eval_type,
            consolidation_strategy='first',
            span_set_type='single',
            max_argument_ss_size=1,
            use_ref_events=False,
            **extra
    ):
        super().__init__(**extra)
        self.eval_type = eval_type
        assert self.eval_type in ['abstract', 'basic']

        self.consolidation_strategy = consolidation_strategy
        self.unitary_spans = span_set_type == 'single'
        # event anchors are always singleton spans
        self.max_arg_spans = max_argument_ss_size
        self.use_ref_events = use_ref_events

        self.n_overlap_arg = 0
        self.n_overlap_trigger = 0
        self.n_skip = 0
        self.n_too_long = 0

    @staticmethod
    def post_process_basic_span(predicted_span, basic_entry):
        # Convert token offsets back to characters, also get the text spans as a sanity check

        # !!!!!
        # SF outputs inclusive idxs
        # char offsets are inc-exc
        # token offsets are inc-inc
        # !!!!!

        start_idx = predicted_span['start_idx']  # inc
        end_idx = predicted_span['end_idx']  # inc

        char_start_idx = basic_entry['tok2char'][predicted_span['start_idx']][0]  # inc
        char_end_idx = basic_entry['tok2char'][predicted_span['end_idx']][-1] + 1  # exc

        span_text = basic_entry['segment-text'][char_start_idx:char_end_idx]  # inc exc
        span_text_tok = basic_entry['segment-text-tok'][start_idx:end_idx + 1]  # inc exc

        span = {'string': span_text,
                'start': char_start_idx,
                'end': char_end_idx,
                'start-token': start_idx,
                'end-token': end_idx,
                'string-tok': span_text_tok,
                'label': predicted_span['label'],
                'predicted': True}
        return span

    @staticmethod
    def _get_shortest_span(spans):
        # shortest_span_length = float('inf')
        # shortest_span = None
        # for span in spans:
        # span_tokens = span['string-tok']
        # span_length = len(span_tokens)
        # if span_length < shortest_span_length:
        # shortest_span_length = span_length
        # shortest_span = span

        # return shortest_span
        return [s[-1] for s in sorted([(len(span['string']), ix, span) for ix, span in enumerate(spans)])]

    @staticmethod
    def _get_first_span(spans):
        spans = [(span['start'], -len(span['string']), ix, span) for ix, span in enumerate(spans)]
        try:
            return [s[-1] for s in sorted(spans)]
        except:
            breakpoint()

    @staticmethod
    def _get_longest_span(spans):
        return [s[-1] for s in sorted([(len(span['string']), ix, span) for ix, span in enumerate(spans)], reverse=True)]

    @staticmethod
    def _subfinder(text, pattern):
        # https://stackoverflow.com/a/12576755
        matches = []
        pattern_length = len(pattern)
        for i, token in enumerate(text):
            try:
                if token == pattern[0] and text[i:i + pattern_length] == pattern:
                    matches.append(SpanTuple(start=i, end=i + pattern_length - 1))  # inclusive boundaries
            except:
                continue
        return matches

    def consolidate_span_set(self, spans):
        if self.consolidation_strategy == 'first':
            spans = BetterDatasetReader._get_first_span(spans)
        elif self.consolidation_strategy == 'shortest':
            spans = BetterDatasetReader._get_shortest_span(spans)
        elif self.consolidation_strategy == 'longest':
            spans = BetterDatasetReader._get_longest_span(spans)
        else:
            raise NotImplementedError(f"{self.consolidation_strategy} does not exist")

        if self.unitary_spans:
            spans = [spans[0]]
        else:
            spans = spans[:self.max_arg_spans]

        # TODO add some sanity checks here

        return spans

    def get_mention_spans(self, text: List[str], span_sets: Dict):
        mention_spans = defaultdict(list)
        for span_set_id in span_sets.keys():
            spans = span_sets[span_set_id]['spans']
            # span = BetterDatasetReader._get_shortest_span(spans)
            # span = BetterDatasetReader._get_earliest_span(spans)
            consolidated_spans = self.consolidate_span_set(spans)
            # if len(spans) > 1:
            # logging.info(f"Truncated a spanset from {len(spans)} spans to 1")

            if self.eval_type == 'abstract':
                span = consolidated_spans[0]
                span_tokens = span['string-tok']

                span_indices = BetterDatasetReader._subfinder(text=text, pattern=span_tokens)

                if len(span_indices) > 1:
                    pass

                if len(span_indices) == 0:
                    continue

                mention_spans[span_set_id] = span_indices[0]
            else:
                # in basic, we already have token offsets in the right form

                # if not span['string-tok'] == text[span['start-token']:span['end-token'] + 1]:
                # print(span, text[span['start-token']:span['end-token'] + 1])

                # we should use these token offsets only!
                for span in consolidated_spans:
                    mention_spans[span_set_id].append(SpanTuple(start=span['start-token'], end=span['end-token']))

        return mention_spans

    def _read_single_file(self, file_path):
        with open(file_path) as fp:
            json_content = json.load(fp)
        if 'entries' in json_content:
            for doc_name, entry in json_content['entries'].items():
                instance = self.text_to_instance(entry, 'train' in file_path)
                yield instance
        else:  # TODO why is this split in 2 cases?
            for doc_name, entry in json_content.items():
                instance = self.text_to_instance(entry, True)
                yield instance

        logger.warning(f'{self.n_overlap_arg} overlapped args detected!')
        logger.warning(f'{self.n_overlap_trigger} overlapped triggers detected!')
        logger.warning(f'{self.n_skip} skipped detected!')
        logger.warning(f'{self.n_too_long} were skipped because they are too long!')
        self.n_overlap_arg = self.n_skip = self.n_too_long = self.n_overlap_trigger = 0

    def _read(self, file_path: str) -> Iterable[Instance]:

        if os.path.isdir(file_path):
            for fn in os.listdir(file_path):
                if not fn.endswith('.json'):
                    logger.info(f'Skipping {fn}')
                    continue
                logger.info(f'Loading from {fn}')
                yield from self._read_single_file(os.path.join(file_path, fn))
        else:
            yield from self._read_single_file(file_path)

    def text_to_instance(self, entry, is_training=False):
        word_tokens = entry['segment-text-tok']

        # span sets have been trimmed to the earliest span mention
        spans = self.get_mention_spans(
            word_tokens, entry['annotation-sets'][f'{self.eval_type}-events']['span-sets']
        )

        # idx of every token that is a part of an event trigger/anchor span
        all_trigger_idxs = set()

        # actual inputs to the model
        input_spans = []

        self._local_child_overlap = 0
        self._local_child_total = 0

        better_events = entry['annotation-sets'][f'{self.eval_type}-events']['events']

        skipped_events = set()
        # check for events that overlap other event's anchors, skip them later
        for event_id, event in better_events.items():
            assert event['anchors'] in spans

            # take the first consolidated span for anchors
            anchor_start, anchor_end = spans[event['anchors']][0]

            if any(ix in all_trigger_idxs for ix in range(anchor_start, anchor_end + 1)):
                logger.warning(
                    f"Skipped {event_id} with anchor span {event['anchors']}, overlaps a previously found event trigger/anchor")
                self.n_overlap_trigger += 1
                skipped_events.add(event_id)
                continue

            all_trigger_idxs.update(range(anchor_start, anchor_end + 1))  # record the trigger

        for event_id, event in better_events.items():
            if event_id in skipped_events:
                continue

            # arguments for just this event
            local_arg_idxs = set()
            # take the first consolidated span for anchors
            anchor_start, anchor_end = spans[event['anchors']][0]

            event_span = Span(anchor_start, anchor_end, event['event-type'], True)
            input_spans.append(event_span)

            def add_a_child(span_id, label):
                # TODO this is a bad way to do this
                assert span_id in spans
                for child_span in spans[span_id]:
                    self._local_child_total += 1
                    arg_start, arg_end = child_span

                    if any(ix in local_arg_idxs for ix in range(arg_start, arg_end + 1)):
                        # logger.warn(f"Skipped argument {span_id}, overlaps a previously found argument")
                        # print(entry['annotation-sets'][f'{self.eval_type}-events']['span-sets'][span_id])
                        self.n_overlap_arg += 1
                        self._local_child_overlap += 1
                        continue

                    local_arg_idxs.update(range(arg_start, arg_end + 1))
                    event_span.add_child(Span(arg_start, arg_end, label, False))

            for agent in event['agents']:
                add_a_child(agent, 'agent')
            for patient in event['patients']:
                add_a_child(patient, 'patient')

            if self.use_ref_events:
                for ref_event in event['ref-events']:
                    if ref_event in skipped_events:
                        continue
                    ref_event_anchor_id = better_events[ref_event]['anchors']
                    add_a_child(ref_event_anchor_id, 'ref-event')

            # if len(event['ref-events']) > 0:
            # breakpoint()

        fields = self.prepare_inputs(word_tokens, spans=input_spans)
        if self._local_child_overlap > 0:
            logging.warning(
                f"Skipped {self._local_child_overlap} / {self._local_child_total} argument spans due to overlaps")
        return Instance(fields)

