from typing import *

import numpy as np

from .common import VIRTUAL_ROOT, DEFAULT_SPAN
from .bio_smoothing import BIOSmoothing
from .functions import max_match


class Span:
    """
    Span is a simple data structure for a span (not necessarily associated with text), along with its label,
    children and possibly its parent and a confidence score.

    Basic usages (suppose span is a Span object):
        1. len(span) -- #children.
        2. span[i] -- i-th child.
        3. for s in span: ... -- iterate its children.
        4. for s in span.bfs: ... -- iterate its descendents.
        5. print(span) -- show its description.
        6. span.tree() -- print the whole tree.

    It provides some utilities:
        1. Re-indexing. BPE will change token indices, and the `re_index` method can convert normal tokens
            BPE word piece indices, or vice versa.
        2. Span object and span dict (JSON format) are mutually convertible (by `to_json` and `from_json` methods).
        3. Recursively truncate spans up to a given length. (see `truncate` method)
        4. Recursively replace all labels with the default label. (see `ignore_labels` method)
        5. Recursively solve the span overlapping problem by removing children overlapped with others.
            (see `remove_overlapping` method)
    """
    def __init__(
            self,
            start_idx: int,
            end_idx: int,
            label: Union[str, int, list] = DEFAULT_SPAN,
            is_parent: bool = False,
            parent: Optional["Span"] = None,
            confidence: Optional[float] = None,
    ):
        """
        Init function. Children should be added using the `add_children` method.
        :param start_idx: Start index in a seq of tokens, inclusive.
        :param end_idx: End index in a seq of tokens, inclusive.
        :param label: Label. If not provided, will assign a default label.
            Can be of various types: String, integer, or list of something.
        :param is_parent: If True, will be treated as parent. This is important because in the training process of BIO
            tagger, when a span has no children, we need to know if it's a parent with no children (so we should have
            an training example with all O tags) or not (then the above example doesn't exist).
            We follow a convention where if a span is not parent, then the key `children` shouldn't appear in its
            JSON dict; if a span is parent but has no children, the key `children` in its JSON dict should appear
            and be an empty list.
        :param parent: A pointer to its parent.
        :param confidence: Confidence value.
        """
        self.start_idx, self.end_idx = start_idx, end_idx
        self.label: Union[int, str, list] = label
        self.is_parent = is_parent
        self.parent = parent
        self._children: List[Span] = list()
        self.confidence = confidence

        # Following are for label smoothing. Leave default is you don't need smoothing.
        # Logic:
        # The label smoothing factors of (i.e. b_smooth, i_smooth, o_smooth) depend on the `child_span` of its parent.
        # The re-weighting factor of a span also depends on the `child_span` of its parent, but can be overridden
        # by its own `smoothing_weight` field if it's not None.
        self.child_smooth: BIOSmoothing = BIOSmoothing()
        self.smooth_weight: Optional[float] = None

    def add_child(self, span: "Span") -> "Span":
        """
        Add a span to children list. Will link current span to child's parent pointer.
        :param span: Child span.
        """
        assert self.is_parent
        span.parent = self
        self._children.append(span)
        return self

    def re_index(
            self,
            offsets: List[Optional[Tuple[int, int]]],
            reverse: bool = False,
            recursive: bool = True,
            inplace: bool = False,
    ) -> "Span":
        """
        BPE will change token indices, and the `re_index` method can convert normal tokens BPE word piece indices,
        or vice versa.
        We assume Virtual Root has a boundary [-1, -1] before being mapped to the BPE space, and a boundary [0, 0]
        after the re-indexing. We use [0, 0] because it's always the BOS token in BPE.
        Mapping to BPE space is straight forward. The reverse mapping has special cases where the span might
        contain BOS or EOS. Usually this is a parsing bug. We will map the BOS index to 0, and EOS index to -1.
        :param offsets: Offsets. Defined by BPE tokenizer and resides in the SpanFinder outputs.
        :param reverse: If True, map from the BPE space to original token space.
        :param recursive: If True, will apply the re-indexing to its children.
        :param inplace: Inplace?
        :return: Re-indexed span.
        """
        span = self if inplace else self.clone()

        span.start_idx, span.end_idx = re_index_span(span.boundary, offsets, reverse)
        if recursive:
            new_children = list()
            for child in span._children:
                new_children.append(child.re_index(offsets, reverse, recursive, True))
            span._children = new_children
        return span

    def truncate(self, max_length: int) -> bool:
        """
        Discard spans whose end_idx exceeds the max_length (inclusive).
        This is done recursively.
        This is useful for some encoder like XLMR that has a limit on input length. (512 for XLMR large)
        :param max_length: Max length.
        :return: You don't need to care return value.
        """
        if self.end_idx >= max_length:
            return False
        else:
            self._children = list(filter(lambda x: x.truncate(max_length), self._children))
            return True

    @classmethod
    def virtual_root(cls: "Span", spans: Optional[List["Span"]] = None) -> "Span":
        """
        An official method to create a tree: Generate the first layer of spans by yourself, and pass them into this
        method.
        E.g., for SRL style task, generate a list of events, assign arguments to them as children. Then pass the
        events to this method to have a virtual root which serves as a parent of events.
        :param spans: 1st layer spans.
        :return: Virtual root.
        """
        vr = Span(-1, -1, VIRTUAL_ROOT, True)
        if spans is not None:
            vr._children = spans
        for child in vr._children:
            child.parent = vr
        return vr

    def ignore_labels(self) -> None:
        """
        Remove all labels. Make them placeholders. Inplace.
        """
        self.label = DEFAULT_SPAN
        for child in self._children:
            child.ignore_labels()

    def clone(self) -> "Span":
        """
        Clone a tree.
        :return: Cloned tree.
        """
        span = Span(self.start_idx, self.end_idx, self.label, self.is_parent, self.parent, self.confidence)
        span.child_smooth, span.smooth_weight = self.child_smooth, self.smooth_weight
        for child in self._children:
            span.add_child(child.clone())
        return span

    def bfs(self) -> Iterable["Span"]:
        """
        Iterate over all descendents with BFS, including self.
        :return: Spans.
        """
        yield self
        yield from self._bfs()

    def _bfs(self) -> List["Span"]:
        """
        Helper function.
        """
        for child in self._children:
            yield child
        for child in self._children:
            yield from child._bfs()

    def remove_overlapping(self, recursive=True) -> int:
        """
        Remove overlapped spans. If spans overlap, will pick the first one and discard the others, judged by start_idx.
        :param recursive: Apply to all of the descendents?
        :return: The number of spans that are removed.
        """
        indices = set()
        new_children = list()
        removing = 0
        for child in self._children:
            if len(set(range(child.start_idx, child.end_idx + 1)) & indices) > 0:
                removing += 1
                continue
            indices.update(set(range(child.start_idx, child.end_idx + 1)))
            new_children.append(child)
            if recursive:
                removing += child.remove_overlapping(True)
        self._children = new_children
        return removing

    def describe(self, sentence: Optional[List[str]] = None) -> str:
        """
        :param sentence: If provided, will replace the indices with real tokens for presentation.
        :return: The description in a single line.
        """
        if self.start_idx >= 0:
            if sentence is None:
                span = f'({self.start_idx}, {self.end_idx})'
            else:
                span = '(' + ' '.join(sentence[self.start_idx: self.end_idx + 1]) + ')'
            if self.is_parent:
                return f'<Span: {span}, {self.label}, {len(self._children)} children>'
            else:
                return f'[Span: {span}, {self.label}]'
        else:
            return f'<Span Annotation: {self.n_nodes - 1} descendents>'

    def __repr__(self) -> str:
        return self.describe()

    @property
    def n_nodes(self) -> int:
        """
        :return: Number of descendents + self.
        """
        return sum([child.n_nodes for child in self._children], 1)

    @property
    def boundary(self):
        """
        :return: (start_idx, end_idx), both inclusive.
        """
        return self.start_idx, self.end_idx

    def __iter__(self) -> Iterable["Span"]:
        """
        Iterate over children.
        """
        yield from self._children

    def __len__(self):
        """
        :return: #children.
        """
        return len(self._children)

    def __getitem__(self, idx: int):
        """
        :return: The indexed child.
        """
        return self._children[idx]

    def tree(self, sentence: Optional[List[str]] = None, printing: bool = True) -> str:
        """
        A tree description of all descendents. Human readable.
        :param sentence: If provided, will replace the indices with real tokens for presentation.
        :param printing: If True, will print out.
        :return: The description.
        """
        ret = list()
        ret.append(self.describe(sentence))
        for child in self._children:
            child_lines = child.tree(sentence, False).split('\n')
            for line in child_lines:
                ret.append('  ' + line)
        desc = '\n'.join(ret)
        if printing: print(desc)
        else: return desc

    def match(
            self,
            other: "Span",
            match_label: bool = True,
            depth: int = -1,
            ignore_parent_boundary: bool = False,
    ) -> int:
        """
        Used for evaluation. Count how many spans two trees share. Two spans are considered to be identical
        if their boundary, label, and parent match.
        :param other: The other tree to compare.
        :param match_label: If False, will ignore label.
        :param depth: If specified as non-negative, will only search thru certain depth.
        :param ignore_parent_boundary: If True, two children can be matched ignoring parent boundaries.
        :return: #spans two tree share.
        """
        if depth == 0:
            return 0
        if self.label != other.label and match_label:
            return 0
        if self.boundary == other.boundary:
            n_match = 1
        elif ignore_parent_boundary:
            # Parents fail, Children might match!
            n_match = 0
        else:
            return 0

        sub_matches = np.zeros([len(self), len(other)], dtype=np.int)
        for self_idx, my_child in enumerate(self):
            for other_idx, other_child in enumerate(other):
                sub_matches[self_idx, other_idx] = my_child.match(
                    other_child, match_label, depth-1, ignore_parent_boundary
                )
        if not ignore_parent_boundary:
            for m in [sub_matches, sub_matches.T]:
                for line in m:
                    assert (line > 0).sum() <= 1
        n_match += max_match(sub_matches)
        return n_match

    def to_json(self) -> dict:
        """
        To JSON dict format. See init.
        """
        ret = {
            "label": self.label,
            "span": list(self.boundary),
        }
        if self.confidence is not None:
            ret['confidence'] = self.confidence
        if self.is_parent:
            children = list()
            for child in self._children:
                children.append(child.to_json())
            ret['children'] = children
        return ret

    @classmethod
    def from_json(cls, span_json: Union[list, dict]) -> "Span":
        """
        Load from JSON. See init.
        """
        if isinstance(span_json, dict):
            span = Span(
                span_json['span'][0], span_json['span'][1], span_json.get('label', None), 'children' in span_json,
                confidence=span_json.get('confidence', None)
            )
            for child_dict in span_json.get('children', []):
                span.add_child(Span.from_json(child_dict))
        else:
            spans = [Span.from_json(child) for child in span_json]
            span = Span.virtual_root(spans)
        return span

    def map_ontology(
            self,
            ontology_mapping: Optional[dict] = None,
            inplace: bool = True,
            recursive: bool = True,
    ) -> Optional["Span"]:
        """
        Map labels to other things, like another ontology of soft labels.
        :param ontology_mapping: Mapping dict. The key should be labels, and values can be anything.
            Labels not in the dict will not be deleted. So be careful.
        :param inplace: Inplace?
        :param recursive: Apply to all descendents if True.
        :return: The mapped tree.
        """
        span = self if inplace else self.clone()
        if ontology_mapping is None:
            # Do nothing if mapping not provided.
            return span

        if recursive:
            new_children = list()
            for child in span:
                new_child = child.map_ontology(ontology_mapping, False, True)
                if new_child is not None:
                    new_children.append(new_child)
            span._children = new_children

        if span.label != VIRTUAL_ROOT:
            if span.parent is not None and (span.parent.label, span.label) in ontology_mapping:
                span.label = ontology_mapping[(span.parent.label, span.label)]
            elif span.label in ontology_mapping:
                span.label = ontology_mapping[span.label]
            else:
                return

        return span

    def isolate(self) -> "Span":
        """
        Generate a span that is identical to self but has no children or parent.
        """
        return Span(self.start_idx, self.end_idx, self.label, self.is_parent, None, self.confidence)

    def remove_child(self, span: Optional["Span"] = None):
        """
        Remove a child. If pass None, will reset the children list.
        """
        if span is None:
            self._children = list()
        else:
            del self._children[self._children.index(span)]


def re_index_span(
        boundary: Tuple[int, int], offsets: List[Tuple[int, int]], reverse: bool = False
) -> Tuple[int, int]:
    """
    Helper function.
    """
    if not reverse:
        if boundary[0] == boundary[1] == -1:
            # Virtual Root
            start_idx = end_idx = 0
        else:
            start_idx = offsets[boundary[0]][0]
            end_idx = offsets[boundary[1]][1]
    else:
        if boundary[0] == boundary[1] == 0:
            # Virtual Root
            start_idx = end_idx = -1
        else:
            start_within = [bo[0] <= boundary[0] <= bo[1] if bo is not None else False for bo in offsets]
            end_within = [bo[0] <= boundary[1] <= bo[1] if bo is not None else False for bo in offsets]
            assert sum(start_within) <= 1 and sum(end_within) <= 1
            start_idx = start_within.index(True) if sum(start_within) == 1 else 0
            end_idx = end_within.index(True) if sum(end_within) == 1 else len(offsets)
            if start_idx > end_idx:
                raise IndexError
    return start_idx, end_idx
