from time import time
from typing import *
from collections import defaultdict

from concrete import (
    Token, TokenList, TextSpan, MentionArgument, SituationMentionSet, SituationMention, TokenRefSequence,
    Communication, EntityMention, EntityMentionSet, Entity, EntitySet, AnnotationMetadata, Sentence
)
from concrete.util import create_comm, AnalyticUUIDGeneratorFactory
from concrete.validate import validate_communication

from ..utils import Span


def _process_sentence(sent, comm_sent, aug, char_idx_offset: int):
    token_list = list()
    for tok_idx, (start_idx, end_idx) in enumerate(sent['tokenization']):
        token_list.append(Token(
            tokenIndex=tok_idx,
            text=sent['sentence'][start_idx:end_idx + 1],
            textSpan=TextSpan(
                start=start_idx + char_idx_offset,
                ending=end_idx + char_idx_offset + 1
            ),
        ))
    comm_sent.tokenization.tokenList = TokenList(tokenList=token_list)

    sm_list, em_dict, entity_list = list(), dict(), list()

    annotation = sent['annotations'] if isinstance(sent['annotations'], Span) else Span.from_json(sent['annotations'])
    for event in annotation:
        char_start_idx = sent['tokenization'][event.start_idx][0]
        char_end_idx = sent['tokenization'][event.end_idx][1]
        sm = SituationMention(
            uuid=next(aug),
            text=sent['sentence'][char_start_idx: char_end_idx + 1],
            situationType='EVENT',
            situationKind=event.label,
            argumentList=list(),
            tokens=TokenRefSequence(
                tokenIndexList=list(range(event.start_idx, event.end_idx + 1)),
                tokenizationId=comm_sent.tokenization.uuid
            ),
        )

        for arg in event:
            em = em_dict.get((arg.start_idx, arg.end_idx + 1))
            if em is None:
                char_start_idx = sent['tokenization'][arg.start_idx][0]
                char_end_idx = sent['tokenization'][arg.end_idx][1]
                em = EntityMention(next(aug), TokenRefSequence(
                    tokenIndexList=list(range(arg.start_idx, arg.end_idx + 1)),
                    tokenizationId=comm_sent.tokenization.uuid,
                ), text=sent['sentence'][char_start_idx: char_end_idx + 1])
                entity_list.append(Entity(next(aug), id=em.text, mentionIdList=[em.uuid]))
                em_dict[(arg.start_idx, arg.end_idx + 1)] = em
            sm.argumentList.append(MentionArgument(
                role=arg.label,
                entityMentionId=em.uuid,
            ))

        sm_list.append(sm)

    return sm_list, list(em_dict.values()), entity_list


def concrete_doc(
        sentences: List[Dict[str, Any]],
        doc_name: str = 'document',
) -> Communication:
    """
    Data format: A list of sentences. Each sentence should be a dict of the following format:
    {
        "sentence": String.
        "tokenization": A list of Tuple[int, int] for start and end indices. Both inclusive.
        "annotations": A list of event dict, or Span object.
    }
    If it is dict, its format should be:

        Each event should be a dict of the following format:
        {
            "span": [start_idx, end_idx]: Integer. Both inclusive.
            "label": String.
            "children": A list of arguments.
        }
        Each argument should be a dict of the following format:
        {
            "span": [start_idx, end_idx]: Integer. Both inclusive.
            "label": String.
        }

    Note the "indices" above all refer to the indices of tokens, instead of characters.
    """
    comm = create_comm(
        doc_name,
        '\n'.join([sent['sentence'] for sent in sentences]),
    )
    aug = AnalyticUUIDGeneratorFactory(comm).create()
    situation_mention_set = SituationMentionSet(next(aug), AnnotationMetadata('Span Finder', time()), list())
    comm.situationMentionSetList = [situation_mention_set]
    entity_mention_set = EntityMentionSet(next(aug), AnnotationMetadata('Span Finder', time()), list())
    comm.entityMentionSetList = [entity_mention_set]
    entity_set = EntitySet(
        next(aug), AnnotationMetadata('O(0) Coref Paser.', time()), list(), None, entity_mention_set.uuid
    )
    comm.entitySetList = [entity_set]
    assert len(sentences) == len(comm.sectionList[0].sentenceList)

    char_idx_offset = 0
    for sent, comm_sent in zip(sentences, comm.sectionList[0].sentenceList):
        sm_list, em_list, entity_list = _process_sentence(sent, comm_sent, aug, char_idx_offset)
        entity_set.entityList.extend(entity_list)
        situation_mention_set.mentionList.extend(sm_list)
        entity_mention_set.mentionList.extend(em_list)
        char_idx_offset += len(sent['sentence']) + 1

    validate_communication(comm)
    return comm


def concrete_doc_tokenized(
        sentences: List[List[str]],
        spans: List[Span],
        doc_name: str = "document",
):
    """
    Similar to concrete_doc, but with tokenized words and spans.
    """
    inputs = list()
    for sent, vr in zip(sentences, spans):
        cur_start = 0
        tokenization = list()
        for token in sent:
            tokenization.append((cur_start, cur_start + len(token) - 1))
            cur_start += len(token) + 1
        inputs.append({
            "sentence": " ".join(sent),
            "tokenization": tokenization,
            "annotations": vr
        })
    return concrete_doc(inputs, doc_name)


def collect_concrete_srl(comm: Communication) -> List[Tuple[List[str], Span]]:
    # Mapping from <sentence uuid> to [<ConcreteSentence>, <Associated situation mentions>]
    sentences = defaultdict(lambda: [None, list()])
    for sec in comm.sectionList:
        for sen in sec.sentenceList:
            sentences[sen.uuid.uuidString][0] = sen
    # Assume there's only ONE situation mention set
    assert len(comm.situationMentionSetList) == 1
    # Assign each situation mention to the corresponding sentence
    for men in comm.situationMentionSetList[0].mentionList:
        if men.tokens is None: continue  # For ACE relations
        sentences[men.tokens.tokenization.sentence.uuid.uuidString][1].append(men)
    ret = list()
    for sen, mention_list in sentences.values():
        tokens = [t.text for t in sen.tokenization.tokenList.tokenList]
        spans = list()
        for mention in mention_list:
            mention_tokens = sorted(mention.tokens.tokenIndexList)
            event = Span(mention_tokens[0], mention_tokens[-1], mention.situationKind, True)
            for men_arg in mention.argumentList:
                arg_tokens = sorted(men_arg.entityMention.tokens.tokenIndexList)
                event.add_child(Span(arg_tokens[0], arg_tokens[-1], men_arg.role, False))
            spans.append(event)
        vr = Span.virtual_root(spans)
        ret.append((tokens, vr))
    return ret
