from typing import *

from allennlp.training.metrics import Metric
import numpy as np
import logging

from .base_f import BaseF
from ..utils import Span, max_match

logger = logging.getLogger('srl_metric')


@Metric.register('srl')
class SRLMetric(Metric):
    def __init__(self, check_type: Optional[bool] = None):
        self.tri_i = BaseF('tri-i')
        self.tri_c = BaseF('tri-c')
        self.arg_i = BaseF('arg-i')
        self.arg_c = BaseF('arg-c')
        if check_type is not None:
            logger.warning('Check type argument is deprecated.')

    def reset(self) -> None:
        for metric in [self.tri_i, self.tri_c, self.arg_i, self.arg_c]:
            metric.reset()

    def get_metric(self, reset: bool) -> Dict[str, Any]:
        ret = dict()
        for metric in [self.tri_i, self.tri_c, self.arg_i, self.arg_c]:
            ret.update(metric.get_metric(reset))
        return ret

    def __call__(self, prediction: Span, gold: Span):
        self.with_label_event(prediction, gold)
        self.without_label_event(prediction, gold)
        self.tuple_eval(prediction, gold)
        # self.with_label_arg(prediction, gold)
        # self.without_label_arg(prediction, gold)

    def tuple_eval(self, prediction: Span, gold: Span):
        def extract_tuples(vr: Span, parent_boundary: bool):
            labeled, unlabeled = list(), list()
            for event in vr:
                for arg in event:
                    if parent_boundary:
                        labeled.append((event.boundary, event.label, arg.boundary, arg.label))
                        unlabeled.append((event.boundary, event.label, arg.boundary))
                    else:
                        labeled.append((event.label, arg.boundary, arg.label))
                        unlabeled.append((event.label, arg.boundary))
            return labeled, unlabeled

        def equal_matrix(l1, l2): return np.array([[e1 == e2 for e2 in l2] for e1 in l1], dtype=np.int64)

        pred_label, pred_unlabel = extract_tuples(prediction, False)
        gold_label, gold_unlabel = extract_tuples(gold, False)

        if len(pred_label) == 0 or len(gold_label) == 0:
            arg_c_tp = arg_i_tp = 0
        else:
            label_bipartite = equal_matrix(pred_label, gold_label)
            unlabel_bipartite = equal_matrix(pred_unlabel, gold_unlabel)
            arg_c_tp, arg_i_tp = max_match(label_bipartite), max_match(unlabel_bipartite)

        arg_c_fp = prediction.n_nodes - len(prediction) - 1 - arg_c_tp
        arg_c_fn = gold.n_nodes - len(gold) - 1 - arg_c_tp
        arg_i_fp = prediction.n_nodes - len(prediction) - 1 - arg_i_tp
        arg_i_fn = gold.n_nodes - len(gold) - 1 - arg_i_tp

        assert arg_i_tp >= 0 and arg_i_fn >= 0 and arg_i_fp >= 0
        self.arg_i.tp += arg_i_tp
        self.arg_i.fp += arg_i_fp
        self.arg_i.fn += arg_i_fn

        assert arg_c_tp >= 0 and arg_c_fn >= 0 and arg_c_fp >= 0
        self.arg_c.tp += arg_c_tp
        self.arg_c.fp += arg_c_fp
        self.arg_c.fn += arg_c_fn

    def with_label_event(self, prediction: Span, gold: Span):
        trigger_tp = prediction.match(gold, True, 2) - 1
        trigger_fp = len(prediction) - trigger_tp
        trigger_fn = len(gold) - trigger_tp
        assert trigger_fp >= 0 and trigger_fn >= 0 and trigger_tp >= 0
        self.tri_c.tp += trigger_tp
        self.tri_c.fp += trigger_fp
        self.tri_c.fn += trigger_fn

    def with_label_arg(self, prediction: Span, gold: Span):
        trigger_tp = prediction.match(gold, True, 2) - 1
        role_tp = prediction.match(gold, True, ignore_parent_boundary=True) - 1 - trigger_tp
        role_fp = (prediction.n_nodes - 1 - len(prediction)) - role_tp
        role_fn = (gold.n_nodes - 1 - len(gold)) - role_tp
        assert role_fp >= 0 and role_fn >= 0 and role_tp >= 0
        self.arg_c.tp += role_tp
        self.arg_c.fp += role_fp
        self.arg_c.fn += role_fn

    def without_label_event(self, prediction: Span, gold: Span):
        tri_i_tp = prediction.match(gold, False, 2) - 1
        tri_i_fp = len(prediction) - tri_i_tp
        tri_i_fn = len(gold) - tri_i_tp
        assert tri_i_tp >= 0 and tri_i_fp >= 0 and tri_i_fn >= 0
        self.tri_i.tp += tri_i_tp
        self.tri_i.fp += tri_i_fp
        self.tri_i.fn += tri_i_fn

    def without_label_arg(self, prediction: Span, gold: Span):
        arg_i_tp = 0
        matched_pairs: List[Tuple[Span, Span]] = list()
        n_gold_arg, n_pred_arg = gold.n_nodes - len(gold) - 1, prediction.n_nodes - len(prediction) - 1
        prediction, gold = prediction.clone(), gold.clone()
        for p in prediction:
            for g in gold:
                if p.match(g, True, 1) == 1:
                    arg_i_tp += (p.match(g, False) - 1)
                    matched_pairs.append((p, g))
                    break
        for p, g in matched_pairs:
            prediction.remove_child(p)
            gold.remove_child(g)

        sub_matches = np.zeros([len(prediction), len(gold)], np.int64)
        for p_idx, p in enumerate(prediction):
            for g_idx, g in enumerate(gold):
                if p.label == g.label:
                    sub_matches[p_idx, g_idx] = p.match(g, False, -1, True)
        arg_i_tp += max_match(sub_matches)

        arg_i_fp = n_pred_arg - arg_i_tp
        arg_i_fn = n_gold_arg - arg_i_tp
        assert arg_i_tp >= 0 and arg_i_fn >= 0 and arg_i_fp >= 0

        self.arg_i.tp += arg_i_tp
        self.arg_i.fp += arg_i_fp
        self.arg_i.fn += arg_i_fn
