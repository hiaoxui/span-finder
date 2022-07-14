from allennlp.training.metrics import Metric

from .base_f import BaseF
from ..utils import Span


@Metric.register('exact_match')
class ExactMatch(BaseF):
    def __init__(self, check_type: bool):
        self.check_type = check_type
        if check_type:
            super(ExactMatch, self).__init__('em')
        else:
            super(ExactMatch, self).__init__('sm')

    def __call__(
            self,
            prediction: Span,
            gold: Span,
    ):
        tp = prediction.match(gold, self.check_type) - 1
        fp = prediction.n_nodes - tp - 1
        fn = gold.n_nodes - tp - 1
        assert tp >= 0 and fp >= 0 and fn >= 0
        self.tp += tp
        self.fp += fp
        self.fn += fn
