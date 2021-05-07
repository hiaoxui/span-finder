from allennlp.training.metrics import FBetaMeasure, Metric


@Metric.register('fbeta_mix')
class FBetaMixMeasure(FBetaMeasure):
    def __init__(self, null_idx, **kwargs):
        super().__init__(**kwargs)
        self.null_idx = null_idx

    def get_metric(self, reset: bool = False):

        tp = float(self._true_positive_sum.sum() - self._true_positive_sum[self.null_idx])
        total_pred = float(self._pred_sum.sum() - self._pred_sum[self.null_idx])
        total_gold = float(self._true_sum.sum() - self._true_sum[self.null_idx])

        beta2 = self._beta ** 2
        p = 0. if total_pred == 0 else tp / total_pred
        r = 0. if total_pred == 0 else tp / total_gold
        f = 0. if p == 0. or r == 0. else ((1 + beta2) * p * r / (p * beta2 + r))

        mix_f = {
            'p': p * 100,
            'r': r * 100,
            'f': f * 100
        }

        if reset:
            self.reset()

        return mix_f

    def add_false_negative(self, labels):
        for lab in labels:
            self._true_sum[lab] += 1
