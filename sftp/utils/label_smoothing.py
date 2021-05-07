import torch
from torch import nn
from torch.nn import KLDivLoss
from torch.nn import LogSoftmax


class LabelSmoothingLoss(nn.Module):
    def __init__(self, label_smoothing=0.0, unreliable_label=None, ignore_index=-100):
        """
        If label_smoothing == 0.0, it is equivalent to xentropy
        """
        assert 0.0 <= label_smoothing <= 1.0
        super(LabelSmoothingLoss, self).__init__()

        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

        self.loss_fn = KLDivLoss(reduction='batchmean')
        self.unreliable_label = unreliable_label
        self.max_gap = 100.
        self.log_softmax = LogSoftmax(1)

    def forward(self, output, target):
        """
        output: logits
        target: labels
        """
        vocab_size = output.shape[1]
        mask = (target != self.ignore_index)
        output, target = output[mask], target[mask]
        output = self.log_softmax(output)

        def get_smooth_prob(ls):
            smoothing_value = ls / (vocab_size - 1)
            prob = output.new_full((target.size(0), vocab_size), smoothing_value)
            prob.scatter_(1, target.unsqueeze(1), 1 - ls)
            return prob

        if self.unreliable_label is not None:
            smoothed_prob = get_smooth_prob(self.label_smoothing)
            hard_prob = get_smooth_prob(0.0)
            unreliable_mask = (target == self.unreliable_label).to(torch.float)
            model_prob = ((smoothed_prob.T * unreliable_mask) + (hard_prob.T * (1 - unreliable_mask))).T
        else:
            model_prob = get_smooth_prob(self.label_smoothing)

        loss = self.loss_fn(output, model_prob)
        return loss
