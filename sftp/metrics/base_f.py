from abc import ABC
from typing import *

from allennlp.training.metrics import Metric


class BaseF(Metric, ABC):
    def __init__(self, prefix: str):
        self.tp = self.fp = self.fn = 0
        self.prefix = prefix

    def reset(self) -> None:
        self.tp = self.fp = self.fn = 0

    def get_metric(
            self, reset: bool
    ) -> Union[float, Tuple[float, ...], Dict[str, float], Dict[str, List[float]]]:
        precision = self.tp * 100 / (self.tp + self.fp) if self.tp > 0 else 0.
        recall = self.tp * 100 / (self.tp + self.fn) if self.tp > 0 else 0.
        rst = {
            f'{self.prefix}_p': precision,
            f'{self.prefix}_r': recall,
            f'{self.prefix}_f': 2 / (1 / precision + 1 / recall) if self.tp > 0 else 0.
        }
        if reset:
            self.reset()
        return rst
