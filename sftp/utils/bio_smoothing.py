from typing import *

import numpy as np
from .common import BIO


class BIOSmoothing:
    def __init__(
            self,
            b_smooth: float = 0.0,
            i_smooth: float = 0.0,
            o_smooth: float = 0.0,
            weight: float = 1.0
    ):
        self.smooth = [b_smooth, i_smooth, o_smooth]
        self.weight = weight

    def apply_sequence(self, sequence: List[str]):
        bio_tags = np.zeros([len(sequence), 3], np.float32)
        for i, tag in enumerate(sequence):
            bio_tags[i] = self.apply_tag(tag)
        return bio_tags

    def apply_tag(self, tag: str):
        j = BIO.index(tag)
        ret = np.zeros([3], np.float32)
        if self.smooth[j] >= 0.0:
            # Smooth
            ret[j] = 1.0 - self.smooth[j]
            for j_ in set(range(3)) - {j}:
                ret[j_] = self.smooth[j] / 2
        else:
            # Marginalize
            ret[:] = 1.0

        return ret * self.weight

    def __repr__(self):
        ret = f'<W={self.weight:.2f}'
        for j, tag in enumerate(BIO):
            if self.smooth[j] != 0.0:
                if self.smooth[j] < 0:
                    ret += f' [marginalize {tag}]'
                else:
                    ret += f' [smooth {tag} by {self.smooth[j]:.2f}]'
        return ret + '>'

    def clone(self):
        return BIOSmoothing(*self.smooth, self.weight)


def apply_bio_smoothing(
        config: Optional[Union[BIOSmoothing, List[BIOSmoothing]]],
        bio_seq: List[str]
) -> np.ndarray:
    if config is None:
        config = BIOSmoothing()
    if isinstance(config, BIOSmoothing):
        return config.apply_sequence(bio_seq)
    else:
        assert len(bio_seq) == len(config)
        return np.stack([cfg.apply_tag(tag) for cfg, tag in zip(config, bio_seq)])
