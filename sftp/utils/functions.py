from typing import *

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn.utils.rnn import pad_sequence


def num2mask(
        nums: torch.Tensor,
        max_length: Optional[int] = None
) -> torch.Tensor:
    """
    E.g. input a tensor [2, 3, 4], return [[T T F F], [T T T F], [T T T T]]
    :param nums: Shape [batch]
    :param max_length: maximum length. if not provided, will choose the largest number from nums.
    :return: 2D binary mask.
    """
    shape_backup = nums.shape
    nums = nums.flatten()
    max_length = max_length or int(nums.max())
    batch_size = len(nums)
    range_nums = torch.arange(0, max_length, device=nums.device).unsqueeze(0).expand([batch_size, max_length])
    ret = (range_nums.T < nums).T
    return ret.reshape(*shape_backup, max_length)


def mask2idx(
        mask: torch.Tensor,
        max_length: Optional[int] = None,
        padding_value: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    E.g. input a tensor [[T T F F], [T T T F], [F F F T]] with padding value -1,
    return [[0, 1, -1], [0, 1, 2], [3, -1, -1]]
    :param mask: Mask tensor. Boolean. Not necessarily to be 2D.
    :param max_length: If provided, will truncate.
    :param padding_value: Padding value. Default to 0.
    :return: Index tensor.
    """
    shape_prefix, mask_length = mask.shape[:-1], mask.shape[-1]
    flat_mask = mask.flatten(0, -2)
    index_list = [torch.arange(mask_length, device=mask.device)[one_mask] for one_mask in flat_mask.unbind(0)]
    index_tensor = pad_sequence(index_list, batch_first=True, padding_value=padding_value)
    if max_length is not None:
        index_tensor = index_tensor[:, :max_length]
    index_tensor = index_tensor.reshape(*shape_prefix, -1)
    return index_tensor, mask.sum(-1)


def one_hot(tags: torch.Tensor, num_tags: Optional[int] = None) -> torch.Tensor:
    num_tags = num_tags or int(tags.max())
    ret = tags.new_zeros(size=[*tags.shape, num_tags], dtype=torch.bool)
    ret.scatter_(2, tags.unsqueeze(2), tags.new_ones([*tags.shape, 1], dtype=torch.bool))
    return ret


def numpy2torch(
        dict_obj: dict
) -> dict:
    """
    Convert list/np.ndarray data to torch.Tensor and add add a batch dim.
    """
    ret = dict()
    for k, v in dict_obj.items():
        if isinstance(v, list) or isinstance(v, np.ndarray):
            ret[k] = torch.tensor(v).unsqueeze(0)
        else:
            ret[k] = v
    return ret


def max_match(mat: np.ndarray):
    row_idx, col_idx = linear_sum_assignment(mat, True)
    return mat[row_idx, col_idx].sum()
