from itertools import repeat
from typing import List, Tuple, Union

# import torch
import mindspore as ms

__all__ = ['make_ntuple']


def make_ntuple(x: Union[int, List[int], Tuple[int, ...], ms.Tensor],
                ndim: int) -> Tuple[int, ...]:
    if isinstance(x, int):
        x = tuple(repeat(x, ndim))
    elif isinstance(x, list):
        x = tuple(x)
    elif isinstance(x, ms.Tensor):
        x = tuple(x.view(-1).asnumpy().tolist())

    assert isinstance(x, tuple) and len(x) == ndim, x
    return x
