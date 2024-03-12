from typing import List
import mindspore.ops as ops
from torchsparse.tensor import SparseTensor

__all__ = ['cat']


def cat(inputs: List[SparseTensor]) -> SparseTensor:
    feats = ops.Concat(axis=1)([input.feats for input in inputs])
    output = SparseTensor(coords=inputs[0].coords,
                          feats=feats,
                          stride=inputs[0].stride)
    output.cmaps = inputs[0].cmaps
    output.kmaps = inputs[0].kmaps
    return output
