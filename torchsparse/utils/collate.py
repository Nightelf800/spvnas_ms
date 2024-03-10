from typing import Any, List

import numpy as np
# import torch
import mindspore as ms

from torchsparse import SparseTensor

__all__ = ['sparse_collate', 'sparse_collate_fn']


def sparse_collate(inputs: List[SparseTensor]) -> SparseTensor:
    coords, feats = [], []
    stride = inputs[0].stride

    for k, x in enumerate(inputs):
        if isinstance(x.coords, np.ndarray):
            x.coords = ms.Tensor(x.coords)
        if isinstance(x.feats, np.ndarray):
            x.feats = ms.Tensor(x.feats)

        assert isinstance(x.coords, ms.Tensor), type(x.coords)
        assert isinstance(x.feats, ms.Tensor), type(x.feats)
        assert x.stride == stride, (x.stride, stride)

        input_size = x.coords.shape[0]
        # batch = ms.ops.full((input_size, 1),
        #                     k,
        #                     dtype=ms.int32)
        batch = ms.numpy.full((input_size, 1),
                              k,
                              dtype=x.coords.dtype)

        coords.append(ms.ops.concat((x.coords, batch), axis=1))
        feats.append(x.feats)

    coords = ms.ops.concat(coords, axis=0)
    feats = ms.ops.concat(feats, axis=0)
    output = SparseTensor(coords=coords, feats=feats, stride=stride)
    return output


def sparse_collate_fn(inputs: List[Any]) -> Any:
    if isinstance(inputs[0], dict):
        output = {}
        for name in inputs[0].keys():
            print('name:', name)
            print('type:', type(inputs[0][name]))
            if isinstance(inputs[0][name], dict):
                output[name] = sparse_collate_fn(
                    [input[name] for input in inputs])
            elif isinstance(inputs[0][name], np.ndarray):
                output[name] = ms.ops.stack(
                    [ms.Tensor(input[name]) for input in inputs], axis=0)
            elif isinstance(inputs[0][name], ms.Tensor):
                output[name] = ms.ops.stack([input[name] for input in inputs], axis=0)
            elif isinstance(inputs[0][name], SparseTensor):
                output[name] = sparse_collate([input[name] for input in inputs])
            else:
                output[name] = [input[name] for input in inputs]
        return output
    else:
        return inputs
