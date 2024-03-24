import mindspore
import torch
import numpy as np

from mindspore import Tensor
import mindspore.ops as ops

if __name__ == '__main__':

    print("------------torch--------------")
    input_torch = torch.tensor([[-1, 2], [1, 5], [3, -1]])
    print("input_torch_shape: ", input_torch.shape)

    nbmaps = torch.nonzero(input_torch != -1)

    print("nbmaps: ", nbmaps)
    print("nbmaps_shape: ", nbmaps.shape)

    input_torch_view = input_torch.view(-1)
    print("input_torch_view: ", input_torch_view)
    print("input_torch_view_shape: ", input_torch_view.shape)

    input_torch_index = nbmaps[:, 0] * input_torch.size(1) + nbmaps[:, 1]
    print("input_torch_index: ", input_torch_index)
    print("input_torch_index_shape: ", input_torch_index.shape)


    nbmaps[:, 0] = input_torch_view[input_torch_index]

    print("result_torch: ", nbmaps)
    print("result_torch_shape: ", nbmaps.shape)
    print("-------------------------------")

    print("------------mindspore--------------")
    input_mindspore = Tensor(np.array([[-1, 2], [1, 5], [3, -1]]))
    print("input_mindspore: ", input_mindspore)
    print("input_mindspore_shape: ", input_mindspore.shape)

    nbmaps_mindspore = (input_mindspore != -1).nonzero()

    print("nbmaps_mindspore: ", nbmaps_mindspore)
    print("nbmaps_mindspore_shape: ", nbmaps_mindspore.shape)

    input_mindspore_view = input_mindspore.view(-1)
    print("input_mindspore_view: ", input_mindspore_view)
    print("input_mindspore_view_shape: ", input_mindspore_view.shape)

    input_mindspore_index_step1 = nbmaps_mindspore[:, 0]

    input_mindspore_index_step1_ = nbmaps_mindspore[:, 1]

    input_mindspore_index_step2 = input_mindspore_index_step1 * input_mindspore.shape[1] + \
                                  input_mindspore_index_step1_
    print("input_mindspore_index: ", input_mindspore_index_step2)
    print("input_mindspore_index_shape: ", input_mindspore_index_step2.shape)


    input_mindspore_view_resized = input_mindspore_view[input_mindspore_index_step2]

    nbmaps_index = ops.range(Tensor(0, mindspore.int32),
                             Tensor(input_mindspore_view_resized.shape[0], mindspore.int32),
                             Tensor(1, mindspore.int32))

    nbmaps_index = ops.expand_dims(nbmaps_index, 1)
    nbmaps_index_expandDim = ops.ZerosLike()(nbmaps_index)

    nbmaps_index_expand = ops.concat((nbmaps_index,
                                      nbmaps_index_expandDim),
                                     axis=1)
    print("input_mindspore_index_expand: ", nbmaps_index_expand)
    print("input_mindspore_index_expand_shape: ", nbmaps_index_expand.shape)

    nbmaps_mindspore = ops.TensorScatterUpdate()(nbmaps_mindspore,
                                                 nbmaps_index_expand,
                                                 input_mindspore_view_resized)

    print("result_mindspore: ", nbmaps_mindspore)
    print("result_mindspore_shape: ", nbmaps_mindspore.shape)
    print("-------------------------------")