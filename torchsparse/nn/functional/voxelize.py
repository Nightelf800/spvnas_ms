from typing import Optional, Tuple, Union
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

from torchsparse.nn.cuda.voxelize import SPVoxelize

__all__ = ['spvoxelize']


class VoxelizeFunction(nn.Cell):
    def __init__(self):
        super(VoxelizeFunction, self).__init__()
        self.sp_voxelize = SPVoxelize()

    def construct(self,
                  feats: Tensor,
                  coords: Tensor,
                  counts: Tensor) -> Tensor:

        output = self.sp_voxelize(coords, counts, feats.shape[0])

        return output

    # def bprop(self, input, weight, nbmaps, nbsizes, sizes, transposed,
    #             output, grad_output):

    #     grad_input = ops.ZerosLike(input)
    #     grad_weight = ops.ZerosLike(weight)

    #     if grad_output.device.type == 'cuda':
    #         grad_input, grad_weight = self.sp_conv(
    #             input, grad_input, grad_output.contiguous(), weight,
    #             grad_weight, nbmaps, nbsizes.cpu(), transposed)
    #     else:
    #         raise NotImplementedError
    #     return grad_input, grad_weight, None, None, None, None


def spvoxelize(feats: Tensor, coords: Tensor,
               counts: Tensor) -> Tensor:
    return VoxelizeFunction.apply(feats, coords, counts)
