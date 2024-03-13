import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

from torchsparse.nn.cuda.voxelize import SPVoxelizeForward, SPVoxelizeBackward

__all__ = ['spvoxelize']


class VoxelizeFunction(nn.Cell):
    def __init__(self):
        super(VoxelizeFunction, self).__init__()
        self.sp_voxelize_forward = SPVoxelizeForward()
        self.sp_voxelize_backward = SPVoxelizeBackward()

    def construct(self,
                  feats: Tensor,
                  coords: Tensor,
                  counts: Tensor) -> Tensor:

        output = self.sp_voxelize_forward(feats, coords, counts)

        return output

    def bprop(self, feats, coords, counts, output, grad_output):
        input_size = ops.Zeros()((feats.shape[0]), ms.int32)
        grad_feats = self.sp_voxelize_backward(
            grad_output, coords, counts, input_size)

        return (grad_feats, )


def spvoxelize(feats: Tensor, coords: Tensor,
               counts: Tensor) -> Tensor:
    return VoxelizeFunction()(feats, coords, counts)
