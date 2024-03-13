import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import ops
from torchsparse import nn as spnn
from torchsparse import PointTensor, SparseTensor

from core.models.utils import initial_voxelize, point_to_voxel, voxel_to_point

__all__ = ['SPVCNN_MS']


class SPVCNN_MS(nn.Cell):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        # self.net = nn.SequentialCell(
        #     spnn.Conv3d(inc,
        #                 outc,
        #                 kernel_size=ks,
        #                 dilation=dilation,
        #                 stride=stride),
        #     spnn.BatchNorm(outc),
        #     spnn.ReLU(True),
        # )
        self.pres = 0.05
        self.vres = 0.05


    def construct(self, x):
        # x.SparseTensor z: PointTensor
        sample = np.load("/home/ubuntu/hdd1/mqh/test_custom_pytorch/spvcnn_sample.npz")
        xf = ms.Tensor(sample['xf'], dtype=ms.float32)
        xc = ms.Tensor(sample['xc'], dtype=ms.int32)
        torch_x0f = ms.Tensor(sample['x0f'], dtype=ms.float32)
        torch_x0c = ms.Tensor(sample['x0c'], dtype=ms.int32)
        x = SparseTensor(coords=xc, feats=xf)

        z = PointTensor(x.F, x.C.astype(ms.float32))

        x0 = initial_voxelize(z, self.pres, self.vres)
        print(f"x0.F.shape:{x0.F.shape}, x0.F.dtype:{x.F.dtype}")
        print(f"x0.C.shape:{x0.C.shape}, x0.C.dtype:{x.C.dtype}")
        print(f"ops.unique(x0.F-torch_x0f):{ops.unique(x0.F-torch_x0f)[0]}")
        print(f"ops.unique(x0.C-torch_x0c):{ops.unique(x0.C-torch_x0c)[0]}")
        exit()

        out = self.net(x)
        return out