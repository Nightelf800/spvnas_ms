import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import ops
from torchsparse import nn as spnn
from torchsparse import PointTensor, SparseTensor

from core.models.utils import initial_voxelize, point_to_voxel, voxel_to_point

__all__ = ['SPVCNN_MS']


class SPVCNN_MS(nn.Cell):
    def __init__(self, **kwargs):
        super().__init__()

        cr = kwargs.get('cr', 1.0)
        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs]

        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']

        self.net = nn.SequentialCell(
            spnn.Conv3d(4, cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]),
            spnn.ReLU())

        self.classifier = nn.SequentialCell([nn.Dense(cs[0], kwargs['num_classes'])])

    def construct(self, x):
        z = PointTensor(x.F, x.C.astype('float32'))
        print(f"before initial_voxelize")
        x0 = initial_voxelize(z, self.pres, self.vres)
        print(f"iniial_voxelize success")
        # exit()
        z0 = self.net(x0)
        print(f"conv3d success")
        out = self.classifier(z0.F)
        return out