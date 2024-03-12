import torchsparse.nn as spnn
import mindspore.nn as nn
from torchsparse import PointTensor

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

    def construct(self, x):
        z = PointTensor(x.F, x.C.astype('float32'))
        x0 = initial_voxelize(z, self.pres, self.vres)
        out = self.net(x0)
        return out