import torchsparse.nn as spnn
import mindspore.nn as nn
from torchsparse import PointTensor

from core.models.utils import initial_voxelize, point_to_voxel, voxel_to_point

__all__ = ['SPVCNN_MS']


class SPVCNN_MS(nn.Cell):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
        )

    def construct(self, x):
        out = self.net(x)
        return out