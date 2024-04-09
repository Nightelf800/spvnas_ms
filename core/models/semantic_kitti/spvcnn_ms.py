import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import ops
from torchsparse import nn as spnn
from torchsparse import PointTensor, SparseTensor

from core.models.utils import initial_voxelize, point_to_voxel, voxel_to_point

__all__ = ['SPVCNN_MS']

def save_ouptut_data(name, output):
    print(f"save {name} data: ")
    np.savez(f'./{name}.npz', output=output.asnumpy())
    print("save successfully")

def compare_output_data(name, output, dtype):
    sample = np.load(f"./{name}.npz")
    print("sample.shape: ", sample["output"].shape, "input.dtype: ", sample["output"].dtype)
    output_ori = ms.Tensor(sample["output"], dtype=dtype)
    print(f"compare {name} data: ")
    print(f"output-output_ori: {ops.unique(output - output_ori)[0]}")


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
        print(f"net.input.data: {x.F}")
        print(f"net.input.data.shape: {x.F.shape}, net.input.data.dtype: {x.F.dtype}")


        z = PointTensor(x.F, x.C.astype('float32'))

        print(f"net.input.pointtensor: {z.F}")
        print(f"net.input.pointtensor.shape: {z.F.shape}")

        print(f"before initial_voxelize")
        x0 = initial_voxelize(z, self.pres, self.vres)
        print(f"iniial_voxelize success")

        print(f"net.voxelize: {x0.F}")
        print(f"net.voxelize.shape: {x0.F.shape}")

        z0 = self.net(x0)

        print(f"net.conv3d: {z0.F}")
        print(f"net.conv3d.shape: {z0.F.shape}")
        print(f"conv3d success")

        out = self.classifier(z0.F)
        return out