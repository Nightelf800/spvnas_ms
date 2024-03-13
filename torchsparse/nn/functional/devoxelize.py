import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
from torchsparse.nn.cuda.devoxelize import SPDevoxelize
from mindspore import Tensor
import mindspore.nn as nn

__all__ = ['spdevoxelize', 'calc_ti_weights']


def calc_ti_weights(coords: torch.Tensor,
                    idx_query: torch.Tensor,
                    scale: float = 1) -> torch.Tensor:
    with torch.no_grad():
        p = coords
        if scale != 1:
            pf = torch.floor(coords / scale) * scale
        else:
            pf = torch.floor(coords)
        pc = pf + scale

        x = p[:, 0].view(-1, 1)
        y = p[:, 1].view(-1, 1)
        z = p[:, 2].view(-1, 1)

        xf = pf[:, 0].view(-1, 1).float()
        yf = pf[:, 1].view(-1, 1).float()
        zf = pf[:, 2].view(-1, 1).float()

        xc = pc[:, 0].view(-1, 1).float()
        yc = pc[:, 1].view(-1, 1).float()
        zc = pc[:, 2].view(-1, 1).float()

        w0 = (xc - x) * (yc - y) * (zc - z)
        w1 = (xc - x) * (yc - y) * (z - zf)
        w2 = (xc - x) * (y - yf) * (zc - z)
        w3 = (xc - x) * (y - yf) * (z - zf)
        w4 = (x - xf) * (yc - y) * (zc - z)
        w5 = (x - xf) * (yc - y) * (z - zf)
        w6 = (x - xf) * (y - yf) * (zc - z)
        w7 = (x - xf) * (y - yf) * (z - zf)

        w = torch.cat([w0, w1, w2, w3, w4, w5, w6, w7], dim=1)
        w = w.transpose(1, 0).contiguous()
        if scale != 1:
            w /= scale ** 3
        w[idx_query == -1] = 0
        w /= torch.sum(w, dim=0) + 1e-8
    return w


class DevoxelizeFunction(nn.Cell):
    def __init__(self):
        super(DevoxelizeFunction, self).__init__()
        self.sp_devoxelize = SPDevoxelize()

    def construct(self, feats: Tensor, coords: Tensor,
                weights: Tensor) -> Tensor:

        if feats.device.type == 'cuda':
            output = self.sp_devoxelize(
                feats, coords, weights)
        else:
            raise NotImplementedError

        return output


    def bprop(self, feats: Tensor, coords: Tensor,
                weights: Tensor, output: Tensor, grad_output: Tensor):

        if grad_output.device.type == 'cuda':
            grad_feats = self.sp_devoxelize(
                grad_output, coords, weights, feats.shape[0])
        else:
            raise NotImplementedError
        return grad_feats, None, None


def spdevoxelize(feats: Tensor, coords: Tensor,
                 weights: Tensor) -> Tensor:
    return DevoxelizeFunction.apply(feats, coords, weights)
