from typing import Any, Dict, Tuple, Union

from mindspore import Tensor

from torchsparse_ms.utils import make_ntuple

__all__ = ['SparseTensor', 'PointTensor']


class SparseTensor:

    def __init__(self,
                 feats: Tensor,
                 coords: Tensor,
                 stride: Union[int, Tuple[int, ...]] = 1) -> None:
        self.feats = feats
        self.coords = coords
        self.stride = make_ntuple(stride, ndim=3)
        self.cmaps: Dict[Tuple[int, ...], Tensor] = {}
        self.kmaps: Dict[Tuple[Any, ...], Any] = {}

    @property
    def F(self) -> Tensor:
        return self.feats

    @F.setter
    def F(self, feats: Tensor) -> None:
        self.feats = feats

    @property
    def C(self) -> Tensor:
        return self.coords

    @C.setter
    def C(self, coords: Tensor) -> None:
        self.coords = coords

    @property
    def s(self) -> Tuple[int, ...]:
        return self.stride

    @s.setter
    def s(self, stride: Union[int, Tuple[int, ...]]) -> None:
        self.stride = make_ntuple(stride, ndim=3)

    def cuda(self):
        self.feats = self.feats.cuda()
        self.coords = self.coords.cuda()
        return self

    def detach(self):
        self.feats = self.feats.detach()
        self.coords = self.coords.detach()
        return self

    def to(self, device: str, non_blocking: bool = True):
        self.feats = self.feats.to(device, non_blocking=non_blocking)
        self.coords = self.coords.to(device, non_blocking=non_blocking)
        return self

    def __add__(self, other):
        output = SparseTensor(coords=self.coords,
                              feats=self.feats + other.feats,
                              stride=self.stride)
        output.cmaps = self.cmaps
        output.kmaps = self.kmaps
        return output


class PointTensor:

    def __init__(self, feats, coords, idx_query=None, weights=None):
        self.F = feats
        self.C = coords
        self.idx_query = idx_query if idx_query is not None else {}
        self.weights = weights if weights is not None else {}
        self.additional_features = {}
        self.additional_features['idx_query'] = {}
        self.additional_features['counts'] = {}

    def cuda(self):
        self.F = self.F.cuda()
        self.C = self.C.cuda()
        return self

    def detach(self):
        self.F = self.F.detach()
        self.C = self.C.detach()
        return self

    def to(self, device, non_blocking=True):
        self.F = self.F.to(device, non_blocking=non_blocking)
        self.C = self.C.to(device, non_blocking=non_blocking)
        return self

    def __add__(self, other):
        tensor = PointTensor(self.F + other.F, self.C, self.idx_query,
                             self.weights)
        tensor.additional_features = self.additional_features
        return tensor
