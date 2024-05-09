
import mindspore.nn as nn
from mindsparse import SparseTensor
from mindsparse.nn.utils import fapply

__all__ = ['ReLU', 'LeakyReLU']


class ReLU(nn.ReLU):
    def construct(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().construct)



class LeakyReLU(nn.LeakyReLU):
    def construct(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().construct)
