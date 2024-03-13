import os, sys
import numpy as np
import mindspore as ms
from mindspore import context
from core.models.semantic_kitti.spvcnn_ms import SPVCNN_MS

if __name__ == '__main__':
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    spvcnn = SPVCNN_MS(64, 64)
    out = spvcnn(0)
    