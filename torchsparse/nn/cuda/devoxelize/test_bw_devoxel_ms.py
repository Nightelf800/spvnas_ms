import torch
import numpy as np
import mindspore as ms
from mindspore.nn import Cell
import mindspore.ops as ops
from mindspore import context


class SPDevoxelize(Cell):
    def __init__(self, ):
        super(SPDevoxelize, self).__init__()

        def infer_func(a, b, c):
            return a

        # spvoxelize_back = ops.Custom("./voxelize_cuda.cu:voxelize_backward_ms",
        #                     infer_func,
        #                     infer_func,
        #                     func_type="aot")
        # def bprop(top_grad, idx, counts, N):
        #     return spvoxelize_back(top_grad, idx, counts, N)

        # self.spvoxelize = ops.Custom("./voxelize_cuda.cu:voxelize_forward_ms",
        #                     infer_func,
        #                     infer_func,
        #                     func_type="aot",
        #                     bprop=bprop)
        self.spdevoxelize = ops.Custom("./devoxelize_cuda.so:devoxelize_backward_ms",
                                       out_shape=infer_func,
                                       out_dtype=infer_func,
                                       func_type="aot")

    def construct(self, grad_output, coords, weights, input_size):
        return self.spdevoxelize(grad_output, coords, weights, input_size)


if __name__ == '__main__':
    context.set_context(device_target='GPU')

    sample = np.load("/home/ubuntu/hdd1/ylc/codes/torchsparse-1.4.0/examples/devoxelize_backward_sample.npz")

    print("grad_output.type: ", sample["grad_output"].dtype)
    print("coords.type: ", sample["coords"].dtype)
    print("weights.type: ", sample["weights"].dtype)
    print("input_size.type: ", sample["input_size"].dtype)
    print("grad_feats.type: ", sample["grad_feats"].dtype)

    grad_output = ms.Tensor(sample["grad_output"], dtype=ms.float32)
    coords = ms.Tensor(sample["coords"], dtype=ms.float32)
    weights = ms.Tensor(sample["weights"], dtype=ms.float32)
    input_size = sample["input_size"].itme()
    grad_feats = ms.Tensor(sample["grad_feats"], dtype=ms.float32)

    print(f"grad_output.shape:{grad_output.shape}")
    print(f"coords.shape:{coords.shape}")
    print(f"weights.shape:{weights.shape}")
    print(f"grad_feats.shape:{grad_feats.shape}")

    test_devoxelize = SPDevoxelize()
    ms_result = test_devoxelize(grad_output, coords, weights, input_size)

    print(f"ms_result.shape:{ms_result.shape}")
    print(f"ms_result - new_feat:{ms_result - new_feat}")
