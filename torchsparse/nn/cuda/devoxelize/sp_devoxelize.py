import torch
import numpy as np
import mindspore as ms
from mindspore.nn import Cell
import mindspore.ops as ops
from mindspore import context, get_context



def bprop():
    def infer_func_back(a, b, c, d):
        if isinstance(a, list):
            return [d[0], a[1]]
        else:
            return a

    sp_devoxelize_backward = ops.Custom(
        "./torchsparse/nn/cuda/devoxelize/devoxelize_cuda.so:devoxelize_backward_ms",
        out_shape=infer_func_back,
        out_dtype=infer_func_back,
        func_type="aot")
    def devoxelize_bprop(feat, indices, weight, out, grad_output):
        input_size = ops.Zeros()((feat.shape[0]), ms.int32)
        # grad_feats = sp_devoxelize_backward(
        #     grad_output, indices.astype(ms.int32), weight, feat.shape[0])
        grad_feats = sp_devoxelize_backward(
            grad_output, indices.astype(ms.int32), weight, input_size)
        return (grad_feats,)

    if get_context("device_target") == 'GPU':
        return devoxelize_bprop
    else:
        print('only support GPU')
        return None

class SPDevoxelizeForward(Cell):
    def __init__(self, ):
        super(SPDevoxelizeForward, self).__init__()

        def infer_func(a, b, c):
            if isinstance(a, list):
                return [b[0], a[1]]
            else:
                return a

        self.spdevoxelize = ops.Custom("./torchsparse/nn/cuda/devoxelize/devoxelize_cuda.so:devoxelize_forward_ms",
                                       out_shape=infer_func,
                                       out_dtype=infer_func,
                                       func_type="aot",
                                       bprop=bprop())
        # self.spdevoxelize = ops.Custom("./devoxelize_cuda.so:devoxelize_forward_ms",
        #                                out_shape=infer_func,
        #                                out_dtype=infer_func,
        #                                func_type="aot",
        #                                bprop=bprop())

    def construct(self, feat, indices, weight):
        return self.spdevoxelize(feat, indices, weight)

class SPDevoxelizeBackward(Cell):
    def __init__(self, ):
        super(SPDevoxelizeBackward, self).__init__()

        def infer_func(a, b, c):
            return a

        self.spdevoxelize = ops.Custom("./torchsparse/nn/cuda/devoxelize/devoxelize_cuda.so:devoxelize_backward_ms",
                                       out_shape=infer_func,
                                       out_dtype=infer_func,
                                       func_type="aot")

    def construct(self, grad_output, coords, weights, input_size):
        return self.spdevoxelize(grad_output, coords, weights, input_size)


if __name__ == '__main__':
    context.set_context(device_target='GPU')

    sample = np.load("/home/stf/workspace/codes/spvnas_ms/examples/devoxelize_forward_sample.npz")
    
    print("x.type: ", sample["x"].dtype)
    print("idx_query.type: ", sample["idx_query"].dtype)
    print("weights.type: ", sample["weights"].dtype)
    print("new_feat.type: ", sample["new_feat"].dtype)

    input = ms.Tensor(sample["x"], dtype=ms.float32)
    idx_query = ms.Tensor(sample["idx_query"], dtype=ms.int32)
    weights = ms.Tensor(sample["weights"], dtype=ms.float32)
    new_feat = ms.Tensor(sample["new_feat"], dtype=ms.float32)

    print(f"input.shape:{input.shape}")
    print(f"idx_query.shape:{idx_query.shape}")
    print(f"weights.shape:{weights.shape}")
    print(f"new_feat.shape:{new_feat.shape}")

    test_devoxelize = SPDevoxelizeForward()
    ms_result = test_devoxelize(input, idx_query, weights)

    print(f"ms_result.shape:{ms_result.shape}")
    print(f"ms_result - new_feat:{ms_result - new_feat}")
