import torch
import numpy as np
import mindspore as ms
from mindspore.nn import Cell
import mindspore.ops as ops
from mindspore import context


class SPConvolutionForward(Cell):
    def __init__(self, ):
        super(SPConvolutionForward, self).__init__()

        def infer_func(a, b, c, d, e, f):
            return b

        # spvoxelize_back = ops.Custom("./voxelize_cuda.cu:voxelize_backward_ms",
        #                     infer_func,
        #                     infer_func,
        #                     func_type="aot")
        # def bprop(top_grad, idx, counts, N):
        #     return spvoxelize_back(top_grad, idx, counts, N)
        #
        # self.spvoxelize = ops.Custom("./voxelize_cuda.cu:voxelize_forward_ms",
        #                     infer_func,
        #                     infer_func,
        #                     func_type="aot",
        #                     bprop=bprop)
        self.spconvolution_transpose = ops.Custom(
            "./torchsparse/nn/cuda/convolution/convolution_cuda.so:convolution_transpose_forward_ms",
            out_shape=infer_func,
            out_dtype=infer_func,
            func_type="aot")

        self.spconvolution_no_transpose = ops.Custom(
            "./torchsparse/nn/cuda/convolution/convolution_cuda.so:convolution_no_transpose_forward_ms",
            out_shape=infer_func,
            out_dtype=infer_func,
            func_type="aot")

    def construct(self, in_feat, out_feat, kernel, neighbor_map, neighbor_offset, transpose):
        if transpose:
            return self.spconvolution_transpose(in_feat, out_feat, kernel, neighbor_map, neighbor_offset, transpose)
        else:
            return self.spconvolution_no_transpose(in_feat, out_feat, kernel, neighbor_map, neighbor_offset, transpose)

class SPConvolutionBackward(Cell):
    def __init__(self, ):
        super(SPConvolutionBackward, self).__init__()

        def infer_func(a, b, c, d, e, f, g, h):
            return b, e

        self.spconvolution_transpose = ops.Custom(
            "./torchsparse/nn/cuda/convolution/convolution_cuda.so:convolution_transpose_backward_ms",
            out_shape=infer_func,
            out_dtype=infer_func,
            func_type="aot")

        self.spconvolution_no_transpose = ops.Custom(
            "./torchsparse/nn/cuda/convolution/convolution_cuda.so:convolution_no_transpose_backward_ms",
            out_shape=infer_func,
            out_dtype=infer_func,
            func_type="aot")

    def construct(self, input, grad_input, grad_output, weight, grad_weight, nbmaps, nbsizes, transposed):
        if transposed:
            return self.spconvolution_transpose(input, grad_input, grad_output, weight, grad_weight,
                                                nbmaps, nbsizes, transposed)
        else:
            return self.spconvolution_no_transpose(input, grad_input, grad_output, weight, grad_weight,
                                                   nbmaps, nbsizes, transposed)


if __name__ == '__main__':
    context.set_context(device_target='GPU')
    sample = np.load("conv_forward_sample.npz")
    
    print("input.type: ", sample["input"].dtype)
    print("output.type: ", sample["output"].dtype)
    print("weight.type: ", sample["weight"].dtype)
    print("nbmaps.type: ", sample["nbmaps"].dtype)
    print("nbsizes.type: ", sample["nbsizes"].dtype)
    print("transposed.type: ", sample["transposed"].dtype)
    print("result.type: ", sample["result"].dtype)
    
    
    result = ms.Tensor(sample["result"], dtype=ms.float32)
    input = ms.Tensor(sample["input"], dtype=ms.float32)
    output = ms.Tensor(sample["output"], dtype=ms.float32)
    weight = ms.Tensor(sample["weight"], dtype=ms.float32)
    nbmaps = ms.Tensor(sample["nbmaps"], dtype=ms.int32)
    nbsizes = ms.Tensor(sample["nbsizes"], dtype=ms.int32)
    transposed = sample["transposed"].item()

    print(f"input.shape:{input.shape}")
    print(f"output.shape:{output.shape}")
    print(f"weight.shape:{weight.shape}")
    print(f"nbmaps.shape:{nbmaps.shape}")
    print(f"nbsizes.shape:{nbsizes.shape}")
    print(f"transposed:{transposed}")
    print(f"result.shape:{result.shape}")

    print("------------test convolution--------------")
    test_convolution = SPConvolution()
    ms_result = test_convolution(input, output, weight, nbmaps, nbsizes, transposed)

    print(f"ms_result.shape:{ms_result.shape}")
    print(f"torch_result - ms_result:{result - ms_result}")
    print(f"ops.unqiue(torch_result - ms_result):{ops.unique(result - ms_result)}")