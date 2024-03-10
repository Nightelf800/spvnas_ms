from mindspore.nn import Cell
import mindspore.ops as ops
from mindspore import context

class SPVoxelize(Cell):
    def __init__(self,):
        super(SPVoxelize, self).__init__()
    
        def infer_func(a, b, c):
            return a
        
        # def back_func(a, b, c, d):
        #     return a
        
        # spvoxelize_back = ops.Custom("./voxelize_cuda.so:voxelize_backward_ms",
        #                     back_func,
        #                     back_func,
        #                     func_type="aot")
        # def bprop(a, b, c, out, dout):
        #     dx = spvoxelize_back()
        #     return dx
        
        self.spvoxelize = ops.Custom("./voxelize_cuda.so:voxelize_forward_ms",
                            infer_func,
                            infer_func,
                            func_type="aot",
                            # bprop=bprop
                        )
    
    def construct(self, inputs, idx, counts):
        return self.spvoxelize(inputs, idx, counts)

# if __name__ == '__main__':
    # context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')

    # --------------------------test voxelize----------------------------
    # sample = np.load("/home/ubuntu/hdd1/mqh/test_custom_pytorch/sample.npz")
    # floor_new_float_coord = ms.Tensor(sample["floor_new_float_coord"], dtype=ms.float32)
    # idx_query = ms.Tensor(sample["idx_query"], dtype=ms.int32)
    # counts = ms.Tensor(sample["counts"], dtype=ms.int32)
    # inserted_coords = ms.Tensor(sample["inserted_coords"], dtype=ms.float32)
    # print(f"floor_new_float_coord.shape:{floor_new_float_coord.shape}")
    # print(f"idx_query.shape:{idx_query.shape}")
    # print(f"counts.shape:{counts.shape}")
    # print(f"inserted_coords.shape:{inserted_coords.shape}")
    
    # test_voxelize = SPVoxelize()
    # ms_inserted_coords = test_voxelize(floor_new_float_coord, idx_query, counts)
    # print(f"ms_inserted_coords.shape:{ms_inserted_coords.shape}")
    # print(f"ms_inserted_coords-torch_inserted_coords:{ms_inserted_coords-inserted_coords}")
    # print(f"ops.unique(ms_inserted_coords-torch_inserted_coords):{ops.unique(ms_inserted_coords-inserted_coords)[0]}")