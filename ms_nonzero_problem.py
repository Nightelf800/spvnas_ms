import mindspore
import numpy as np

from mindspore import Tensor
import mindspore.ops as ops

if __name__ == '__main__':

    print("------------mindspore right sample--------------")
    input_mindspore = Tensor(np.array([[-1, 2], [1, 5], [3, -1]]))
    print("input_mindspore: ", input_mindspore)
    print("input_mindspore_shape: ", input_mindspore.shape)

    input_mindspore_all_0 = input_mindspore[:, 0]
    print("input_mindspore[:, 0]: ", input_mindspore_all_0)
    input_mindspore_all_1 = input_mindspore[:, 1]
    print("input_mindspore[:, 1]: ", input_mindspore_all_1)

    print("input_mindspore[0][0]: ", input_mindspore[0][0])
    print("input_mindspore[0][1]: ", input_mindspore[0][1])
    print("input_mindspore[0:1, 0]: ", input_mindspore[0:1, 0])
    print("input_mindspore[0:1, 1]: ", input_mindspore[0:1, 1])
    print("input_mindspore[0:1, 0:2]: ", input_mindspore[0:1, 0:2])
    print("input_mindspore[0:2, 0]: ", input_mindspore[0:2, 0])
    print("input_mindspore[0:2, 1]: ", input_mindspore[0:2, 1])
    print("input_mindspore[0:2, 0:2]: ", input_mindspore[0:2, 0:2])
    print("input_mindspore[1:3, 0]: ", input_mindspore[1:3, 0])
    print("input_mindspore[1:3, 1]: ", input_mindspore[1:3, 1])
    print("input_mindspore[1:3, 0:2]: ", input_mindspore[1:3, 0:2])

    print("------------------------------------------------\n\n")

    print("------------mindspore .nonzero() problem sample--------------")

    nonzero_mindspore = (input_mindspore != -1).nonzero()
    print("(input_mindspore != -1).nonzero(): ", nonzero_mindspore)
    nonzero_mindspore = nonzero_mindspore.astype(mindspore.int32)

    nonzero_mindspore_all_0 = nonzero_mindspore[:, 0]
    print("nonzero_mindspore[:, 0]: ", nonzero_mindspore_all_0)

    nonzero_mindspore_all_1 = nonzero_mindspore[:, 1]
    print("nonzero_mindspore[:, 1]: ", nonzero_mindspore_all_1)

    print("nonzero_mindspore[0][0]: ", nonzero_mindspore[0][0])
    print("nonzero_mindspore[0][1]: ", nonzero_mindspore[0][1])
    print("nonzero_mindspore[0:1, 0]: ", nonzero_mindspore[0:1, 0])
    print("nonzero_mindspore[0:1, 1]: ", nonzero_mindspore[0:1, 1])
    print("nonzero_mindspore[0:1, 0:2]: ", nonzero_mindspore[0:1, 0:2])
    print("nonzero_mindspore[0:2, 0]: ", nonzero_mindspore[0:2, 0])
    print("nonzero_mindspore[0:2, 1]: ", nonzero_mindspore[0:2, 1])
    print("nonzero_mindspore[0:2, 0:2]: ", nonzero_mindspore[0:2, 0:2])
    print("nonzero_mindspore[1:3, 0]: ", nonzero_mindspore[1:3, 0])
    print("nonzero_mindspore[1:3, 1]: ", nonzero_mindspore[1:3, 1])
    print("nonzero_mindspore[1:3, 0:2]: ", nonzero_mindspore[1:3, 0:2])

    print("------------------------------------------------")