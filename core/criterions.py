import mindspore as ms
import mindspore.nn as nn

class CrossEntropyLossWithIgnored(nn.Cell):

    def __init__(self, sparse=False, reduction='none', ignore_index=255):
        super(CrossEntropyLossWithIgnored, self).__init__()
        self.ignore_index = ignore_index
        self.ce = nn.SoftmaxCrossEntropyWithLogits(sparse=sparse, reduction=reduction)

    def construct(self, logits, labels):
        valid_index = ms.ops.nonzero(labels != self.ignore_index).flatten()
        return self.ce(logits[valid_index], labels[valid_index])

