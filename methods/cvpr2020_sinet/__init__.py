import torch

from .sinet import SINet_ResNet50


def sinet(in_h=352, in_w=352):
    model = SINet_ResNet50()
    data = torch.randn(1, 3, in_h, in_w)
    return dict(model=model, data=data)
