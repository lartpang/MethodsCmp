import torch

from .conet import CoNet


def conet(in_h=256, in_w=256):
    model = CoNet()
    data = torch.randn(1, 3, in_h, in_w)
    return dict(model=model, data=data)
