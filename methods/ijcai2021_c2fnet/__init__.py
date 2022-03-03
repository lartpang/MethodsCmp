import torch

from .c2fnet import C2FNet


def c2fnet(in_h=352, in_w=352):
    model = C2FNet()
    data = torch.randn(1, 3, in_h, in_w)
    return dict(model=model, data=data)
