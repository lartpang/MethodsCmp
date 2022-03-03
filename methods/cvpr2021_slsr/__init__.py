import torch

from .slsr import Generator
from .custom_ops import custom_ops


def slsr(in_h=480, in_w=480):
    model = Generator(channel=32)
    data = torch.randn(1, 3, in_h, in_w)
    return dict(model=model, data=data, custom_ops=custom_ops)
