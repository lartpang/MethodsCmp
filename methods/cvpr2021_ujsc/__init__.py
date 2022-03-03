import torch

from .ujsc import Generator


def ujsc(in_h=480, in_w=480):
    model = Generator()
    data = torch.randn(1, 3, in_h, in_w)
    return dict(model=model, data=data)
