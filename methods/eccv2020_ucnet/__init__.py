import torch

from .ucnet import Generator
from .custom_ops import custom_ops


def ucnet(in_h=352, in_w=352):
    model = Generator(channel=32, latent_dim=3)
    data = dict(
        image=torch.randn(1, 3, in_h, in_w), depth=torch.randn(1, 3, in_h, in_w)
    )
    return dict(model=model, data=data, custom_ops=custom_ops)
