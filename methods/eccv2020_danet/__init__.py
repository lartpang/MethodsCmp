import torch

from .custom_ops import custom_ops
from .danet import DANet_V19


def danet(in_h=384, in_w=384):
    model = DANet_V19()
    data = dict(
        image=torch.randn(1, 3, in_h, in_w), depth=torch.randn(1, 1, in_h, in_w)
    )
    return dict(model=model, data=data, custom_ops=custom_ops)
