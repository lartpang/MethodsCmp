import torch

from .custom_ops import custom_ops
from .pfnet import PFNet


def pfnet(in_h=416, in_w=416):
    model = PFNet()
    data = torch.randn(1, 3, in_h, in_w)
    return dict(model=model, data=data, custom_ops=custom_ops)
