import torch

from .ugtr import UGTRNet
from .custom_ops import custom_ops


def ugtr(in_h=473, in_w=473):
    model = UGTRNet(layers=50, classes=1, zoom_factor=8, pretrained=False)
    data = torch.randn(1, 3, in_h, in_w)
    return dict(model=model, data=data, custom_ops=custom_ops)
