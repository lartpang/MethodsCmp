import torch

from .custom_ops import custom_ops
from .mgl import MGLNet


def mgl_r(in_h=473, in_w=473):
    model = MGLNet(layers=50, classes=1, zoom_factor=8, pretrained=False)
    data = torch.randn(1, 3, in_h, in_w)
    return dict(model=model, data=data, custom_ops=custom_ops)
