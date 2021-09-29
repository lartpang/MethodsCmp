import torch

from .hdfnet import HDFNet_Res50, HDFNet_VGG16


def hdfnet_res50(in_h=320, in_w=320):
    model = HDFNet_Res50()
    data = dict(
        image=torch.randn(1, 3, in_h, in_w), depth=torch.randn(1, 1, in_h, in_w)
    )
    return dict(model=model, data=data)


def hdfnet_vgg16(in_h=320, in_w=320):
    model = HDFNet_VGG16()
    data = dict(
        image=torch.randn(1, 3, in_h, in_w), depth=torch.randn(1, 1, in_h, in_w)
    )
    return dict(model=model, data=data)
