import torch

from .zoomnet import ZoomNet


def zoomnet(in_h=384, in_w=384):
    model = ZoomNet()
    data = {
        "image1.5": torch.randn(1, 3, int(in_h * 1.5), int(in_w * 1.5)),
        "image1.0": torch.randn(1, 3, in_h, in_w),
        "image0.5": torch.randn(1, 3, int(in_h * 0.5), int(in_w * 0.5)),
    }
    return dict(model=model, data=data)
