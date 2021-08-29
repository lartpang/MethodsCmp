# -*- coding: utf-8 -*-
# @Time    : 2021/8/6
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang
import torch

from .hdfnet import HDFNet_Res50
from utils.builder import MODELS


@MODELS.register()
def hdfnet(in_h=352, in_w=352):
    model = HDFNet_Res50()
    data = dict(image=torch.randn(1, 3, in_h, in_w), depth=torch.randn(1, 1, in_h, in_w))
    return dict(model=model, data=data)