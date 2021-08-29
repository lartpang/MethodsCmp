# -*- coding: utf-8 -*-
# @Time    : 2021/8/6
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang
import torch

from .custom_ops import custom_ops
from .danet import DANet_V19
from utils.builder import MODELS


@MODELS.register()
def danet(in_h=384, in_w=384):
    model = DANet_V19()
    data = dict(image=torch.randn(1, 3, in_h, in_w), depth=torch.randn(1, 1, in_h, in_w))
    return dict(model=model, data=data, custom_ops=custom_ops)
