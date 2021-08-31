# -*- coding: utf-8 -*-
# @Time    : 2021/5/12
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang


import torch
import torch.nn as nn
import sys

sys.path.append("..")
from model_counter.num_ops_params import count_info, tool_funcs


class SubModule(nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()
        self.conv = nn.Conv2d(4, 8, 1, bias=False)

    def forward(self, x):
        x = x * x
        return self.conv(x)


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.conv = nn.Conv2d(2, 4, 1, bias=False)
        self.sub_module = SubModule()

    def forward(self, x):
        return self.sub_module(self.conv(x))


def count_flops_for_sub_module(m, x, y):
    x = x[0]
    _, _, H, W = x.shape
    m.total_ops += torch.DoubleTensor([int(H * W)])


model = MyModule()

print(sum([m.numel() for m in model.parameters()]))

input = torch.randn(1, 2, 5, 5)
macs, params = count_info.profile(
    model,
    inputs=(input,),
    custom_ops={
        SubModule: count_flops_for_sub_module,
    },
    # verbose_for_hook=True,
    verbose_for_count=True,
    # exclude_modules=(Attention,),
)
macs, params = tool_funcs.clever_format([macs, params], "%.3f")
print(macs, params)


"""
40
main
LayerSelf: ops: 0, params: 0
main->conv(Conv2d)
LayerSelf: ops: 200.0, params: 8.0
main->conv(Conv2d)
LayerTotal: ops: 200.0, params: 8.0
main->sub_module(SubModule)
LayerSelf: ops: 25.0, params: 0.0
main->sub_module(SubModule)->conv(Conv2d)
LayerSelf: ops: 800.0, params: 32.0
main->sub_module(SubModule)->conv(Conv2d)
LayerTotal: ops: 800.0, params: 32.0
main->sub_module(SubModule)
LayerTotal: ops: 825.0, params: 32.0
main
LayerTotal: ops: 1025.0, params: 40.0
1.025K 40.000B
"""
