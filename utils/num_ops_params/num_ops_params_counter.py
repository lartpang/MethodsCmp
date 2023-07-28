import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, parameter_count

from . import tool_funcs
from .flops_of_ops_for_fvcore import customized_ops


@torch.no_grad()
def cal_macs_params(model: nn.Module, data, return_flops=False, return_number=False):
    macs = FlopCountAnalysis(model, inputs=data)
    for name, count_counter in customized_ops.items():
        macs.set_op_handle(name, count_counter)
    macs = macs.total()
    if return_flops:
        macs *= 2

    params = parameter_count(model)['']
    if not return_number:
        macs, params = tool_funcs.clever_format([macs, params], "%.3f")
    return macs, params
