from . import tool_funcs, count_info
import torch


@torch.no_grad()
def cal_macs_params(model, data, custom_ops=None):
    print(f"Calculating the MACs and the number of parameters for model {model.__class__.__name__} with data ")

    model.eval()
    macs, params = count_info.profile(
        model,
        inputs=(data,),
        custom_ops=custom_ops,
        # verbose_for_hook=True,
        # verbose_for_count=True,
        # exclude_self_modules=(Attention,),
    )
    macs, params = tool_funcs.clever_format([macs, params], "%.3f")
    return macs, params
