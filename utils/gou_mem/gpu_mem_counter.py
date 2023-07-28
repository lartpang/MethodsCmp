import torch
from torch import nn

from utils.misc import to_cuda


@torch.no_grad()
def cal_gpu_mem(model: nn.Module, data, return_number=False):
    assert torch.cuda.is_available()

    print(f"Counting GPU memory for {model.__class__.__name__}")
    model.cpu()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    data = to_cuda(data)
    model.cuda()
    initial_mem = torch.cuda.max_memory_allocated()

    model(data)
    forward_mem = torch.cuda.max_memory_allocated()

    if not return_number:
        average_mem = [
            f"Total: {forward_mem / 1024 ** 2:.3f}MB",
            f"Model: {initial_mem / 1024 ** 2:.3f}MB",
            f"Other: {(forward_mem - initial_mem) / 1024 ** 2:.3f}MB",
        ]
        average_mem = " | ".join(average_mem)
    else:
        average_mem = forward_mem / 1024 ** 2
    return average_mem
