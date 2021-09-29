import pynvml
import torch

from ..misc import to_cuda

pynvml.nvmlInit()


@torch.no_grad()
def cal_gpu_mem(model, data, device=0):
    assert torch.cuda.is_available()

    print(f"Counting GPU memory for {model.__class__.__name__}")
    data = to_cuda(data)
    model.cpu()
    torch.cuda.empty_cache()

    handle = pynvml.nvmlDeviceGetHandleByIndex(device)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    initial_mem = meminfo.used

    model.cuda()
    model.eval()
    data = (
        data.cuda()
        if isinstance(data, torch.Tensor)
        else {k: v.cuda() for k, v in data.items()}
    )
    runtime_mems = []
    for _ in range(100):
        model(data)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        runtime_mems.append(meminfo.used - initial_mem)
    average_mem = [
        f"Total: {meminfo.used / 1024 ** 2:.3f}MB",
        f"Model: {sum(runtime_mems) / 100 / 1024 ** 2:.3f}MB",
        f"Other: {initial_mem / 1024 ** 2:.3f}MB",
    ]
    average_mem = " | ".join(average_mem)
    return average_mem
