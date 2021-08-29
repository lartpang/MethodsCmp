import torch
import pynvml
from utils.misc import to_cuda

pynvml.nvmlInit()


@torch.no_grad()
def cal_gpu_mem(model, data, device=0):
    print(f"Calculating the GPU memory for model {model.__class__.__name__}")
    data = to_cuda(data)

    torch.cuda.empty_cache()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    initial_mem = meminfo.used

    model.cuda()
    model.eval()
    data = data.cuda() if isinstance(data, torch.Tensor) else {k: v.cuda() for k, v in data.items()}
    runtime_mems = []
    for _ in range(10):
        model(data)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        runtime_mems.append(meminfo.used - initial_mem)
    average_mem = f"{sum(runtime_mems) / 10 / 1024 ** 2:.3f}MB"
    return average_mem
