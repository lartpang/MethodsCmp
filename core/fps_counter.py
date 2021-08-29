import torch
from tqdm import tqdm
from utils.misc import to_cuda


@torch.no_grad()
def cal_fps(model, data, num_samples=1000):
    print(f"Calculating the FPS for model {model.__class__.__name__} with {num_samples} data ")
    data = to_cuda(data)

    model.cuda()
    model.eval()

    tqdm_iter = tqdm(range(num_samples), total=num_samples, leave=False)
    elapsed_time_s_list = []
    for i in tqdm_iter:
        tqdm_iter.set_description(f"te=>{i + 1} ")
        # https://pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        model(data)
        end_event.record()
        torch.cuda.synchronize()  # Wait for the events to be recorded!
        elapsed_time_ms = start_event.elapsed_time(end_event)
        elapsed_time_s_list.append(elapsed_time_ms / 1000)

        # https://github.com/lartpang/MINet/blob/master/code/utils/cal_fps.py
        # start_time = time.time()
        # torch.cuda.synchronize()
        # model(data)  # 按照实际情况改写
        # torch.cuda.synchronize()
        # end_time = time.time()
        # elapsed_time_s_list.append(end_time - start_time)
    fps = f"{len(elapsed_time_s_list) / sum(elapsed_time_s_list):.3f}"
    return fps
