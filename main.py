import argparse

import methods
from utils.fps.fps_counter import cal_fps
from utils.gou_mem.gpu_mem_counter import cal_gpu_mem
from utils.num_ops_params.num_ops_params_counter import cal_macs_params

parser = argparse.ArgumentParser(
    description="A simple toolkit for counting the FLOPs/MACs, Parameters, FPS and GPU Memory of the model."
)
parser.add_argument(
    "--method-names", nargs="+", help="The names of the methods you want to evaluate."
)
parser.add_argument(
    "--mode",
    nargs="+",
    choices=["ops_params", "fps", "gpu_mem"],
    default=["ops_params", "fps", "gpu_mem"],
)
args = parser.parse_args()

for method_name in args.method_names:
    print(f" ==>> PROCESSING THE METHOD {method_name}... <<== ")

    model_func = vars(methods).get(method_name)
    if model_func is None:
        raise KeyError(
            f"{method_name} is not be supported ({list(vars(methods).keys())})"
        )

    model_info = model_func()
    model = model_info["model"]
    data = model_info["data"]
    model.eval()

    if "ops_params" in args.mode:
        num_ops, num_params = cal_macs_params(model=model, data=data, return_flops=True)
        print(f"[{method_name}] FLOPs: {num_ops}, Params: {num_params}")

    if "fps" in args.mode:
        fps = cal_fps(model=model, data=data, num_samples=100, on_gpu=True)
        print(f"[{method_name}] FPS: {fps}")

    if "gpu_mem" in args.mode:
        gpu_mem = cal_gpu_mem(model=model, data=data)
        print(f"[{method_name}] MEM: {gpu_mem}")
