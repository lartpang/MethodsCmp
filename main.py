import argparse
import os

import methods
from utils.fps.fps_counter import cal_fps
from utils.gou_mem.gpu_mem_counter import cal_gpu_mem
from utils.num_ops_params.num_ops_params_counter import cal_macs_params_v2

parser = argparse.ArgumentParser(
    description="A simple toolkit for counting the FLOPs/MACs, Parameters and FPS of the model."
)
parser.add_argument(
    "--gpu", default=0, type=int, help="The gpu where your want to test your method."
)
parser.add_argument(
    "--method-names",
    nargs="+",
    default=["hdfnet", "danet"],
    help="The names of the methods you want to evaluate.",
)
parser.add_argument(
    "--height",
    type=int,
    help="The height of the randomly constructed input image.",
)
parser.add_argument(
    "--width",
    type=int,
    help="The width of the randomly constructed input image.",
)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

for method_name in args.method_names:
    print(f" ==>> PROCESSING THE METHOD {method_name}... <<== ")

    model_func = vars(methods).get(method_name)
    if model_func is None:
        raise KeyError(
            f"{method_name} is not be supported ({list(vars(methods).keys())})"
        )

    hw_kwargs = {}
    if args.height is not None:
        hw_kwargs["in_h"] = args.height
    if args.width is not None:
        hw_kwargs["in_w"] = args.width
    model_info = model_func()

    model = model_info["model"]
    data = model_info["data"]
    custom_ops = model_info.get("custom_ops", None)

    num_ops, num_params = cal_macs_params_v2(
        model=model, data=data, custom_ops=custom_ops, return_flops=True
    )
    print(f"[{method_name}] FLOPs: {num_ops}, Params: {num_params}")

    gpu_mem = cal_gpu_mem(model=model, data=data, device=args.gpu)
    print(f"[{method_name}] GPU {args.gpu} MEM: {gpu_mem}")

    fps = cal_fps(model=model, data=data, num_samples=100, on_gpu=True)
    print(f"[{method_name}] FPS: {fps}")
