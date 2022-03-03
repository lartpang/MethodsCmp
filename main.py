import argparse
import copy

import methods
from utils.fps.fps_counter import cal_fps
from utils.num_ops_params.num_ops_params_counter import cal_macs_params_v2

parser = argparse.ArgumentParser(
    description="A simple toolkit for counting the FLOPs/MACs, Parameters and FPS of the model."
)
parser.add_argument(
    "--method-names", nargs="+", help="The names of the methods you want to evaluate."
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
    custom_ops = model_info.get("custom_ops", None)

    model.eval()

    num_ops, num_params = cal_macs_params_v2(
        model=copy.deepcopy(model),
        data=copy.deepcopy(data),
        custom_ops=custom_ops,
        return_flops=True,
    )
    print(f"[{method_name}] FLOPs: {num_ops}, Params: {num_params}")

    # gpu_mem = cal_gpu_mem(
    #     model=copy.deepcopy(model),
    #     data=copy.deepcopy(data),
    #     device=args.gpu,
    # )
    # print(f"[{method_name}] GPU {args.gpu} MEM: {gpu_mem}")

    fps = cal_fps(
        model=copy.deepcopy(model),
        data=copy.deepcopy(data),
        num_samples=100,
        on_gpu=True,
    )
    print(f"[{method_name}] FPS: {fps}")
