# -*- coding: utf-8 -*-
# @Time    : 2021/8/6
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

from utils.registry import import_module_from_module_names
from utils.builder import build_obj_from_registry
import argparse
import os

parser = argparse.ArgumentParser(
    description="A simple toolkit for counting the FLOPs/MACs, Parameters and FPS of the model."
)
parser.add_argument("--gpu", required=True, type=int, help="The gpu where your want to test your method.")
parser.add_argument("--method-dirs", nargs="+", default=["methods"], help="The dir containing some methods.")
parser.add_argument(
    "--method-names", nargs="+", default=["hdfnet", "danet"], help="The names of the methods you want to evaluate."
)
parser.add_argument("--height", default=320, type=int, help="The height of the randomly constructed input image.")
parser.add_argument("--width", default=320, type=int, help="The width of the randomly constructed input image.")
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import_module_from_module_names(args.method_dirs, verbose=False)

from core import cal_fps, cal_gpu_mem, cal_macs_params

for method_name in args.method_names:
    model_func = build_obj_from_registry(registry_name="MODELS", obj_name=method_name)
    model_info = model_func(in_h=args.height, in_w=args.width)
    model = model_info["model"]
    data = model_info["data"]
    custom_ops = model_info.get("custom_ops", None)

    MACs, Params = cal_macs_params(model=model, data=data, custom_ops=custom_ops)
    print(f"[{method_name}] FLOPs: {float(MACs[:-1]) * 2}{MACs[-1]}, Params: {Params}")

    GPU_MEM = cal_gpu_mem(model=model, data=data, device=args.gpu)
    print(f"[{method_name}] GPU {args.gpu} MEM: {GPU_MEM}")

    FPS = cal_fps(model=model, data=data, num_samples=100)
    print(f"[{method_name}] FPS: {FPS}")
