# A simple toolkit for counting the FLOPs/MACs, Parameters and FPS of the model.

```shell
$ python main.py --help
usage: main.py [-h] [--method-names METHOD_NAMES [METHOD_NAMES ...]]
               [--mode {ops_params,fps,gpu_mem} [{ops_params,fps,gpu_mem} ...]]

A simple toolkit for counting the FLOPs/MACs, Parameters, FPS and GPU Memory of the model.

optional arguments:
  -h, --help            show this help message and exit
  --method-names METHOD_NAMES [METHOD_NAMES ...]
                        The names of the methods you want to evaluate.
  --mode {ops_params,fps,gpu_mem} [{ops_params,fps,gpu_mem} ...]
```

## Usage

```bash
pip install fvcore
```

In `methods` folder, I have provided some recent methods, i.e. CoNet, DANet, HDFNet, JL-DCF, and UC-Net, as the examples.

More functional improvements and suggestions are welcome.

An example:

```shell
python main.py --method-names zoomnet ugtr c2fnet ujsc pfnet mgl_r slsr sinet
```

## Change Log

* 2023-07-28: 
  * [New & Important] Update the library for count FLOPs/MACS from `pytorch-OpCounter` to `fvcore` which can count FLOPs/MACs of the complex module, like Transformer.
  * [Experimental Feature] Add the new feature for counting the peak inference GPU memory of the model.
* 2022-03-03: Add more methods, add the gpu warmup process in counting FPS and update the readme.
* 2021-09-29: Refactor again.
* 2021-08-31: Refactor.
* 2021-08-29: Create a new repository and upload the code.

```latex
@misc{MethodsCmp,
	author       = {Youwei Pang},
	title        = {MethodsCmp: A Simple Toolkit for Counting the FLOPs/MACs, Parameters and FPS of Pytorch-based Methods},
	howpublished = {\url{https://github.com/lartpang/MethodsCmp}},
	year         = {2021}
}
```
