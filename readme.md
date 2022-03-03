# A simple toolkit for counting the FLOPs/MACs, Parameters and FPS of the model.

```shell
$ python main.py --help
usage: main.py [-h] [--method-names METHOD_NAMES [METHOD_NAMES ...]]

A simple toolkit for counting the FLOPs/MACs, Parameters and FPS of the model.

optional arguments:
  -h, --help            show this help message and exit
  --method-names METHOD_NAMES [METHOD_NAMES ...]
                        The names of the methods you want to evaluate.
```

## Usage

In `methods` folder, I have provided some recent methods, i.e. CoNet, DANet, HDFNet, JL-DCF, and UC-Net, as the examples.

More functional improvements and suggestions are welcome.

An example:

```shell
python main.py --method-names zoomnet ugtr c2fnet ujsc pfnet mgl_r slsr sinet
```

## Reference

- The code for counting flops and parameters refers to the code of thop (https://github.com/Lyken17/pytorch-OpCounter).
- At the same time, I also made appropriate improvements with reference to PR (https://github.com/Lyken17/pytorch-OpCounter/pull/122).

## Change Log

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
