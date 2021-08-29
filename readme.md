#  A simple toolkit for counting the FLOPs/MACs, Parameters and FPS of the model.

```shell
$ python main.py --help
usage: main.py [-h] --gpu GPU [--method-dirs METHOD_DIRS [METHOD_DIRS ...]] [--method-names METHOD_NAMES [METHOD_NAMES ...]] [--height HEIGHT] [--width WIDTH]

A simple toolkit for counting the FLOPs/MACs, Parameters and FPS of the model.

optional arguments:
  -h, --help            show this help message and exit
  --gpu GPU             The gpu where your want to test your method.
  --method-dirs METHOD_DIRS [METHOD_DIRS ...]
                        The dir containing some methods.
  --method-names METHOD_NAMES [METHOD_NAMES ...]
                        The names of the methods you want to evaluate.
  --height HEIGHT       The height of the randomly constructed input image.
  --width WIDTH         The width of the randomly constructed input image.
```

## Usage

```python
# ./yourmethods_dir/yourmethod/__init__.py
from utils.builder import MODELS

@MODELS.register()
def hdfnet(in_h=352, in_w=352):
    # use the default arguments to construct the model
    model = HDFNet_Res50()
    # construct the input data as  your model needed
    data = dict(image=torch.randn(1, 3, in_h, in_w), depth=torch.randn(1, 1, in_h, in_w))
    return dict(
        model=model,
        data=data,
        # If your model contains some specitial ops, you should count its flops by yourself.
        # in main.py, if the returned dict does not contain the key `custom_ops`, it will use the default value `None`.
        # custom_ops=custom_ops
    )

```

In `methods` folder, I have provided two methods, i.e. HDFNet and DANet, as the examples.

Besides, if your want to add your own methods, your can put them into the folder `untracked_methods` . This folder will not be tracked by git.

More functional improvements and suggestions are welcome.

## Reference

The code for counting flops and parameters refers to the code of thop (https://github.com/Lyken17/pytorch-OpCounter).
At the same time, I also made appropriate improvements with reference to PR (https://github.com/Lyken17/pytorch-OpCounter/pull/122).

## Change Log

* 2021-08-29: Create a new repository and upload the code.
