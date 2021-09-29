# A simple toolkit for counting the FLOPs/MACs, Parameters and FPS of the model.

```shell
$ python main.py --help
usage: main.py [-h] [--gpu GPU] [--method-names METHOD_NAMES [METHOD_NAMES ...]] [--height HEIGHT] [--width WIDTH]

A simple toolkit for counting the FLOPs/MACs, Parameters and FPS of the model.

optional arguments:
  -h, --help            show this help message and exit
  --gpu GPU             The gpu where your want to test your method.
  --method-names METHOD_NAMES [METHOD_NAMES ...]
                        The names of the methods you want to evaluate.
  --height HEIGHT       The height of the randomly constructed input image.
  --width WIDTH         The width of the randomly constructed input image.
```

## Usage

In `methods` folder, I have provided five methods, i.e. CoNet, DANet, HDFNet, JL-DCF, and UC-Net, as the examples.

Besides, if your want to add your own methods to the repo, your can put them into the folder `methods` and push it me.

More functional improvements and suggestions are welcome.

## Reference

- The code for counting flops and parameters refers to the code of thop (https://github.com/Lyken17/pytorch-OpCounter).
- At the same time, I also made appropriate improvements with reference to PR (https://github.com/Lyken17/pytorch-OpCounter/pull/122).

## Change Log

* 2021-09-29: Refactor again.
* 2021-08-31: Refactor.
* 2021-08-29: Create a new repository and upload the code.
