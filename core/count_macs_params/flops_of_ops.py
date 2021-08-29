# -*- coding: utf-8 -*-
# @Time    : 2021/5/11
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang
import logging
import operator
from distutils.version import LooseVersion
from functools import reduce

import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd

from .tool_funcs import print_ops_name

multiply_adds = 1


def count_parameters(ops: nn.Module, in_tensor, out_tensor):
    total_params = 0
    for p in ops.parameters(recurse=False):  # 仅计算当前层的参数，不包含子模块
        total_params += torch.DoubleTensor([p.numel()])
    ops.total_params[0] = total_params


def zero_ops(ops: nn.Module, in_tensor, out_tensor):
    ops.total_ops += torch.DoubleTensor([int(0)])


@print_ops_name(verbose=False)
def count_convNd(ops: _ConvNd, in_tensor: (torch.Tensor,), out_tensor: torch.Tensor):
    in_tensor = in_tensor[0]

    kernel_ops = torch.zeros(ops.weight.size()[2:]).numel()  # Kw in_tensor Kh
    bias_ops = 1 if ops.bias is not None else 0

    # N in_tensor Cout in_tensor H in_tensor W in_tensor  (Cin in_tensor Kw in_tensor Kh + bias)
    total_ops = out_tensor.nelement() * (ops.in_channels // ops.groups * kernel_ops + bias_ops)

    ops.total_ops += torch.DoubleTensor([int(total_ops)])


def count_convNd_ver2(ops: _ConvNd, in_tensor: (torch.Tensor,), out_tensor: torch.Tensor):
    in_tensor = in_tensor[0]

    # N in_tensor H in_tensor W (exclude Cout)
    output_size = torch.zeros((out_tensor.size()[:1] + out_tensor.size()[2:])).numel()
    # Cout in_tensor Cin in_tensor Kw in_tensor Kh
    kernel_ops = ops.weight.nelement()
    if ops.bias is not None:
        # Cout in_tensor 1
        kernel_ops += ops.bias.nelement()
    # in_tensor N in_tensor H in_tensor W in_tensor Cout in_tensor (Cin in_tensor Kw in_tensor Kh + bias)
    ops.total_ops += torch.DoubleTensor([int(output_size * kernel_ops)])


def count_convNd_mul(ops: _ConvNd, in_tensor: (torch.Tensor,), out_tensor: torch.Tensor):
    in_tensor = in_tensor[0]

    # N in_tensor H in_tensor W (exclude Cout)
    output_size = torch.zeros((out_tensor.size()[:1] + out_tensor.size()[2:])).numel()
    # Cout in_tensor Cin in_tensor Kw in_tensor Kh
    kernel_ops = ops.weight.nelement()
    # in_tensor N in_tensor H in_tensor W in_tensor Cout in_tensor (Cin in_tensor Kw in_tensor Kh + bias)
    ops.total_ops += torch.DoubleTensor([int(output_size * kernel_ops)])


def count_relu(ops: nn.Module, in_tensor, out_tensor):
    in_tensor = in_tensor[0]

    nelements = in_tensor.numel()

    ops.total_ops += torch.DoubleTensor([int(nelements)])


@print_ops_name(verbose=False)
def count_softmax(ops: nn.Module, in_tensor, out_tensor):
    in_tensor = in_tensor[0]

    N, C, *HW = in_tensor.size()
    nelements = C * reduce(operator.mul, HW)

    total_exp = nelements
    total_add = nelements - 1
    total_div = nelements
    total_ops = N * (total_exp + total_add + total_div)

    ops.total_ops += torch.DoubleTensor([int(total_ops)])


def count_avgpool(ops: nn.Module, in_tensor, out_tensor):
    # total_add = torch.prod(torch.Tensor([ops.kernel_size]))
    # total_div = 1
    # kernel_ops = total_add + total_div
    kernel_ops = 1
    num_elements = out_tensor.numel()
    total_ops = kernel_ops * num_elements

    ops.total_ops += torch.DoubleTensor([int(total_ops)])


def count_adap_avgpool(ops: nn.Module, in_tensor, out_tensor):
    kernel = torch.DoubleTensor([*(in_tensor[0].shape[2:])]) // torch.DoubleTensor([*(out_tensor.shape[2:])])
    total_add = torch.prod(kernel)
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = out_tensor.numel()
    total_ops = kernel_ops * num_elements

    ops.total_ops += torch.DoubleTensor([int(total_ops)])


# TODO: verify the accuracy
def count_upsample(ops: nn.Module, in_tensor, out_tensor):
    if ops.mode not in (
        "nearest",
        "linear",
        "bilinear",
        "bicubic",
    ):  # "trilinear"
        logging.warning("mode %s is not implemented yet, take it a zero op" % ops.mode)
        return zero_ops(ops, in_tensor, out_tensor)

    if ops.mode == "nearest":
        return zero_ops(ops, in_tensor, out_tensor)

    in_tensor = in_tensor[0]
    if ops.mode == "linear":
        total_ops = out_tensor.nelement() * 5  # 2 muls + 3 add
    elif ops.mode == "bilinear":
        # https://en.wikipedia.org/wiki/Bilinear_interpolation
        total_ops = out_tensor.nelement() * 11  # 6 muls + 5 adds
    elif ops.mode == "bicubic":
        # https://en.wikipedia.org/wiki/Bicubic_interpolation
        # Product matrix [4x4] in_tensor [4x4] in_tensor [4x4]
        ops_solve_A = 224  # 128 muls + 96 adds
        ops_solve_p = 35  # 16 muls + 12 adds + 4 muls + 3 adds
        total_ops = out_tensor.nelement() * (ops_solve_A + ops_solve_p)
    elif ops.mode == "trilinear":
        # https://en.wikipedia.org/wiki/Trilinear_interpolation
        # can viewed as 2 bilinear + 1 linear
        total_ops = out_tensor.nelement() * (13 * 2 + 5)

    ops.total_ops += torch.DoubleTensor([int(total_ops)])


def count_linear(ops, in_tensor, out_tensor):
    # per output element
    total_mul = ops.in_features
    # total_add = ops.in_features - 1
    # total_add += 1 if ops.bias is not None else 0
    num_elements = out_tensor.numel()
    total_ops = total_mul * num_elements

    ops.total_ops += torch.DoubleTensor([int(total_ops)])


def count_bn(ops: nn.Module, in_tensor, out_tensor):
    """
    如何计算CNN中batch normalization的计算复杂度（FLOPs）？ - 采石工的回答 - 知乎
    https://www.zhihu.com/question/400039617/answer/1270642900

    PyTorch 源码解读之 BN & SyncBN：BN 与 多卡同步 BN 详解 - OpenMMLab的文章 - 知乎
    https://zhuanlan.zhihu.com/p/337732517

    pytorch BatchNorm参数详解，计算过程:
    https://blog.csdn.net/weixin_39228381/article/details/107896863

    https://github.com/sovrasov/flops-counter.pytorch/blob/469b7430ec7c6aa8c258da1bca2c04de81fc9613/ptflops/flops_counter.py#L285-L291

    :param base_ops: bn ops
    :param num_elements: C in_tensor H in_tensor W
    :return:
    """
    in_tensor = in_tensor[0]

    total_ops = in_tensor.numel()
    if ops.affine:
        # subtract, divide, gamma, beta
        total_ops *= 2

    ops.total_ops += torch.DoubleTensor([int(total_ops)])


register_hooks = {
    nn.ZeroPad2d: zero_ops,  # padding does not involve any multiplication.
    nn.Conv1d: count_convNd_mul,
    nn.Conv2d: count_convNd_mul,
    nn.Conv3d: count_convNd_mul,
    nn.ConvTranspose1d: count_convNd_mul,
    nn.ConvTranspose2d: count_convNd_mul,
    nn.ConvTranspose3d: count_convNd_mul,
    nn.BatchNorm1d: count_bn,
    nn.BatchNorm2d: count_bn,
    nn.BatchNorm3d: count_bn,
    nn.ReLU: zero_ops,
    nn.ReLU6: zero_ops,
    nn.PReLU: count_relu,
    nn.LeakyReLU: count_relu,
    nn.Softmax: count_softmax,
    nn.MaxPool1d: zero_ops,
    nn.MaxPool2d: zero_ops,
    nn.MaxPool3d: zero_ops,
    nn.AdaptiveMaxPool1d: zero_ops,
    nn.AdaptiveMaxPool2d: zero_ops,
    nn.AdaptiveMaxPool3d: zero_ops,
    nn.AvgPool1d: count_avgpool,
    nn.AvgPool2d: count_avgpool,
    nn.AvgPool3d: count_avgpool,
    nn.AdaptiveAvgPool1d: count_adap_avgpool,
    nn.AdaptiveAvgPool2d: count_adap_avgpool,
    nn.AdaptiveAvgPool3d: count_adap_avgpool,
    nn.Linear: count_linear,
    nn.Dropout: zero_ops,
    nn.Upsample: count_upsample,
    nn.UpsamplingBilinear2d: count_upsample,
    nn.UpsamplingNearest2d: count_upsample,
    nn.Identity: zero_ops,
}

if LooseVersion(torch.__version__) < LooseVersion("1.0.0"):
    logging.warning(
        "You are using an old version PyTorch {version}, which THOP is not going to support in the future.".format(
            version=torch.__version__
        )
    )

if LooseVersion(torch.__version__) >= LooseVersion("1.1.0"):
    register_hooks.update({nn.SyncBatchNorm: count_bn})
