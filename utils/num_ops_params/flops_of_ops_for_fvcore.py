import typing
from collections import Counter, OrderedDict
from numbers import Number
from typing import Any, Callable, List, Optional, Union

import numpy as np
from fvcore.nn.jit_handles import (
    Handle,
    elementwise_flop_counter,
    generic_activation_jit,
    get_shape,
)

try:
    from math import prod
except ImportError:
    from numpy import prod


def cosine_similarity_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    # inputs is a list of length 2.
    input_shapes = [get_shape(v) for v in inputs]
    # input_shapes[0]: [batch size, ..., input feature dimension]
    # input_shapes[1]: [batch size, ..., output feature dimension]
    assert 1 <= max(len(input_shapes), len(input_shapes)) <= 3
    assert input_shapes[0][0] == input_shapes[1][0], input_shapes
    assert input_shapes[0][-1] == input_shapes[1][-1], input_shapes

    flops = input_shapes[0][-1] * prod(get_shape(outputs))
    return flops


# fmt: off
def scaled_dot_product_attention_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    # scaled_dot_product_attention: (((q @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5).softmax(dim=-1) @ v)
    # q: bs,nh,l,hd; k: bs,nh,s,hd; v: bs,nh,s,hd
    q, k, v = inputs[:3]
    q_shape = get_shape(q)
    k_shape = get_shape(k)
    v_shape = get_shape(v)
    assert q_shape[:2] == k_shape[:2] == v_shape[:2], (q_shape, k_shape, v_shape)  # bs,nh
    assert q_shape[3] == k_shape[-1], (q_shape, k_shape, v_shape)  # hd
    assert k_shape[2] == v_shape[2], (q_shape, k_shape, v_shape)  # s

    bs, nh, l, hdi = q_shape
    s = k_shape[2]
    hdo = v_shape[-1]

    # q @ k.transpose(-1, -2)
    flops = bs * nh * l * hdi * s  # bs x nh x l x hd x s
    # (q @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5
    flops += bs * nh * l * s
    # ((q @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5).softmax(dim=-1)
    flops += bs * nh * l * s * 2  # exp and div (sum is ignored)
    # ((q @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5).softmax(dim=-1) @ v
    flops += bs * nh * l * s * hdo
    return flops
# fmt on


customized_ops = {
    "aten::ones_like": None,
    "aten::zeros_like": None,
    "aten::to": None,
    "aten::view": None,
    "aten::view_as": None,
    "aten::reshape": None,
    "aten::reshape_as": None,
    "aten::contiguous": None,
    "aten::permute": None,
    "aten::permute_": None,
    "aten::pad": None,
    "aten::fill": None,
    "aten::fill_": None,
    "aten::repeat": None,
    "aten::expand_as": None,
    "aten::im2col": None,
    "aten::pixel_shuffle": None,
    "aten::upsample_bicubic2d": None,
    "aten::lift_fresh": None,
    "aten::clone": None,
    "aten::scaled_dot_product_attention": scaled_dot_product_attention_flop_jit,
    #
    "aten::cumsum": None,
    "aten::clamp_min": None,
    #
    "aten::lt": None,
    "aten::ne": None,
    "aten::neg": None,
    #
    "aten::add": elementwise_flop_counter(1, 0),
    "aten::add_": elementwise_flop_counter(1, 0),
    "aten::sub": elementwise_flop_counter(1, 0),
    "aten::sub_": elementwise_flop_counter(1, 0),
    "aten::rsub": elementwise_flop_counter(1, 0),
    "aten::mul": elementwise_flop_counter(1, 0),
    "aten::mul_": elementwise_flop_counter(1, 0),
    "aten::div": elementwise_flop_counter(1, 0),
    "aten::div_": elementwise_flop_counter(1, 0),
    "aten::sum": elementwise_flop_counter(1, 0),
    "aten::mean": elementwise_flop_counter(1, 0),
    "aten::normal_": elementwise_flop_counter(1, 0),
    "aten::norm": elementwise_flop_counter(1, 0),
    "aten::frobenius_norm": elementwise_flop_counter(1, 0),
    "aten::cosine_similarity": cosine_similarity_flop_jit,
    "aten::min": None,
    "aten::max": None,
    "aten::topk": None,
    "aten::abs": None,
    "aten::pow_": elementwise_flop_counter(1, 0),
    "aten::pow": elementwise_flop_counter(1, 0),
    "aten::exp_": elementwise_flop_counter(1, 0),
    "aten::exp": elementwise_flop_counter(1, 0),
    "aten::sqrt": elementwise_flop_counter(1, 0),
    "aten::var": elementwise_flop_counter(1, 0),
    "aten::std": elementwise_flop_counter(1, 0),
    "aten::max_pool2d": elementwise_flop_counter(1, 0),
    "aten::avg_pool2d": elementwise_flop_counter(1, 0),
    "aten::max_pool3d": elementwise_flop_counter(1, 0),
    "aten::avg_pool3d": elementwise_flop_counter(1, 0),
    "aten::adaptive_max_pool2d": elementwise_flop_counter(1, 0),
    "aten::adaptive_avg_pool2d": elementwise_flop_counter(1, 0),
    # activation layer
    "aten::sigmoid": generic_activation_jit(op_name="aten::sigmoid"),
    "aten::softmax": generic_activation_jit(op_name="aten::softmax"),
    "aten::log_softmax": generic_activation_jit(op_name="aten::log_softmax"),
    "aten::feature_dropout": generic_activation_jit(op_name="aten::feature_dropout"),
    "aten::gelu": generic_activation_jit(op_name="aten::gelu"),
    "aten::sin": generic_activation_jit(op_name="aten::sin"),
    "aten::cos": generic_activation_jit(op_name="aten::cos"),
    "aten::tanh": generic_activation_jit(op_name="aten::tanh"),
    "aten::tanh_": generic_activation_jit(op_name="aten::tanh_"),
    "aten::hardtanh": generic_activation_jit(op_name="aten::hardtanh"),
    "aten::hardtanh_": generic_activation_jit(op_name="aten::hardtanh_"),
    "aten::prelu": generic_activation_jit(op_name="aten::prelu"),
    "aten::leaky_relu_": generic_activation_jit(op_name="aten::leaky_relu_"),
    "aten::silu": generic_activation_jit(op_name="aten::silu"),
}
