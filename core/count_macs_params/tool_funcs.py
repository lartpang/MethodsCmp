# -*- coding: utf-8 -*-
# @Time    : 2021/5/11
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang
from collections.abc import Iterable
from itertools import chain

from decorator import decorator
from torch import nn


def prRed(skk):
    print("\033[91m{}\033[00m".format(skk))


def prGreen(skk):
    print("\033[92m{}\033[00m".format(skk))


def prYellow(skk):
    print("\033[93m{}\033[00m".format(skk))


@decorator
def print_ops_name(func, verbose: bool = False, *args, **kwargs):
    if verbose:
        for item in chain(args, kwargs.values()):
            if isinstance(item, nn.Module):
                print(item)
    func(*args, **kwargs)


def clever_format(nums, format="%.2f"):
    if not isinstance(nums, Iterable):
        nums = [nums]
    clever_nums = []

    for num in nums:
        if num > 1e12:
            clever_nums.append(format % (num / 1e12) + "T")
        elif num > 1e9:
            clever_nums.append(format % (num / 1e9) + "G")
        elif num > 1e6:
            clever_nums.append(format % (num / 1e6) + "M")
        elif num > 1e3:
            clever_nums.append(format % (num / 1e3) + "K")
        else:
            clever_nums.append(format % num + "B")

    clever_nums = clever_nums[0] if len(clever_nums) == 1 else (*clever_nums,)
    return clever_nums
