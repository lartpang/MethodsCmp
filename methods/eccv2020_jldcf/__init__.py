import torch
import torch.nn as nn

from .jldcf import CMLayer, FAModule, JL_DCF, JLModule, k, ScoreLayer
from .resnet import Bottleneck, ResNet


def jldcf():
    feature_aggregation_module = []
    for i in range(5):
        feature_aggregation_module.append(FAModule())
    upsampling = []
    for i in range(0, 4):
        upsampling.append([])
        for j in range(0, i + 1):
            upsampling[i].append(
                nn.ConvTranspose2d(
                    k,
                    k,
                    kernel_size=2 ** (j + 2),
                    stride=2 ** (j + 1),
                    padding=2 ** (j),
                )
            )
    parameter = [3, 4, 23, 3]
    backbone = ResNet(Bottleneck, parameter)

    model = JL_DCF(
        JLModule(backbone),
        CMLayer(),
        feature_aggregation_module,
        ScoreLayer(k),
        ScoreLayer(k),
        upsampling,
    )
    data = torch.cat([torch.randn(1, 3, 320, 320), torch.randn(1, 3, 320, 320)], dim=0)
    return dict(model=model, data=data)
