from collections import OrderedDict

import torch
import torchvision.models as models
from torch.nn import Softmax

cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-8)

import math

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter

import numpy as np
import scipy.stats as st


def gkern(kernlen=16, nsig=3):
    interval = (2 * nsig + 1.0) / kernlen
    x = np.linspace(-nsig - interval / 2.0, nsig + interval / 2.0, kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


def min_max_norm(in_):
    max_ = in_.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    min_ = in_.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    in_ = in_ - min_
    return in_.div(max_ - min_ + 1e-8)


class HA(nn.Module):
    # holistic attention module
    def __init__(self):
        super(HA, self).__init__()
        gaussian_kernel = np.float32(gkern(31, 4))
        gaussian_kernel = gaussian_kernel[np.newaxis, np.newaxis, ...]
        self.gaussian_kernel = Parameter(torch.from_numpy(gaussian_kernel))

    def forward(self, attention, x):
        soft_attention = F.conv2d(attention, self.gaussian_kernel, padding=15)
        soft_attention = min_max_norm(soft_attention)
        x = torch.mul(x, soft_attention.max(attention))
        return x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class B2_ResNet(nn.Module):
    # ResNet50 with two branches
    def __init__(self):
        # self.inplanes = 128
        self.inplanes = 64
        super(B2_ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3_1 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4_1 = self._make_layer(Bottleneck, 512, 3, stride=2)

        self.inplanes = 512
        self.layer3_2 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4_2 = self._make_layer(Bottleneck, 512, 3, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x1 = self.layer3_1(x)
        x1 = self.layer4_1(x1)

        x2 = self.layer3_2(x)
        x2 = self.layer4_2(x2)

        return x1, x2


class FCDiscriminator_SOD(nn.Module):
    def __init__(self, ndf=64):
        super(FCDiscriminator_SOD, self).__init__()
        self.conv1_1 = nn.Conv2d(7, ndf, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(4, ndf, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1)
        self.classifier = nn.Conv2d(ndf, 1, kernel_size=3, stride=2, padding=1)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.leaky_relu3 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.leaky_relu4 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.bn1_1 = nn.BatchNorm2d(ndf)
        self.bn1_2 = nn.BatchNorm2d(ndf)
        self.bn2 = nn.BatchNorm2d(ndf)
        self.bn3 = nn.BatchNorm2d(ndf)
        self.bn4 = nn.BatchNorm2d(ndf)
        # self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
        # #self.sigmoid = nn.Sigmoid()

    def forward(self, x, pred):
        x = torch.cat((x, pred), 1)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.leaky_relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leaky_relu4(x)
        x = self.classifier(x)
        return x


class FCDiscriminator_COD(nn.Module):
    def __init__(self, ndf=64):
        super(FCDiscriminator_COD, self).__init__()
        self.conv1_1 = nn.Conv2d(7, ndf, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(4, ndf, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1)
        self.classifier = nn.Conv2d(ndf, 1, kernel_size=3, stride=2, padding=1)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.leaky_relu3 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.leaky_relu4 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.bn1_1 = nn.BatchNorm2d(ndf)
        self.bn1_2 = nn.BatchNorm2d(ndf)
        self.bn2 = nn.BatchNorm2d(ndf)
        self.bn3 = nn.BatchNorm2d(ndf)
        self.bn4 = nn.BatchNorm2d(ndf)
        # self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
        # #self.sigmoid = nn.Sigmoid()

    def forward(self, x, pred):
        x = torch.cat((x, pred), 1)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.leaky_relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leaky_relu4(x)
        x = self.classifier(x)
        return x


class CAM_Module(nn.Module):
    """Channel attention module"""

    # paper: Dual Attention Network for Scene Segmentation
    def __init__(self):
        super(CAM_Module, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : input feature maps( B X C X H X W)
        returns :
            out : attention value + input feature ( B X C X H X W)
            attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class PAM_Module(nn.Module):
    """Position attention module"""

    # paper: Dual Attention Network for Scene Segmentation
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
        )
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
        )
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1
        )
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : input feature maps( B X C X H X W)
        returns :
            out : attention value + input feature ( B X C X H X W)
            attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = (
            self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        )
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class Classifier_Module(nn.Module):
    def __init__(self, dilation_series, padding_series, NoLabels, input_channel):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(
                    input_channel,
                    NoLabels,
                    kernel_size=3,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=True,
                )
            )
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    # paper: Image Super-Resolution Using Very DeepResidual Channel Attention Networks
    # input: B*C*H*W
    # output: B*C*H*W
    def __init__(
        self, n_feat, kernel_size=3, reduction=16, bias=True, bn=False, res_scale=1
    ):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(
                self.default_conv(n_feat, n_feat, kernel_size, bias=bias)
            )
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(nn.ReLU(True))
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size // 2),
            bias=bias,
        )

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


class RFB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=3, dilation=3),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=5, dilation=5),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=7, dilation=7),
        )
        self.conv_cat = nn.Conv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = nn.Conv2d(in_channel, out_channel, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = torch.cat((x0, x1, x2, x3), 1)
        x_cat = self.conv_cat(x_cat)

        x = self.relu(x_cat + self.conv_res(x))
        return x


class BasicConv2d(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1
    ):
        super(BasicConv2d, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
        )

    def forward(self, x):
        x = self.conv_bn(x)
        return x


class Triple_Conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Triple_Conv, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=1),
        )

    def forward(self, x):
        return self.reduce(x)


class Share_feat_decoder(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32):
        super(Share_feat_decoder, self).__init__()
        self.resnet = B2_ResNet()
        self.relu = nn.ReLU(inplace=True)
        self.upsample8 = nn.Upsample(
            scale_factor=8, mode="bilinear", align_corners=True
        )
        self.dropout = nn.Dropout(0.3)
        self.clc_layer = self._make_pred_layer(
            Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, channel * 4
        )
        self.upsample4 = nn.Upsample(
            scale_factor=4, mode="bilinear", align_corners=True
        )
        self.upsample2 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )

        self.conv4 = self._make_pred_layer(
            Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 2048
        )
        self.conv3 = self._make_pred_layer(
            Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 1024
        )
        self.conv2 = self._make_pred_layer(
            Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 512
        )
        self.conv1 = self._make_pred_layer(
            Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 256
        )

        self.racb_43 = RCAB(channel * 2)
        self.racb_432 = RCAB(channel * 3)
        self.racb_4321 = RCAB(channel * 4)

        self.conv43 = self._make_pred_layer(
            Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 2 * channel
        )
        self.conv432 = self._make_pred_layer(
            Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 3 * channel
        )
        self.pam4 = PAM_Module(channel)
        self.pam3 = PAM_Module(channel)

        self.HA = HA()

    def _make_pred_layer(
        self, block, dilation_series, padding_series, NoLabels, input_channel
    ):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, x1, x2, x3, x4):
        conv1_feat = self.conv1(x1)
        conv2_feat = self.conv2(x2)
        conv3_feat = self.conv3(x3)
        conv3_feat = self.pam3(conv3_feat)
        conv4_feat = self.conv4(x4)
        conv4_feat = self.pam4(conv4_feat)
        conv4_feat = self.upsample2(conv4_feat)  # (b, 32, 22, 22)
        conv43 = torch.cat((conv4_feat, conv3_feat), 1)  # (b, 65, 22, 22)
        conv43 = self.racb_43(conv43)
        conv43 = self.conv43(conv43)
        conv43 = self.upsample2(conv43)  # (b, 32, 44, 44)
        conv432 = torch.cat((self.upsample2(conv4_feat), conv43, conv2_feat), 1)
        conv432 = self.racb_432(conv432)  # (b, 97, 44, 44)
        conv432 = self.conv432(conv432)
        conv432 = self.upsample2(conv432)

        conv4321 = torch.cat(
            (self.upsample4(conv4_feat), self.upsample2(conv43), conv432, conv1_feat), 1
        )
        conv4321 = self.racb_4321(conv4321)

        pred_init = self.clc_layer(conv4321)  # (b, 1, 44, 44)
        return pred_init


class Saliency_feat_encoder(nn.Module):
    # resnet based encoder decoder
    def __init__(self):
        super(Saliency_feat_encoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.resnet = B2_ResNet()
        self.aspp = _ASPPModule(2048, 256, 16)
        if self.training:
            self.initialize_weights()

        # self.conv_edge_1 = nn.Conv2d(64, 32, 1)
        # self.conv_edge_2 = nn.Conv2d(512, 32, 1)
        # self.conv_edge_4 = nn.Conv2d(2048, 32, 1)
        # self.conv_edge_11 = nn.Conv2d(32, 32, 3, padding=1)
        # self.conv_edge_21 = nn.Conv2d(512, 32, 1)
        # self.conv_edge_41 = nn.Conv2d(2048, 32, 1)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x0 = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x0)  # 256 x 64 x 64
        x2 = self.resnet.layer2(x1)  # 512 x 32 x 32
        x3 = self.resnet.layer3_1(x2)  # 1024 x 16 x 16
        x4 = self.resnet.layer4_1(x3)  # 2048 x 8 x 8
        # x4 = self.aspp(x4)

        return x1, x2, x3, x4, self.resnet

    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True)
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif "_1" in k:
                name = k.split("_1")[0] + k.split("_1")[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif "_2" in k:
                name = k.split("_2")[0] + k.split("_2")[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)


class Saliency_feat_encoder_ref(nn.Module):
    # resnet based encoder decoder
    def __init__(self):
        super(Saliency_feat_encoder_ref, self).__init__()
        self.HA = HA()
        self.upsample05 = nn.Upsample(
            scale_factor=0.5, mode="bilinear", align_corners=True
        )

        # self.conv_edge_1 = nn.Conv2d(64, 32, 1)
        # self.conv_edge_2 = nn.Conv2d(512, 32, 1)
        # self.conv_edge_4 = nn.Conv2d(2048, 32, 1)
        # self.conv_edge_11 = nn.Conv2d(32, 32, 3, padding=1)
        # self.conv_edge_21 = nn.Conv2d(512, 32, 1)
        # self.conv_edge_41 = nn.Conv2d(2048, 32, 1)

    def forward(self, resnet, init_pred, x2):
        x2_2 = self.HA(self.upsample05(init_pred).sigmoid(), x2)
        x3_2 = resnet.layer3_2(x2_2)  # 1024 x 16 x 16
        x4_2 = resnet.layer4_2(x3_2)  # 2048 x 8 x 8

        return x2_2, x3_2, x4_2

    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True)
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif "_1" in k:
                name = k.split("_1")[0] + k.split("_1")[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif "_2" in k:
                name = k.split("_2")[0] + k.split("_2")[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)


class _ConvBatchNormReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        relu=True,
    ):
        super(_ConvBatchNormReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
        )
        self.add_module(
            "bn",
            nn.BatchNorm2d(out_channels),
        )

        if relu:
            self.add_module("relu", nn.ReLU())

    def forward(self, x):
        return super(_ConvBatchNormReLU, self).forward(x)


class _ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling with image pool"""

    def __init__(self, in_channels, out_channels, output_stride):
        super(_ASPPModule, self).__init__()
        if output_stride == 8:
            pyramids = [12, 24, 36]
        elif output_stride == 16:
            pyramids = [6, 12, 18]
        self.stages = nn.Module()
        self.stages.add_module(
            "c0", _ConvBatchNormReLU(in_channels, out_channels, 1, 1, 0, 1)
        )
        for i, (dilation, padding) in enumerate(zip(pyramids, pyramids)):
            self.stages.add_module(
                "c{}".format(i + 1),
                _ConvBatchNormReLU(in_channels, out_channels, 3, 1, padding, dilation),
            )
        self.imagepool = nn.Sequential(
            OrderedDict(
                [
                    ("pool", nn.AdaptiveAvgPool2d((1, 1))),
                    ("conv", _ConvBatchNormReLU(in_channels, out_channels, 1, 1, 0, 1)),
                ]
            )
        )
        self.fire = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        _ConvBatchNormReLU(out_channels * 5, out_channels, 3, 1, 1, 1),
                    ),
                    ("dropout", nn.Dropout2d(0.1)),
                ]
            )
        )

    def forward(self, x):
        h = self.imagepool(x)
        h = [F.interpolate(h, size=x.shape[2:], mode="bilinear", align_corners=False)]
        for stage in self.stages.children():
            h += [stage(x)]
        h = torch.cat(h, dim=1)
        h = self.fire(h)
        return h


class Cod_feat_encoder(nn.Module):
    # resnet based encoder decoder
    def __init__(self):
        super(Cod_feat_encoder, self).__init__()
        self.resnet = B2_ResNet()
        # self.aspp = _ASPPModule(2048, 256, 16)
        if self.training:
            self.initialize_weights()

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x0 = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x0)  # 256 x 64 x 64
        x2 = self.resnet.layer2(x1)  # 512 x 32 x 32
        x3 = self.resnet.layer3_1(x2)  # 1024 x 16 x 16
        x4 = self.resnet.layer4_1(x3)  # 2048 x 8 x 8
        # x4 = self.aspp(x4)

        return x1, x2, x3, x4, self.resnet

    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True)
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif "_1" in k:
                name = k.split("_1")[0] + k.split("_1")[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif "_2" in k:
                name = k.split("_2")[0] + k.split("_2")[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)


class Generator(nn.Module):
    def __init__(self, channel=32):
        super(Generator, self).__init__()

        self.sal_encoder = Saliency_feat_encoder()
        self.cod_encoder = Cod_feat_encoder()
        self.shared_decoder1 = Share_feat_decoder(channel)
        self.shared_decoder2 = Share_feat_decoder(channel)
        self.ref_feat1 = Saliency_feat_encoder_ref()
        self.ref_feat2 = Saliency_feat_encoder_ref()
        self.upsample4 = nn.Upsample(
            scale_factor=4, mode="bilinear", align_corners=True
        )
        self.upsample2 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.conv4_cod = self._make_pred_layer(
            Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 2048
        )
        self.conv3_cod = self._make_pred_layer(
            Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 1024
        )
        self.conv4_sod = self._make_pred_layer(
            Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 2048
        )
        self.conv3_sod = self._make_pred_layer(
            Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 1024
        )
        self.rcab_cod = RCAB(2 * channel)
        self.rcab_sod = RCAB(2 * channel)

    def _make_pred_layer(
        self, block, dilation_series, padding_series, NoLabels, input_channel
    ):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, x, sim_img=None, is_sal=False):
        if is_sal:
            if sim_img != None:
                x1, x2, x3, x4, bck_bone = self.sal_encoder(x)
                pred_int = self.shared_decoder1(x1, x2, x3, x4)
                x2_2, x3_2, x4_2 = self.ref_feat1(bck_bone, pred_int, x2)
                pred_ref = self.shared_decoder2(x1, x2_2, x3_2, x4_2)
                x1_sim, x2_sim, x3_sim, x4_sim, _ = self.sal_encoder(sim_img)
                sim_sod = self.shared_decoder1(x1_sim, x2_sim, x3_sim, x4_sim)
                x34_sim_sod = self.upsample4(
                    self.rcab_sod(
                        torch.cat(
                            (
                                self.upsample2(self.conv4_sod(x4_sim)),
                                self.conv3_sod(x3_sim),
                            ),
                            1,
                        )
                    )
                )
                x34_sim_sod = x34_sim_sod * torch.sigmoid(sim_sod)
                x34_sod = x34_sim_sod.mean(dim=(-2, -1))

                x1_sim, x2_sim, x3_sim, x4_sim, _ = self.cod_encoder(sim_img)
                sim_cod = self.shared_decoder1(x1_sim, x2_sim, x3_sim, x4_sim)
                x34_sim_cod = self.upsample4(
                    self.rcab_cod(
                        torch.cat(
                            (
                                self.upsample2(self.conv4_cod(x4_sim)),
                                self.conv3_cod(x3_sim),
                            ),
                            1,
                        )
                    )
                )
                x34_sim_cod = x34_sim_cod * torch.sigmoid(sim_cod)
                x34_cod = x34_sim_cod.mean(dim=(-2, -1))
                # print(x34_cod.size())

                sim_loss = torch.abs(cos_sim(x34_sod, x34_cod)).sum()

                return self.upsample4(pred_int), self.upsample4(pred_ref), sim_loss
            else:
                x1, x2, x3, x4, bck_bone = self.sal_encoder(x)
                pred_int = self.shared_decoder1(x1, x2, x3, x4)
                x2_2, x3_2, x4_2 = self.ref_feat1(bck_bone, pred_int, x2)
                pred_ref = self.shared_decoder2(x1, x2_2, x3_2, x4_2)
                return self.upsample4(pred_int), self.upsample4(pred_ref)
        else:
            x1, x2, x3, x4, bck_bone = self.cod_encoder(x)
            pred_int = self.shared_decoder1(x1, x2, x3, x4)
            x2_2, x3_2, x4_2 = self.ref_feat2(bck_bone, pred_int, x2)
            pred_ref = self.shared_decoder2(x1, x2_2, x3_2, x4_2)
            return self.upsample4(pred_int), self.upsample4(pred_ref)
