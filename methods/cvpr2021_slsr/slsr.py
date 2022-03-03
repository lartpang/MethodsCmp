import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import Parameter, Softmax

from .resnet import B2_ResNet


class CAM_Module(nn.Module):
    """Channel attention module"""

    # paper: Dual Attention Network for Scene Segmentation
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)
        bn = False
        modules_body = []
        for i in range(2):
            modules_body.append(self.default_conv(in_dim, in_dim, 3, bias=True))
            if bn:
                modules_body.append(nn.BatchNorm2d(in_dim))
            if i == 0:
                modules_body.append(nn.ReLU(True))
        self.body = nn.Sequential(*modules_body)

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size // 2),
            bias=bias,
        )

    def forward(self, x):
        x = self.body(x)
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
        self,
        n_feat,
        kernel_size=3,
        reduction=16,
        bias=True,
        bn=False,
        res_scale=1,
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


class Saliency_feat_decoder(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel):
        super(Saliency_feat_decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.upsample8 = nn.Upsample(
            scale_factor=8, mode="bilinear", align_corners=True
        )
        self.upsample4 = nn.Upsample(
            scale_factor=4, mode="bilinear", align_corners=True
        )
        self.upsample2 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.upsample05 = nn.Upsample(
            scale_factor=0.5, mode="bilinear", align_corners=True
        )
        self.dropout = nn.Dropout(0.3)
        self.layer6 = self._make_pred_layer(
            Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, channel * 4
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
        self.conv4321 = self._make_pred_layer(
            Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 4 * channel
        )

        self.cls_layer = self._make_pred_layer(
            Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, channel * 4
        )
        self.pam2 = PAM_Module(channel)
        self.pam3 = PAM_Module(channel)
        self.pam4 = PAM_Module(channel)
        self.cam2 = CAM_Module(channel)
        self.cam3 = CAM_Module(channel)
        self.cam4 = CAM_Module(channel)

    def _make_pred_layer(
        self, block, dilation_series, padding_series, NoLabels, input_channel
    ):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, x1, x2, x3, x4):
        conv1_feat = self.conv1(x1)
        conv2_feat = self.conv2(x2)
        conv2_feat1 = self.pam2(conv2_feat)
        conv2_feat2 = self.cam2(conv2_feat)
        conv2_feat = conv2_feat + conv2_feat1 + conv2_feat2
        conv3_feat = self.conv3(x3)
        conv3_feat1 = self.pam3(conv3_feat)
        conv3_feat2 = self.cam3(conv3_feat)
        conv3_feat = conv3_feat + conv3_feat1 + conv3_feat2
        conv4_feat = self.conv4(x4)
        conv4_feat1 = self.pam4(conv4_feat)
        conv4_feat2 = self.cam4(conv4_feat)
        conv4_feat = conv4_feat + conv4_feat1 + conv4_feat2
        conv4_feat = self.upsample2(conv4_feat)

        conv43 = torch.cat((conv4_feat, conv3_feat), 1)
        conv43 = self.racb_43(conv43)
        conv43 = self.conv43(conv43)

        conv43 = self.upsample2(conv43)
        conv432 = torch.cat((self.upsample2(conv4_feat), conv43, conv2_feat), 1)
        conv432 = self.racb_432(conv432)
        conv432 = self.conv432(conv432)
        conv432 = self.upsample2(conv432)
        conv4321 = torch.cat(
            (self.upsample4(conv4_feat), self.upsample2(conv43), conv432, conv1_feat), 1
        )
        conv4321 = self.racb_4321(conv4321)

        sal_pred = self.cls_layer(conv4321)

        return sal_pred


class Saliency_feat_encoder(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel):
        super(Saliency_feat_encoder, self).__init__()
        self.resnet = B2_ResNet()
        self.relu = nn.ReLU(inplace=True)
        self.upsample8 = nn.Upsample(
            scale_factor=8, mode="bilinear", align_corners=True
        )
        self.upsample4 = nn.Upsample(
            scale_factor=4, mode="bilinear", align_corners=True
        )
        self.upsample2 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.upsample05 = nn.Upsample(
            scale_factor=0.5, mode="bilinear", align_corners=True
        )
        self.dropout = nn.Dropout(0.3)
        self.sal_dec = Saliency_feat_decoder(channel)

        if self.training:
            self.initialize_weights()

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)  # 256 x 64 x 64
        x2 = self.resnet.layer2(x1)  # 512 x 32 x 32
        x3 = self.resnet.layer3_1(x2)  # 1024 x 16 x 16
        x4 = self.resnet.layer4_1(x3)  # 2048 x 8 x 8

        sal_init = self.sal_dec(x1, x2, x3, x4)

        return self.upsample4(sal_init)

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
    def __init__(self, channel):
        super(Generator, self).__init__()
        self.sal_encoder = Saliency_feat_encoder(channel)

    def forward(self, x):
        self.sal_init_post = self.sal_encoder(x)
        self.sal_init_post = F.upsample(
            self.sal_init_post,
            size=(x.shape[2], x.shape[3]),
            mode="bilinear",
            align_corners=True,
        )
        return self.sal_init_post
