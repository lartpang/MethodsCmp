# -*- coding: utf-8 -*-
# @Time    : 2021/8/6
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from .base_modules import PAFEM


class DANet_V19(nn.Module):
    def __init__(self):
        super(DANet_V19, self).__init__()
        feats = list(models.vgg19_bn(pretrained=True).features.children())
        self.conv0 = nn.Conv2d(4, 64, kernel_size=3, padding=1)
        self.conv1 = nn.Sequential(*feats[1:6])
        self.conv2 = nn.Sequential(*feats[6:13])
        self.conv3 = nn.Sequential(*feats[13:26])
        self.conv4 = nn.Sequential(*feats[26:39])
        self.conv5 = nn.Sequential(*feats[39:52])

        # PAFEM
        self.dem1 = PAFEM(512, 512)
        # vanilla convolution
        # self.dem1 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.PReLU())

        self.dem2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
        )
        self.dem3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
        )
        self.dem4 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        self.dem5 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.PReLU())

        self.fuse_1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 1, kernel_size=3, padding=1),
        )
        self.fuse_2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=3, padding=1),
        )
        self.fuse_3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )
        self.fuse_4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )
        self.fuse_5 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

        self.output1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
        )
        self.output1_rev = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
        )
        self.output2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
        )
        self.output2_rev = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
        )
        self.output3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        self.output3_rev = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        self.output4 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.PReLU())
        self.output4_rev = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.PReLU())
        self.output5 = nn.Sequential(nn.Conv2d(32, 1, kernel_size=3, padding=1))
        self.output5_rev = nn.Sequential(nn.Conv2d(32, 1, kernel_size=3, padding=1))
        self.fuseout = nn.Sequential(nn.Conv2d(2, 1, kernel_size=3, padding=1), nn.PReLU())

        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True

    def forward(self, data):
        image = data["image"]
        depth = data["depth"]
        input = image
        B, _, _, _ = input.size()
        c0 = self.conv0(torch.cat((image, depth), 1))
        c1 = self.conv1(c0)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        ################################PAFEM#######################################
        dem1 = self.dem1(c5)
        ############################################################################
        dem2 = self.dem2(c4)
        dem3 = self.dem3(c3)
        dem4 = self.dem4(c2)
        dem5 = self.dem5(c1)
        ################################DAM for Saliency branch&Background branch#######################################
        dem1_attention = torch.sigmoid(
            self.fuse_1(dem1 + F.interpolate(depth, size=dem1.size()[2:], mode="bilinear", align_corners=False))
        )
        output1 = self.output1(
            dem1
            * (
                dem1_attention
                * (
                    F.interpolate(
                        depth,
                        size=dem1.size()[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    + dem1_attention
                )
            )
        )
        output1_rev = self.output1_rev(
            dem1
            * (
                (1 - dem1_attention)
                * (
                    F.interpolate(
                        depth,
                        size=dem1.size()[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    + (1 - dem1_attention)
                )
            )
        )

        dem2_attention = torch.sigmoid(
            self.fuse_2(
                dem2
                + F.interpolate(output1, size=dem2.size()[2:], mode="bilinear", align_corners=False)
                + F.interpolate(depth, size=dem2.size()[2:], mode="bilinear", align_corners=False)
            )
        )
        output2 = self.output2(
            F.interpolate(output1, size=dem2.size()[2:], mode="bilinear", align_corners=False)
            + dem2
            * (
                dem2_attention
                * (
                    F.interpolate(
                        depth,
                        size=dem2.size()[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    + dem2_attention
                )
            )
        )
        output2_rev = self.output2_rev(
            F.interpolate(output1_rev, size=dem2.size()[2:], mode="bilinear", align_corners=False)
            + dem2
            * (
                (1 - dem2_attention)
                * (
                    F.interpolate(
                        depth,
                        size=dem2.size()[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    + (1 - dem2_attention)
                )
            )
        )

        dem3_attention = torch.sigmoid(
            self.fuse_3(
                dem3
                + F.interpolate(output2, size=dem3.size()[2:], mode="bilinear", align_corners=False)
                + F.interpolate(depth, size=dem3.size()[2:], mode="bilinear", align_corners=False)
            )
        )
        output3 = self.output3(
            F.interpolate(output2, size=dem3.size()[2:], mode="bilinear", align_corners=False)
            + dem3
            * (
                dem3_attention
                * (
                    F.interpolate(
                        depth,
                        size=dem3.size()[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    + dem3_attention
                )
            )
        )
        output3_rev = self.output3_rev(
            F.interpolate(output2_rev, size=dem3.size()[2:], mode="bilinear", align_corners=False)
            + dem3
            * (
                (1 - dem3_attention)
                * (
                    F.interpolate(
                        depth,
                        size=dem3.size()[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    + (1 - dem3_attention)
                )
            )
        )

        dem4_attention = torch.sigmoid(
            self.fuse_4(
                dem4
                + F.interpolate(output3, size=dem4.size()[2:], mode="bilinear", align_corners=False)
                + F.interpolate(depth, size=dem4.size()[2:], mode="bilinear", align_corners=False)
            )
        )
        output4 = self.output4(
            F.interpolate(output3, size=dem4.size()[2:], mode="bilinear", align_corners=False)
            + dem4
            * (
                dem4_attention
                * (
                    F.interpolate(
                        depth,
                        size=dem4.size()[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    + dem4_attention
                )
            )
        )
        output4_rev = self.output4_rev(
            F.interpolate(output3_rev, size=dem4.size()[2:], mode="bilinear", align_corners=False)
            + dem4
            * (
                (1 - dem4_attention)
                * (
                    F.interpolate(
                        depth,
                        size=dem4.size()[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    + (1 - dem4_attention)
                )
            )
        )

        dem5_attention = torch.sigmoid(
            self.fuse_5(
                dem5
                + F.interpolate(output4, size=dem5.size()[2:], mode="bilinear", align_corners=False)
                + F.interpolate(depth, size=dem5.size()[2:], mode="bilinear", align_corners=False)
            )
        )
        output5 = self.output5(
            F.interpolate(output4, size=dem5.size()[2:], mode="bilinear", align_corners=False)
            + dem5
            * (
                dem5_attention
                * (
                    F.interpolate(
                        depth,
                        size=dem5.size()[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    + dem5_attention
                )
            )
        )
        output5_rev = self.output5_rev(
            F.interpolate(output4_rev, size=dem5.size()[2:], mode="bilinear", align_corners=False)
            + dem5
            * (
                (1 - dem5_attention)
                * (
                    F.interpolate(
                        depth,
                        size=dem5.size()[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    + (1 - dem5_attention)
                )
            )
        )

        ################################Dual Branch Fuse#######################################
        output5 = F.interpolate(output5, size=input.size()[2:], mode="bilinear", align_corners=False)
        output5_rev = F.interpolate(output5_rev, size=input.size()[2:], mode="bilinear", align_corners=False)
        output = self.fuseout(torch.cat((output5, -output5_rev), 1))
        output = -output5_rev + output

        if self.training:
            return (
                output,
                output5,
                output5_rev,
                dem1_attention,
                dem2_attention,
                dem3_attention,
                dem4_attention,
                dem5_attention,
            )
        return torch.sigmoid(output)
