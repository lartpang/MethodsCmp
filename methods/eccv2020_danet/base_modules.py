import torch
import torch.nn as nn
from torch.nn import Conv2d, functional as F, Parameter, Softmax


###ECCV2020 A Single Stream Network for Robust and Real-time RGB-D Salient Object Detection
class PAFEM(nn.Module):
    def __init__(self, dim, in_dim):
        super(PAFEM, self).__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(dim, in_dim, 3, padding=1), nn.BatchNorm2d(in_dim), nn.PReLU()
        )
        self.down_dim = in_dim // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, self.down_dim, kernel_size=1),
            nn.BatchNorm2d(self.down_dim),
            nn.PReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, self.down_dim, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm2d(self.down_dim),
            nn.PReLU(),
        )
        self.query_conv2 = Conv2d(
            in_channels=self.down_dim, out_channels=self.down_dim // 8, kernel_size=1
        )
        self.key_conv2 = Conv2d(
            in_channels=self.down_dim, out_channels=self.down_dim // 8, kernel_size=1
        )
        self.value_conv2 = Conv2d(
            in_channels=self.down_dim, out_channels=self.down_dim, kernel_size=1
        )
        self.gamma2 = Parameter(torch.zeros(1))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim, self.down_dim, kernel_size=3, dilation=4, padding=4),
            nn.BatchNorm2d(self.down_dim),
            nn.PReLU(),
        )
        self.query_conv3 = Conv2d(
            in_channels=self.down_dim, out_channels=self.down_dim // 8, kernel_size=1
        )
        self.key_conv3 = Conv2d(
            in_channels=self.down_dim, out_channels=self.down_dim // 8, kernel_size=1
        )
        self.value_conv3 = Conv2d(
            in_channels=self.down_dim, out_channels=self.down_dim, kernel_size=1
        )
        self.gamma3 = Parameter(torch.zeros(1))

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, self.down_dim, kernel_size=3, dilation=6, padding=6),
            nn.BatchNorm2d(self.down_dim),
            nn.PReLU(),
        )
        self.query_conv4 = Conv2d(
            in_channels=self.down_dim, out_channels=self.down_dim // 8, kernel_size=1
        )
        self.key_conv4 = Conv2d(
            in_channels=self.down_dim, out_channels=self.down_dim // 8, kernel_size=1
        )
        self.value_conv4 = Conv2d(
            in_channels=self.down_dim, out_channels=self.down_dim, kernel_size=1
        )
        self.gamma4 = Parameter(torch.zeros(1))

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, self.down_dim, kernel_size=1),
            nn.BatchNorm2d(self.down_dim),
            nn.PReLU()
            # 如果batch=1 ，进行batchnorm会有问题
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(5 * self.down_dim, in_dim, kernel_size=1),
            nn.BatchNorm2d(in_dim),
            nn.PReLU(),
        )
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        x = self.down_conv(x)

        # Branch 1
        conv1 = self.conv1(x)
        B, C, H, W = conv1.size()

        # Branch 2
        conv2 = self.conv2(x)
        proj_query2 = self.query_conv2(conv2).reshape(B, -1, W * H).permute(0, 2, 1)
        proj_key2 = self.key_conv2(conv2).reshape(B, -1, W * H)
        attention2 = self.softmax(torch.bmm(proj_query2, proj_key2))

        proj_value2 = self.value_conv2(conv2).reshape(B, -1, W * H)
        out2 = torch.bmm(proj_value2, attention2.permute(0, 2, 1))
        out2 = self.gamma2 * out2.reshape(B, C, H, W) + conv2

        # Branch 3
        conv3 = self.conv3(x)
        proj_query3 = self.query_conv3(conv3).reshape(B, -1, W * H).permute(0, 2, 1)
        proj_key3 = self.key_conv3(conv3).reshape(B, -1, W * H)
        attention3 = self.softmax(torch.bmm(proj_query3, proj_key3))

        proj_value3 = self.value_conv3(conv3).reshape(B, -1, W * H)
        out3 = torch.bmm(proj_value3, attention3.permute(0, 2, 1))
        out3 = self.gamma3 * out3.reshape(B, C, H, W) + conv3

        # Branch 4
        conv4 = self.conv4(x)
        proj_query4 = self.query_conv4(conv4).reshape(B, -1, W * H).permute(0, 2, 1)
        proj_key4 = self.key_conv4(conv4).reshape(B, -1, W * H)
        attention4 = self.softmax(torch.bmm(proj_query4, proj_key4))

        proj_value4 = self.value_conv4(conv4).reshape(B, -1, W * H)
        out4 = torch.bmm(proj_value4, attention4.permute(0, 2, 1))
        out4 = self.gamma4 * out4.reshape(B, C, H, W) + conv4

        # Branch 5
        conv5 = F.interpolate(
            self.conv5(F.adaptive_avg_pool2d(x, 1)),
            size=x.size()[2:],
            mode="bilinear",
            align_corners=False,
        )
        # 如果batch设为1，这里就会有问题。

        return self.fuse(torch.cat((conv1, out2, out3, out4, conv5), 1))
