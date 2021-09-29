from torch import nn
from torchvision.models import resnet50


def Backbone_ResNet50_in3(pretrained=True):
    if pretrained:
        print("The backbone model loads the pretrained parameters...")
    net = resnet50(pretrained=pretrained)
    div_2 = nn.Sequential(*list(net.children())[:3])
    div_4 = nn.Sequential(*list(net.children())[3:5])
    div_8 = net.layer2
    div_16 = net.layer3
    div_32 = net.layer4

    return div_2, div_4, div_8, div_16, div_32


def Backbone_ResNet50_in1(pretrained=True):
    if pretrained:
        print("The backbone model loads the pretrained parameters...")
    net = resnet50(pretrained=pretrained)
    net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    div_2 = nn.Sequential(*list(net.children())[:3])
    div_4 = nn.Sequential(*list(net.children())[3:5])
    div_8 = net.layer2
    div_16 = net.layer3
    div_32 = net.layer4

    return div_2, div_4, div_8, div_16, div_32
