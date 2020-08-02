import torch
import torch.nn as nn


def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, dilation=dilation, bias=False)


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """ Resnet Basic Block """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                "Basic block only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in Basic Block")

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleNeck(nn.Module):
    """ Resnet BottleNeck """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BottleNeck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(out_channels * (base_width/64.)) * groups

        self.conv1 = conv1x1(in_channels, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, out_channels*self.expansion)
        self.bn3 = norm_layer(out_channels*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Fire(nn.Module):
    """ Fire Layer for SqueezeNet """

    def __init__(self, in_channels, squeeze_channels,
                 expand1x1_channels, expand3x3_channels):
        super(Fire, self).__init__()

        self.in_channels = in_channels

        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)

        self.expand1x1 = nn.Conv2d(
            squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)

        self.expand3x3 = nn.Conv2d(
            squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        # print(x.shape)
        out = self.squeeze_activation(self.squeeze(x))
        # print(out.shape)
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(out)),
            self.expand3x3_activation(self.expand3x3(out))
        ], 1)
