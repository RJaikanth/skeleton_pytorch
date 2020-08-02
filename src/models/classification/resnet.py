import torch
import torch.nn as nn

from .custom_layers import BasicBlock, BottleNeck, conv1x1


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=True,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.in_channels = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.conv1 = nn.Conv2d(3, self.in_channels,
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleNeck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels *
                        block.expansion, stride),
                norm_layer(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

    def forward(self, x):
        return self._forward_impl(x)


def ResNet18(num_classes=1000):
    return ResNet(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes)


def ResNet34(num_classes=1000):
    return ResNet(block=BasicBlock, layers=[3, 4, 6, 3], num_classes=num_classes)


def ResNet50(num_classes=1000):
    return ResNet(block=BottleNeck, layers=[2, 2, 2, 2], num_classes=num_classes)


def ResNet101(num_classes=1000):
    return ResNet(block=BottleNeck, layers=[3, 4, 23, 3], num_classes=num_classes)


def ResNet152(num_classes=1000):
    return ResNet(block=BottleNeck, layers=[3, 8, 86, 3], num_classes=num_classes)


def ResNeXt50_32x4d(num_classes=1000):
    return ResNet(block=BottleNeck, layers=[3, 4, 6, 3], groups=32, width_per_group=4, num_classes=num_classes)


def ResNeXt50_32x8d(num_classes=1000):
    return ResNet(block=BottleNeck, layers=[3, 4, 6, 3], groups=32, width_per_group=8, num_classes=num_classes)


def ResNeXt101_32x8d(num_classes=1000):
    return ResNet(block=BottleNeck, layers=[3, 4, 23, 3], groups=32, width_per_group=8, num_classes=num_classes)


def WideResNet50_2(num_classes=1000):
    return ResNet(BottleNeck, layers=[3, 4, 6, 3], width_per_group=64*2)


def WideResNet101_2(num_classes=1000):
    return ResNet(BottleNeck, layers=[3, 4, 23, 3], width_per_group=64*2)


def WideResNet152_2(num_classes=1000):
    return ResNet(BottleNeck, layers=[3, 8, 86, 3], width_per_group=64*2)


if __name__ == "__main__":
    import torchsummary
    from torch.utils.tensorboard import SummaryWriter

    model_path = "logs/tensorboard/models/"

    base_dict = {
        "BasicBlock": BasicBlock(3, 64),
        "Bottleneck": BottleNeck(3, 64)
    }

    resnet_dict = {
        "ResNet18": ResNet18(10),
        "ResNet34": ResNet34(10),
        "ResNet50": ResNet50(10),
        "ResNet101": ResNet101(10),
        "ResNet152": ResNet152(10),
        "ResNet50_32x4d": ResNet50_32x4d(10),
        "ResNet50_32x8d": ResNet50_32x8d(10),
        "ResNet101_32x8d": ResNet101_32x8d(10),
        "WideResNet50_2": WideResNet50_2(10),
        "WideResNet101_2": WideResNet101_2(10)
    }

    # for k, v in base_dict.items():
    #     writer = SummaryWriter("{}{}".format(model_path, k))
    #     writer.add_graph(v, torch.zeros([32, 3, 64, 64]))
    #     writer.close()
    #     print("{} Done".format(k))

    for k, v in resnet_dict.items():
        writer = SummaryWriter("{}{}".format(model_path, k))
        writer.add_graph(v, torch.zeros([32, 3, 64, 64]))
        writer.close()
        print("{} Done".format(k))
