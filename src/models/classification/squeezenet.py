import torch
import torch.nn as nn

from .custom_layers import Fire


class SqueezeNet(nn.Module):
    def __init__(self, version='1.0', num_classes=1000):
        super(SqueezeNet, self).__init__()

        self.num_classes = num_classes
        if version == '1.0':
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256)
            )

        elif version == '1.1':
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )

        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.features(x)
        out = self.classifier(out)
        return torch.flatten(out, 1)


def SqueezeNet1_0(num_classes):
    return SqueezeNet(num_classes=num_classes)


def SqueezeNet1_1(num_classes):
    return SqueezeNet(version="1.1", num_classes=num_classes)


if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter("temp/sn1")
    writer.add_graph(SqueezeNet1_0(10), torch.zeros([32, 3, 64, 64]))
    writer.close()
    print("sq1 Done")

    writer = SummaryWriter("temp/sn2")
    writer.add_graph(SqueezeNet1_1(10), torch.zeros([32, 3, 64, 64]))
    writer.close()
    print("sq1.1 Done")
