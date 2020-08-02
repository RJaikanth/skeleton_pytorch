import torch
from torch import nn
from torch.nn import functional as F


config = {
    # 8 Conv + 3 FC
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    # 10 Conv + 3 FC
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    # 13 Conv + 3 FC
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M",],
    # 16 Conv + 3 FC
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG(nn.Module):
    def __init__(self, cfg, nc=10, batch_norm=False, init_weights=True):
        super(VGG, self).__init__()
        self.nc = nc
        self.conf = config[cfg]
        self.batch_norm = batch_norm

        self.features = make_layers(self.conf, self.batch_norm)
        self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, self.nc),
            nn.Sigmoid(),
        )

        if init_weights:
            self.__init_weights()

    def forward(self, image):
        out = self.features(image)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out

    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(tensor=m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(tensor=m.weight, val=1)
                nn.init.constant_(tensor=m.bias, val=0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(tensor=m.weight, mean=0, std=0.01)
                nn.init.constant_(tensor=m.bias, val=0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def VGG11(init_weights=True, num_classes=10):
    return VGG("VGG11", init_weights=init_weights, nc=num_classes)


def VGG13(init_weights=True, num_classes=10):
    return VGG("VGG13", init_weights=init_weights, nc=num_classes)


def VGG16(init_weights=True, num_classes=10):
    return VGG("VGG16", init_weights=init_weights, nc=num_classes)


def VGG19(init_weights=True, num_classes=10):
    return VGG("VGG19", init_weights=init_weights, nc=num_classes)


def VGG11_BN(init_weights=True, num_classes=10):
    return VGG("VGG11", batch_norm=True, init_weights=init_weights, nc=num_classes)


def VGG13_BN(init_weights=True, num_classes=10):
    return VGG("VGG13", batch_norm=True, init_weights=init_weights, nc=num_classes)


def VGG16_BN(init_weights=True, num_classes=10):
    return VGG("VGG16", batch_norm=True, init_weights=init_weights, nc=num_classes)


def VGG19_BN(init_weights=True, num_classes=10):
    return VGG("VGG19", batch_norm=True, init_weights=init_weights, nc=num_classes)

