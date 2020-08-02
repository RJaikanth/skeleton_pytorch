from .classification import *


CLS_DICT = {
    "alexnet": AlexNet,
    "vgg11": VGG11,
    "vgg11_bn": VGG11_BN,
    "vgg13": VGG13,
    "vgg13_bn": VGG13_BN,
    "vgg16": VGG16,
    "vgg16_bn": VGG16_BN,
    "vgg19": VGG19,
    "vgg19_bn": VGG19_BN,
    "resnet18": ResNet18
    "resnet34": ResNet34
    "resnet50": ResNet50
    "resnet101": ResNet101
    "resnet152": ResNet152
    "resnext50_32x4d": ResNeXt50_32x4d
    "resnext50_32x8d": ResNeXt50_32x8d
    "resnext101_32x8d": ResNeXt101_32x8d
    "wideresnet50_2": WideResNet50_2
    "wideresnet101_2": WideResNet101_2
    "wideresnet152_2": WideResNet152_2
}
