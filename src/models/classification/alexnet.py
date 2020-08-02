import torch
from torch import nn
from torch.nn import functional as F


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=True):
        super(AlexNet, self).__init__()

        self.nc = num_classes

        # Block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Block 2
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(192)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Block 3
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(384)
        self.relu3 = nn.ReLU(inplace=True)

        # Block 4
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)

        # Block 5
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU(inplace=True)

        self.bn = nn.BatchNorm2d(256)

        # Avg Pool
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        # FC
        self.drop1 = nn.Dropout()
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout()
        self.fc2 = nn.Linear(4096, 4096)
        self.relu7 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(4096, self.nc)

        if init_weights:
            self.__init_weights()

    def forward(self, x):
        # Block 1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)

        # Block 2
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)

        # Block 3
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)

        # Block 4
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu4(out)

        # Block 5
        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu5(out)

        # FC
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.drop1(out)
        out = self.fc1(out)
        out = self.relu6(out)
        out = self.drop2(out)
        out = self.fc2(out)
        out = self.relu7(out)
        out = self.fc3(out)

        return out

    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(tensor=m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(tensor=m.bias, val=0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(tensor=m.weight, val=1)
                nn.init.constant_(tensor=m.bias, val=0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(tensor=m.weight, mean=0, std=0.01)
                nn.init.constant_(tensor=m.bias, val=0)

