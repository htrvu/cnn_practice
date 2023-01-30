import torch
from torch import nn
from base import BaseModel
from torchsummary import summary


class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(_ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x


class _DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dw_stride, alpha=1.0):
        super(_DepthwiseSeparableBlock, self).__init__()

        # depthwise
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                               stride=dw_stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # pointwise
        out_channels = int(out_channels * alpha)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x

class MobileNet(BaseModel):
    def __init__(self, n_classes=1000, alpha=1.0, dropout=None):
        super(MobileNet, self).__init__()

        self.features = self.__build_features(alpha)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = self.__build_classifier(n_classes, dropout)

        self.init_weights()

    def __build_features(self, alpha):
        layers = []

        # head
        self.in_channels = int(alpha * 32)
        layers.append(_ConvBlock(3, self.in_channels, kernel_size=3, stride=2, padding=1))

        # depthwise separable blocks
        cfg = [
            # dw_stride, out_channels
            [1, 64],
            [2, 128],
            [1, 128],
            [2, 256],
            [1, 256],
            [2, 512],
            [1, 512],
            [1, 512],
            [1, 512],
            [1, 512],
            [1, 512],
            [2, 1024],
            [1, 1024],
        ]
        for dw_stride, out_channels in cfg:
            layers.append(_DepthwiseSeparableBlock(self.in_channels, out_channels, dw_stride, alpha))
            self.in_channels = int(out_channels * alpha)

        return nn.Sequential(*layers)

    def __build_classifier(self, n_classes, dropout):
        layers = []
        if dropout is not None:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(self.in_channels, n_classes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)


if __name__ == '__main__':
    x = MobileNet(n_classes=1000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    summary(x, (3, 224, 224))


