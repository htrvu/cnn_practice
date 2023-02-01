import torch
from torch import nn
from pytorch.models.base import BaseModel
from torchsummary import summary


class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, activation=nn.ReLU):
        super(_ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = activation(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x


class _InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dw_stride, expansion_ratio=1.0):
        super(_InvertedResidualBlock, self).__init__()

        self.use_skip_connect = dw_stride == 1 and in_channels == out_channels

        layers = []
        hidden_dims = int(in_channels * expansion_ratio)

        # expand if expansion_ratio != 1.0
        if expansion_ratio != 1.0:
            layers.append(_ConvBlock(in_channels, hidden_dims,
                                     kernel_size=1, stride=1, padding=0))

        # depthwise
        layers.append(_ConvBlock(hidden_dims, hidden_dims, kernel_size=3,
                                 stride=dw_stride, padding=1, groups=hidden_dims, activation=nn.ReLU6))

        # pointwise (linear bottleneck)
        layers.extend([
            nn.Conv2d(hidden_dims, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_skip_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(BaseModel):
    def __init__(self, n_classes=1000, alpha=1.0, dropout=None):
        super(MobileNetV2, self).__init__()

        self.features = self.__build_features(alpha)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = self.__build_classifier(n_classes, dropout)

        self.init_weights()

    def __build_features(self, alpha):
        layers = []

        # head
        self.in_channels = int(alpha * 32)
        layers.append(_ConvBlock(3, self.in_channels, kernel_size=3,
                                 stride=2, padding=1, activation=nn.ReLU6))

        # inverted residual blocks
        cfg = [
            # expansion_ratio (t), out_channels (c), repeated times (n), first_dw_stride (s)
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        for expansion_ratio, out_channels, repeated_times, first_dw_stride in cfg:
            out_channels = int(out_channels * alpha)
            for i in range(repeated_times):
                dw_stride = first_dw_stride if i == 0 else 1
                layers.append(_InvertedResidualBlock(self.in_channels, out_channels, dw_stride, expansion_ratio))
                self.in_channels = out_channels

        # last conv layer
        self.last_channels = int(1280 * alpha)
        layers.append(_ConvBlock(self.in_channels, self.last_channels, kernel_size=1, stride=1, padding=0, activation=nn.ReLU6))

        return nn.Sequential(*layers)

    def __build_classifier(self, n_classes, dropout):
        layers = []
        if dropout is not None:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(self.last_channels, n_classes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    x = MobileNetV2(n_classes=1000, alpha=1.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    summary(x, (3, 224, 224))
