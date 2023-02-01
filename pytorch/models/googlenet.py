import torch
from torch import nn
from torchsummary import summary
from pytorch.models.base import BaseModel

class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(_ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class _InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels_arr):
        super(_InceptionBlock, self).__init__()

        self.branch1 = _ConvBlock(in_channels, out_channels_arr[0], kernel_size=1)
        
        self.branch2 = nn.Sequential(
            _ConvBlock(in_channels, out_channels_arr[1][0], kernel_size=1),
            _ConvBlock(out_channels_arr[1][0], out_channels_arr[1][1], kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            _ConvBlock(in_channels, out_channels_arr[2][0], kernel_size=1),
            _ConvBlock(out_channels_arr[2][0], out_channels_arr[2][1], kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            _ConvBlock(in_channels, out_channels_arr[3], kernel_size=1)
        )
    
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat((branch1, branch2, branch3, branch4), 1)


class GoogLeNet(BaseModel):
    def __init__(self, n_classes=None, dropout=None):
        super(GoogLeNet, self).__init__()

        self.features = self.__build_features()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = self.__build_classifier(n_classes, dropout)

        self.init_weights()

    def __build_features(self):
        layers = []

        # head
        layers.extend([
            _ConvBlock(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            _ConvBlock(64, 64, kernel_size=1, stride=1, padding=0),
            _ConvBlock(64, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])

        # inception blocks
        layers.extend([
            _InceptionBlock(in_channels=192, out_channels_arr=[64, (96, 128), (16, 32), 32]),
            _InceptionBlock(in_channels=256, out_channels_arr=[128, (128, 192), (32, 96), 64]),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            _InceptionBlock(in_channels=480, out_channels_arr=[192, (96, 208), (16, 48), 64]),
            _InceptionBlock(in_channels=512, out_channels_arr=[160, (112, 224), (24, 64), 64]),
            _InceptionBlock(in_channels=512, out_channels_arr=[128, (128, 256), (24, 64), 64]),
            _InceptionBlock(in_channels=512, out_channels_arr=[112, (144, 288), (32, 64), 64]),
            _InceptionBlock(in_channels=528, out_channels_arr=[256, (160, 320), (32, 128), 128]),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            _InceptionBlock(in_channels=832, out_channels_arr=[256, (160, 320), (32, 128), 128]),
            _InceptionBlock(in_channels=832, out_channels_arr=[384, (192, 384), (48, 128), 128]),
        ])

        return nn.Sequential(*layers)

    def __build_classifier(self, n_classes, dropout):
        layers = []
        if dropout is not None:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(1024, n_classes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)


if __name__ == '__main__':
    x = GoogLeNet(n_classes=1000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    summary(x, (3, 224, 224))