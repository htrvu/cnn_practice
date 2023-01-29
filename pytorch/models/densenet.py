import torch
from torch import nn
from base import BaseModel
from torchsummary import summary


class _BottleneckLayers(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(_BottleneckLayers, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

class _DenseBlock(nn.Module):
    def __init__(self, in_channels, n_bn_layers, growth_rate):
        super(_DenseBlock, self).__init__()
        layers = []
        for i in range(n_bn_layers):
            layers.append(_BottleneckLayers(in_channels + i * growth_rate, growth_rate))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        features = x
        for layer in self.layers.children():
            new_x = layer(features)
            features = torch.cat((features, new_x), 1)
        return features


class _TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x


CONFIGS = {
    'densenet121': {
        'n_bn_layers': [6, 12, 24, 16],
        'growth_rate': 32,
        'theta': 0.5
    },
    'densenet169': {
        'n_bn_layers': [6, 12, 32, 32],
        'growth_rate': 32,
        'theta': 0.5
    },
    'densenet201': {
        'n_bn_layers': [6, 12, 48, 32],
        'growth_rate': 32,
        'theta': 0.5
    },
    'densenet264': {
        'n_bn_layers': [6, 12, 64, 48],
        'growth_rate': 32,
        'theta': 0.5
    }
}


class _DenseNet(BaseModel):
    def __init__(self, model_name='densenet121', n_classes=1000, dropout=0.5):
        super(_DenseNet, self).__init__()

        self.features = self.__build_features(model_name)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = self.__build_classifier(n_classes, dropout)

        self.init_weights()

    def __build_features(self, model_name):
        cfg = CONFIGS[model_name]

        n_bn_layers, growth_rate, theta = cfg['n_bn_layers'], cfg['growth_rate'], cfg['theta']
        layers = []

        # head
        layers.extend([
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        ])

        # dense blocks and transition
        self.in_channels = 64
        for i, n_bn_layer in enumerate(n_bn_layers):
            layers.append(_DenseBlock(self.in_channels, n_bn_layer, growth_rate))
            self.in_channels += n_bn_layer * growth_rate
            if i != len(n_bn_layers) - 1:
                layers.append(_TransitionLayer(self.in_channels, int(theta * self.in_channels)))
                self.in_channels = int(theta * self.in_channels)

        # final batch norm and relu
        layers.append(nn.BatchNorm2d(self.in_channels))
        layers.append(nn.ReLU(inplace=True))

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

def DenseNet121(n_classes=1000, dropout=None):
    return _DenseNet('densenet121', n_classes=n_classes, dropout=dropout)

def DenseNet169(n_classes=1000, dropout=None):
    return _DenseNet('densenet169', n_classes=n_classes, dropout=dropout)

def DenseNet201(n_classes=1000, dropout=None):
    return _DenseNet('densenet201', n_classes=n_classes, dropout=dropout)

def DenseNet264(n_classes=1000, dropout=None):
    return _DenseNet('densenet264', n_classes=n_classes, dropout=dropout)


if __name__ == '__main__':
    x = DenseNet121(n_classes=1000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    summary(x, (3, 224, 224))