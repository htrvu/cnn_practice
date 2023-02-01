import torch
from torch import nn
from pytorch.models.base import BaseModel
from torchsummary import summary

class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, filters, first_stride):
        super(_BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, filters, kernel_size=3, padding=1, stride=first_stride, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)

        self.projection = None
        if in_channels != filters:
            self.projection = nn.Sequential(
                nn.Conv2d(in_channels, filters, kernel_size=1, stride=first_stride, bias=False),
                nn.BatchNorm2d(filters)
            )
        
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # projection skip connection
        if self.projection is not None:
            identity = self.projection(identity)

        x += identity
        x = self.relu(x)

        return x

class _BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, filters, first_stride):
        super(_BottleneckBlock, self).__init__()

        # we will calculate out_channels based on filters
        self.conv1 = nn.Conv2d(in_channels, filters, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=first_stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)
        
        self.conv3 = nn.Conv2d(filters, self.expansion * filters, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * filters)

        self.projection = None
        if in_channels != self.expansion * filters:
            self.projection = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * filters, kernel_size=1, stride=first_stride, bias=False),
                nn.BatchNorm2d(self.expansion * filters)
            )
        
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        
        # projection skip connection
        if self.projection is not None:
            identity = self.projection(identity)
        
        x += identity
        x = self.relu(x)
        
        return x


CONFIGS = {
    'resnet18': {
        'n_blocks': [2, 2, 2, 2],
        'block_class': _BasicBlock
    },
    'resnet34': {
        'n_blocks': [3, 4, 6, 3],
        'block_class': _BasicBlock
    },
    'resnet50': {
        'n_blocks': [3, 4, 6, 3],
        'block_class': _BottleneckBlock
    },
    'resnet101': {
        'n_blocks': [3, 4, 23, 3],
        'block_class': _BottleneckBlock
    },
    'resnet152': {
        'n_blocks': [3, 8, 36, 3],
        'block_class': _BottleneckBlock
    }
}

class _ResNet(BaseModel):
    def __init__(self, model_name='resnet18', n_classes=1000, dropout=0.5):
        super(_ResNet, self).__init__()

        self.features = self.__build_features(model_name)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = self.__build_classifier(n_classes, dropout)

        self.init_weights()

    def __build_features(self, model_name):
        cfg = CONFIGS[model_name]
        layers = []

        # head
        layers.extend([
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        ])

        # blocks
        self.in_channels = 64
        layers1 = self.__create_layers(cfg['block_class'], cfg['n_blocks'][0], filters=64, first_stride=1)
        layers2 = self.__create_layers(cfg['block_class'], cfg['n_blocks'][1], filters=128, first_stride=2)
        layers3 = self.__create_layers(cfg['block_class'], cfg['n_blocks'][2], filters=256, first_stride=2)
        layers4 = self.__create_layers(cfg['block_class'], cfg['n_blocks'][3], filters=512, first_stride=2)
        layers.extend([layers1, layers2, layers3, layers4])

        return nn.Sequential(*layers)

    def __build_classifier(self, n_classes, dropout):
        layers = []
        if dropout is not None:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(self.in_channels, n_classes))
        return nn.Sequential(*layers)

    def __create_layers(self, block_class, n_blocks, filters, first_stride):
        layers = []
        layers.append(block_class(self.in_channels, filters, first_stride))

        self.in_channels = block_class.expansion * filters
        for i in range(1, n_blocks):
            layers.append(block_class(self.in_channels, filters, 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)


def ResNet18(n_classes=1000, dropout=None):
    return _ResNet('resnet18', n_classes=n_classes, dropout=dropout)

def ResNet34(n_classes=1000, dropout=None):
    return _ResNet('resnet34', n_classes=n_classes, dropout=dropout)

def ResNet50(n_classes=1000, dropout=None):
    return _ResNet('resnet50', n_classes=n_classes, dropout=dropout)

def ResNet101(n_classes=1000, dropout=None):
    return _ResNet('resnet101', n_classes=n_classes, dropout=dropout)

def ResNet152(n_classes=1000, dropout=None):
    return _ResNet('resnet152', n_classes=n_classes, dropout=dropout)


if __name__ == '__main__':
    x = ResNet18(n_classes=1000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    summary(x, (3, 224, 224))