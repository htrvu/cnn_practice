import torch
from torch import nn
from torchsummary import summary
from base import BaseModel

class _VGGBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(_VGGBlock, self).__init__()
        layers = []
        for out_channels in n_filters:
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = out_channels
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        self.vggblock = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.vggblock(x)

CONFIGS = {
    'vgg11': [[64], [128], [256, 256], [512, 512], [512, 512]],
    'vgg13': [[64, 64], [128, 128], [256, 256], [512, 512], [512, 512]],
    'vgg16': [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]],
    'vgg19': [[64, 64], [128, 128], [256, 256, 256, 256], [512, 512, 512, 512], [512, 512, 512, 512]],
}

class _VGG(BaseModel):
    def __init__(self, model_name='vgg16', n_classes=1000, dropout=0.5):
        super(_VGG, self).__init__()
        
        self.features = self.__build_features(model_name)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = self.__build_classifier(dropout, n_classes)

        self.init_weights()

    def __build_features(self, model_name):
        cfg = CONFIGS[model_name]
        layers = []
        in_channels = 3
        for n_filters in cfg:
            layers.append(_VGGBlock(in_channels=in_channels, n_filters=n_filters))
            in_channels = n_filters[-1]
        return nn.Sequential(*layers)

    def __build_classifier(self, dropout, n_classes):
        layers = []
        layers.extend([nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True)])
        if dropout is not None:
            layers.append(nn.Dropout(dropout))
        layers.extend([nn.Linear(4096, 4096), nn.ReLU(inplace=True)])
        if dropout is not None:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(4096, n_classes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def VGG11(n_classes=1000, dropout=None):
    return _VGG('vgg11', n_classes=n_classes, dropout=dropout)

def VGG13(n_classes=1000, dropout=None):
    return _VGG('vgg13', n_classes=n_classes, dropout=dropout)

def VGG16(n_classes=1000, dropout=None):
    return _VGG('vgg16', n_classes=n_classes, dropout=dropout)

def VGG19(n_classes=1000, dropout=None):
    return _VGG('vgg19', n_classes=n_classes, dropout=dropout)


if __name__ == '__main__':
    x = VGG16(n_classes=1000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    summary(x, (3, 224, 224))