import torch
from torch import nn
from torchsummary import summary
from base import BaseModel

class AlexNet(BaseModel):
    def __init__(self, n_classes = 1000, dropout = 0.5):
        super(AlexNet, self).__init__()

        self.features = self.__build_features()
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = self.__build__classifier(dropout, n_classes)

        self.init_weights()

    def __build_features(self):
        return nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def __build__classifier(self, dropout, n_classes):
        layers = []
        if dropout is not None:
            layers.append(nn.Dropout(dropout))
        layers.extend([nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True)])
        if dropout is not None:
            layers.append(nn.Dropout(dropout))
        layers.extend([nn.Linear(4096, 4096), nn.ReLU(inplace=True)])
        layers.append(nn.Linear(4096, n_classes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    x = AlexNet(n_classes=1000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    summary(x, (3, 224, 224))