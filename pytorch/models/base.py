from torch import nn

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
    
    def init_weights(self):
        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)