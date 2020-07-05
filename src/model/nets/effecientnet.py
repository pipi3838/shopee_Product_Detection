from efficientnet_pytorch import EfficientNet
from src.model.nets import BaseNet
from torch import nn

class EffNet(BaseNet):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=out_channels)
        for param in self.model.parameters(): param.requires_grad = False
        self.model._fc = nn.Sequential(
               nn.Linear(2560, 512),
               nn.ReLU(inplace=True),
               nn.Dropout(0.4),
               nn.Linear(512, 256),
               nn.ReLU(inplace=True),
               nn.Dropout(0.3),
               nn.Linear(256, out_channels)
               )
        
    def forward(self, input):
        return self.model(input)

class EffNet_fix(BaseNet):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b2', num_classes=out_channels)
        # for param in self.model.parameters(): param.requires_grad = False
        # self.model._fc = nn.Sequential(
        #        nn.Linear(2048, 512),
        #        nn.ReLU(inplace=True),
        #        nn.Dropout(0.4),
        #     #    nn.Linear(512, 256),
        #     #    nn.ReLU(inplace=True),
        #     #    nn.Dropout(0.3),
        #        nn.Linear(512, out_channels)
        #        )
        
    def forward(self, input):
        return self.model(input)

class EffNet_b4(BaseNet):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=out_channels)
        # for param in self.model.parameters(): param.requires_grad = False
        # self.model._fc = nn.Sequential(
        #        nn.Linear(2048, 512),
        #        nn.ReLU(inplace=True),
        #        nn.Dropout(0.4),
        #     #    nn.Linear(512, 256),
        #     #    nn.ReLU(inplace=True),
        #     #    nn.Dropout(0.3),
        #        nn.Linear(512, out_channels)
        #        )
        
    def forward(self, input):
        return self.model(input)

class EffNet_b3_fc(BaseNet):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=out_channels)
        # for param in self.model.parameters(): param.requires_grad = False
        self.model._fc = nn.Sequential(
               nn.Linear(1536, 512),
               nn.ReLU(inplace=True),
               nn.Dropout(0.4),
               nn.Linear(512, 256),
               nn.ReLU(inplace=True),
               nn.Dropout(0.3),
               nn.Linear(256, out_channels)
               )
        
    def forward(self, input):
        return self.model(input)