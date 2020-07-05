from torchvision import models
from src.model.nets import BaseNet
from torch import nn


class ResNet(BaseNet):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
      #   for param in self.model.parameters(): param.requires_grad = False
        self.model.fc = nn.Sequential(
               nn.Linear(2048, 512),
               nn.ReLU(inplace=True),
               nn.Dropout(0.4),
               nn.Linear(512, 256),
               nn.ReLU(inplace=True),
               nn.Dropout(0.3),
               nn.Linear(256, out_channels)
               )
        
    def forward(self, input):
        return self.model(input)