from src.model.nets import BaseNet
import torch
import torch.nn as nn
import numpy as np

# class SimpleNet(BaseNet):
#     def __init__(self, ):
#         super().__init__()
    
#     def forward(self, input):

class SimpleNet(BaseNet):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        num_features = [64, 128, 256, 512, 1024, 1024]

        self.block1 = make_block(in_channels, num_features[0])
        self.block2 = make_block(num_features[0], num_features[1])
        self.block3 = make_block(num_features[1], num_features[2])
        self.block4 = make_block(num_features[2], num_features[3])
        self.block5 = make_block(num_features[3], num_features[4])
        self.block6 = make_block(num_features[4], num_features[5])

        self.fcn1 = nn.Linear(65536 , 1024)
        self.fcn2 = nn.Linear(1024, 512)
        self.fcn3 = nn.Linear(512, out_channels)

        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU(inplace=True)
        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        output = self.block1(x)
        output = self.block2(output)
        output = self.block3(output)
        output = self.block4(output)
        output = self.block5(output)
        output = self.block6(output)
        
        output = output.view(output.size(0), -1)
        output = self.fcn1(output)
        output = self.relu(output)
        output = self.dropout(output)

        output = self.fcn2(output)
        output = self.relu(output)
        output = self.dropout(output)
        
        output = self.fcn3(output)

        return output


class make_block(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.add_module('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        self.add_module('norm1', nn.BatchNorm2d(out_channels))
        self.add_module('relu1', nn.ReLU(inplace=True))
        # self.add_module('conv2', nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        # self.add_module('norm2', nn.BatchNorm2d(out_channels))
        # self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('pool1', nn.MaxPool2d(kernel_size=2, stride=2))