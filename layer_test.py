'''
layer_test.py
Written by Seungpyo Hong
Test codes for pytorch_to_hls.py implementations
'''
import hls4ml
import torch
from torch import nn
import numpy as np
import yaml
import os


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.my_fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.my_fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def linear_test():
    f = MLP()
    f.eval()
    torch.save(f, os.path.join('pytorch', 'linear.pth'))

    with open('pytorch_config.yml', 'r') as yaml_f:
        config = yaml.load(yaml_f)
        # print(config)
        hls = hls4ml.converters.pytorch_to_hls(config)
        hls.compile()
        x = np.random.randn(16)
        y = hls.predict(x)
        print('x:', x)
        print('y:', y)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 4, dilation=2, padding_mode='circular', groups=1,  stride=3, bias=False, padding=1)
        self.conv2 = nn.Conv2d(16, 32, (3, 3), padding=2)
        self.conv3 = nn.Conv2d(32, 64, (3, 3), stride=2)
        self.conv4 = nn.Conv2d(64, 128, (3, 3), stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


def conv_test():
    f = CNN()
    f.eval()
    torch.save(f, os.path.join('pytorch', 'conv.pth'))

    with open('pytorch_config.yml', 'r') as yaml_f:
        config = yaml.load(yaml_f)
        # print(config)
        hls = hls4ml.converters.pytorch_to_hls(config)
        hls.compile()
        x = np.random.randn((1, 3, 32, 32))
        y = hls.predict(x)
        print('x:', x)
        print('y:', y)


if __name__ == '__main__':
    print('PyTorch Layer Test')
    # linear_test()
    conv_test()
