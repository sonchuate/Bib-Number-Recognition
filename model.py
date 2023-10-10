import torch
from torch import nn
from torch.nn.functional import softmax



class OCR(nn.Module):
    def __init__(self, shape = [28, 28, 1], dropout = 0.4):  
        super(OCR, self).__init__()      
        self.conv1a = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pooling = nn.MaxPool2d(kernel_size=2,stride=2)
        self.dropout = nn.Dropout(dropout)
        self.dense1 = nn.LazyLinear(256)
        self.dense2 = nn.LazyLinear(11)#10 là số chữ số + 1 background
        self.relu = nn.ReLU()
    def forward(self, x):
        batch_size = x.shape[0]
        conv = self.conv1a(x)
        conv = self.conv1b(conv)
        conv = self.pooling(conv)
        conv = self.conv2a(conv)
        conv = self.conv2b(conv)
        conv = self.pooling(conv)
        conv = self.conv3a(conv)
        conv = self.conv3b(conv)
        conv = self.pooling(conv)
        features = conv.view(batch_size,-1)
        features = self.dense1(features)
        features = self.relu(features)
        features = self.dense2(features)
        return softmax(features)
    



