# -*- coding: utf-8 -*-
"""
Created on Sun May 17 10:34:15 2020

@author: admin
"""

import torch
from torch import nn
import torch.nn.functional as F
import math


class SoftmaxClassifier(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(SoftmaxClassifier, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        outputs = []
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        outputs.append(x)
        return outputs

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden1 = nn.Linear(dim_hidden, dim_hidden)
        self.layer_hidden2 = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        outputs = []
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        outputs.append(x)
        x = self.layer_hidden1(x)
        x = self.dropout(x)
        x = self.relu(x)
        outputs.append(x)
        x = self.layer_hidden2(x)
        outputs.append(x)
        return outputs


class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(args.num_channels, 32, kernel_size=5,padding=0,
                               stride=1,
                               bias=True)
        self.norm1 = torch.nn.GroupNorm(num_channels=32, num_groups=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=0,
                               stride=1,
                               bias=True)
        self.norm2 = torch.nn.GroupNorm(num_channels=64, num_groups=2)
        
        self.columns_fc1 = nn.Linear(1024, 512)
        self.norm3 = torch.nn.GroupNorm(num_channels=512, num_groups=2)
        
        self.fc2 = nn.Linear(512, args.num_classes)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        
    def forward(self, x):
        outputs = []
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.norm1(x)
        outputs.append(x)
        
        x = self.maxpool(F.relu(self.conv2(x)))
        x = self.norm2(x)
        outputs.append(x)
       
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.columns_fc1(x))
        x = self.norm3(x)
        outputs.append(x)
        
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        outputs.append(x)
        return outputs


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(args.num_channels, 32, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.norm1 = torch.nn.GroupNorm(num_channels=32, num_groups=2)
        
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.norm2 = torch.nn.GroupNorm(num_channels=64, num_groups=2)
        self.columns_fc1 = nn.Linear(64 * 5 * 5, 512)
        self.norm3 = torch.nn.GroupNorm(num_channels=512, num_groups=2)
        
        self.columns_fc2 = nn.Linear(512, 128)
        self.norm4 = torch.nn.GroupNorm(num_channels=128, num_groups=2)
        
        self.fc3 = nn.Linear(128, args.num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        outputs = []
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.norm1(x)
        
        outputs.append(x)
    
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.norm2(x)
        outputs.append(x)
        
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.columns_fc1(x))
        x = self.norm3(x)
        outputs.append(x)
        
        x = F.relu(self.columns_fc2(x))
        x = self.norm4(x)
        outputs.append(x)
        
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        outputs.append(x)
        return outputs #F.log_softmax(x, dim=1)
    

