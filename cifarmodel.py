# -*- coding: utf-8 -*-
"""CifarModel.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1nxG7lf2_OnIiG-_FHfTzoNP371fGQd4p
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ) # output_size = 32

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ) # output_size = 32
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ) # output_size = 30

        # TRANSITION BLOCK 1
        self.pool1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), padding=1, stride = 2,bias=False) # output_size = 16
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,1), padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ) # output_size = 16
        self.depthwiseseparable1 = nn.Sequential(
             nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size= 3, padding =1, groups=64),# output_size = 16
             nn.Conv2d(in_channels= 64, out_channels= 64, kernel_size=1, padding =0),
             nn.BatchNorm2d(64),
             nn.ReLU()
        )# output_size = 16
        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=2, dilation = 2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ) # output_size = 16
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1,stride =2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ) # output_size = 8

      
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=24, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU()
        ) # output_size = 8
        self.depthwiseseparable2 = nn.Sequential(
             nn.Conv2d(in_channels = 24, out_channels = 24, kernel_size= 3, padding =1, groups=24),# output_size = 10
             nn.Conv2d(in_channels= 24, out_channels= 24, kernel_size=1, padding =0),
             nn.BatchNorm2d(24),
             nn.ReLU()
        )# output_size = 8
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=16, kernel_size=(3, 3), padding=1, stride = 2,  bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ) # output_size = 4
        
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )# output_size = 4
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
        )
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        ) # output_size = 1

        self.dropout = nn.Dropout(0.03)

    def forward(self, x):
        x = self.convblock1(x) #32
        x = self.dropout(x)
        x = self.convblock2(x)#32
        x = self.dropout(x)
        x = self.convblock3(x)#30
        x = self.dropout(x)
        x = self.pool1(x)#16
        x = self.convblock4(x)#16
        x = self.dropout(x)
        x = self.depthwiseseparable1(x)#16,16
        x = self.convblock5(x)#16
        x = self.dropout(x)
        x = self.convblock6(x)#8
        x = self.dropout(x)
        x = self.convblock7(x)#8
        x = self.depthwiseseparable2(x)#8,8
        x = self.dropout(x)
        x = self.convblock8(x)#8
        x = self.dropout(x)
        x = self.convblock9(x)#8
        x = self.dropout(x)
        x = self.convblock10(x)#8
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)