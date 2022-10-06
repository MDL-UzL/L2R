import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import nibabel as nib
import sys
from tqdm import tqdm

import os

from typing import List, Optional

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv3d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-3:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="trilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256) -> None:
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(nn.Conv3d(in_channels, out_channels, 1, bias=False), nn.BatchNorm3d(out_channels), nn.ReLU())
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv3d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels,out_channels,3,padding=1,bias=False),\
                                   nn.BatchNorm3d(out_channels),nn.PReLU())
        self.conv2 = nn.Sequential(nn.Conv3d(out_channels,out_channels,1,bias=False),\
                                   nn.BatchNorm3d(out_channels),nn.PReLU())

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x)
    
base_ch = 16

class UNet_Segment(nn.Module):
    def __init__(self,max_label):
        super().__init__()
        self.encoder = nn.ModuleDict({'enc1':ConvBlock(1,base_ch*2),'enc2':ConvBlock(base_ch*2,base_ch*3),\
                                      'enc3':ConvBlock(base_ch*3,base_ch*3),'enc4':ConvBlock(base_ch*3,base_ch*4),\
                                      'enc5':ConvBlock(base_ch*4,base_ch*4)})
        self.decoder = nn.ModuleDict({'dec1':ConvBlock(base_ch*8,base_ch*3),'dec2':ConvBlock(base_ch*6,base_ch*3),\
                                      'dec3':ConvBlock(base_ch*6,base_ch*3),'dec4':ConvBlock(base_ch*3,base_ch*2)})
        self.conv1 = ConvBlock(base_ch*2,base_ch*4)
        self.conv2 = nn.Sequential(nn.Conv3d(base_ch*4,base_ch*2,1,bias=False),nn.BatchNorm3d(base_ch*2),nn.PReLU(),\
                                 nn.Conv3d(base_ch*2,base_ch*2,1,bias=False),nn.BatchNorm3d(base_ch*2),nn.PReLU(),\
                                 nn.Conv3d(base_ch*2,max_label,1))
        self.aspp = ASPP(64,(1,2,4,8),64)


        self.final = nn.Identity()
    def forward(self, x):
        y = []
        upsample = nn.Upsample(scale_factor=2,mode='trilinear')
        for i in range(5):
            x = self.encoder['enc'+str(i+1)](x)
            if(i<4):
                y.append(x)
                x = F.max_pool3d(x,2) 
        x = self.aspp(x)
        for i in range(4):
            if(i<3):
                x = torch.cat((upsample(x),y.pop()),1)
            x = self.decoder['dec'+str(i+1)](x)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.final(x)

  