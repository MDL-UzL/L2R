import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from tqdm.notebook import tqdm
import sys
import time


def get_layer(model, name):
    layer = model
    for attr in name.split("."):
        layer = getattr(layer, attr)
    return layer


def set_layer(model, name, layer):
    try:
        attrs, name = name.rsplit(".", 1)
        model = get_layer(model, attrs)
    except ValueError:
        pass
    setattr(model, name, layer)

import torchvision.models as models
from torchvision.models.segmentation.lraspp import LRASPP,LRASPPHead

from typing import Dict
from collections import OrderedDict
from torch import nn, Tensor

def create_lraspp3d(num_classes):
    class LRASPPHead3d(LRASPPHead):
        def __init__(self,low_channels: int,high_channels: int,num_classes: int,inter_channels: int):
            super(LRASPPHead, self).__init__()
            self.cbr = nn.Sequential(nn.Conv3d(high_channels, inter_channels, 1, bias=False),
                nn.BatchNorm3d(inter_channels),nn.ReLU(inplace=True))
            self.scale = nn.Sequential(nn.AdaptiveAvgPool3d(1),\
                    nn.Conv3d(high_channels, inter_channels, 1, bias=False),nn.Sigmoid())
            self.low_classifier = nn.Conv3d(low_channels, num_classes, 1)
            self.high_classifier = nn.Conv3d(inter_channels, num_classes, 1)
        def forward(self, input: Dict[str, Tensor]) -> Tensor:
            low = input["low"]; high = input["high"]
            x = self.cbr(high)
            s = self.scale(high)
            x = x * s
            x = F.interpolate(x, size=low.shape[-3:], mode='trilinear', align_corners=False)
            return self.low_classifier(low) + self.high_classifier(x)

    class LRASPP3d(nn.Module):
        def __init__(self,backbone: nn.Module,low_channels: int,high_channels: int,num_classes: int,inter_channels: int = 128) -> None:
            super().__init__()
            self.backbone = backbone
            self.classifier = LRASPPHead3d(low_channels, high_channels, num_classes, inter_channels)

        def forward(self, input: Tensor) -> Dict[str, Tensor]:
            features = self.backbone(input)
            out = self.classifier(features)
            out = F.interpolate(out, size=(int(input.shape[2]),int(input.shape[3]),int(input.shape[4])),mode='trilinear', align_corners=False)
            return out

    backbone = torchvision.models.segmentation.lraspp_mobilenet_v3_large(pretrained=False,num_classes=5).backbone
    network = LRASPP3d(backbone,40,960,num_classes,128)
    count = 0; count2 = 0
    for name, module in network.named_modules():
        if isinstance(module, nn.Conv2d):
            before = get_layer(network, name)
            kernel_size = tuple((list(before.kernel_size)*2)[:3])
            stride = tuple((list(before.stride)*2)[:3])
            padding = tuple((list(before.padding)*2)[:3])
            dilation = tuple((list(before.dilation)*2)[:3])
            in_channels = before.in_channels
            if(in_channels==3):
                before.in_channels = 1
            after = nn.Conv3d(before.in_channels,before.out_channels,kernel_size,stride=stride,\
                              padding=padding,dilation=dilation,groups=before.groups)
            set_layer(network, name, after); count += 1

        if isinstance(module, nn.BatchNorm2d):
            before = get_layer(network, name)
            after = nn.InstanceNorm3d(before.num_features)
            set_layer(network, name, after); count2 += 1
        if isinstance(module, nn.AdaptiveAvgPool2d):
            before = get_layer(network, name)
            after = nn.AdaptiveAvgPool3d(before.output_size)
            set_layer(network, name, after); count2 += 1
        if isinstance(module, nn.Hardswish):
            before = get_layer(network, name)
            after = nn.LeakyReLU(negative_slope=1e-2,inplace=True)
            set_layer(network, name, after); count2 += 1


    return network

def train_lraspp3d(img_all,seg_all,num_classes,num_iterations):
    H,W,D = img_all.shape[-3:]
    network = create_lraspp3d(num_classes)
    network.cuda()
    optimizer = torch.optim.Adam(network.parameters(),lr=0.001)
    run_loss = torch.zeros(num_iterations)
    t0 = time.time()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,num_iterations//3,2)
    scaler = torch.cuda.amp.GradScaler()

    weight = torch.bincount(seg_all.long().reshape(-1)).float().pow(-.35).cuda()
    weight /= weight.mean() 

    idx_rand = torch.randperm(seg_all.shape[0])
    idx_train = idx_rand[:3*seg_all.shape[0]//4]

    with tqdm(total=num_iterations, file=sys.stdout) as pbar:


        for i in range(num_iterations):
            idx = idx_train[torch.randperm(len(idx_train))][:2]
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                affine_matrix = (0.07*torch.randn(2,3,4)+torch.eye(3,4).unsqueeze(0)).cuda()
                augment_grid = F.affine_grid(affine_matrix,(2,1,H,W,D),align_corners=False)
                input = F.grid_sample(img_all[idx].cuda(),augment_grid,align_corners=False)
                target = F.grid_sample(seg_all[idx].cuda().unsqueeze(1).float(),\
                                       augment_grid,mode='nearest',align_corners=False).squeeze(1).long()
                output = network(input)
                loss = nn.CrossEntropyLoss(weight)(output,target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            run_loss[i] = loss.item()
            str1 = f"iter: {i}, loss: {'%0.3f'%(run_loss[i-28:i-1].mean())}, runtime: {'%0.3f'%(time.time()-t0)} sec, GPU max/memory: {'%0.2f'%(torch.cuda.max_memory_allocated()*1e-9)} GByte"
            pbar.set_description(str1)
            pbar.update(1)
            
    network.eval()
    network.cpu()
    return network


