import torch
import torch.nn as nn
import torch.nn.functional as F

from filters import *
from utils import *

def mindssc(img, delta=1, sigma=1):
    device = img.device
    
    # define start and end locations for self-similarity pattern
    six_neighbourhood = torch.tensor([[0, 1, 1],
                                      [1, 1, 0],
                                      [1, 0, 1],
                                      [1, 1, 2],
                                      [2, 1, 1],
                                      [1, 2, 1]], dtype=torch.float, device=device)
    
    # squared distances
    dist = cdist(six_neighbourhood.unsqueeze(0), six_neighbourhood.unsqueeze(0)).squeeze(0)
    
    # define comparison mask
    x, y = torch.meshgrid(torch.arange(6, device=device), torch.arange(6, device=device))
    mask = ((x > y).view(-1) & (dist == 2).view(-1))
    
    # build kernel
    idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1,6,1).view(-1,3)[mask, :].long()
    idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6,1,1).view(-1,3)[mask, :].long()
    mshift1 = torch.zeros((12, 1, 3, 3, 3), device=device)
    mshift1.view(-1)[torch.arange(12, device=device) * 27 + idx_shift1[:,0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
    mshift2 = torch.zeros((12, 1, 3, 3, 3), device=device)
    mshift2.view(-1)[torch.arange(12, device=device) * 27 + idx_shift2[:,0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
    rpad = nn.ReplicationPad3d(delta)
    
    # compute patch-ssd
    mind_ = []
    for i in range(img.shape[1]):
        ssd = smooth(((F.conv3d(rpad(img[:,i:i+1]), mshift1, dilation=delta) - F.conv3d(rpad(img[:,i:i+1]), mshift2, dilation=delta)) ** 2), sigma)

        # MIND equation
        mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
        mind_var = torch.mean(mind, 1, keepdim=True)
        mind_var = torch.clamp(mind_var, mind_var.mean() * 0.001, mind_var.mean() * 1000)
        mind /= mind_var
        mind = torch.exp(-mind)
        mind_.append(mind)
    
    return torch.cat(mind_, dim=1)