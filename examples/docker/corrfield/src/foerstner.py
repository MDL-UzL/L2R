import torch
import torch.nn.functional as F

from filters import *
from utils import *

def structure_tensor(img, sigma):
    B, C, D, H, W = img.shape
    
    struct = []
    for i in range(C):
        for j in range(i, C):
            struct.append(smooth((img[:, i, ...] * img[:, j, ...]).unsqueeze(1), sigma))

    return torch.cat(struct, dim=1)

def invert_structure_tensor(struct):
    a = struct[:, 0, ...]
    b = struct[:, 1, ...]
    c = struct[:, 2, ...]
    e = struct[:, 3, ...]
    f = struct[:, 4, ...]
    i = struct[:, 5, ...]

    A =   e*i - f*f
    B = - b*i + c*f
    C =   b*f - c*e
    E =   a*i - c*c
    F = - a*f + b*c
    I =   a*e - b*b

    det = (a*A + b*B + c*C).unsqueeze(1)

    struct_inv = (1./det) * torch.stack([A, B, C, E, F, I], dim=1)

    return struct_inv

def foerstner_kpts(img, mask, sigma=1.4, d=9, thresh=1e-8):
    _, _, D, H, W = img.shape
    device = img.device
    
    filt = torch.tensor([1.0 / 12.0, -8.0 / 12.0, 0.0, 8.0 / 12.0, -1.0 / 12.0], device=device)
    grad = torch.cat([filter1D(img, filt, 0),
                      filter1D(img, filt, 1),
                      filter1D(img, filt, 2)], dim=1)
    
    struct_inv = invert_structure_tensor(structure_tensor(grad, sigma))
    
    distinctiveness = 1. / (struct_inv[:, 0, ...] + struct_inv[:, 3, ...] + struct_inv[:, 5, ...]).unsqueeze(1)
    
    pad1 = d//2
    pad2 = d - pad1 - 1
    
    maxfeat = F.max_pool3d(F.pad(distinctiveness, (pad2, pad1, pad2, pad1, pad2, pad1)), d, stride=1)
    
    structure_element = torch.tensor([[[0., 0,  0],
                                       [0,  1,  0],
                                       [0,  0,  0]],
                                      [[0,  1,  0],
                                       [1,  0,  1],
                                       [0,  1,  0]],
                                      [[0,  0,  0],
                                       [0,  1,  0],
                                       [0,  0,  0]]], device=device)
    
 
    mask_eroded = (1 - F.conv3d(1 - mask.float(), structure_element.unsqueeze(0).unsqueeze(0), padding=1).clamp_(0, 1)).bool()
    
    kpts = torch.nonzero(mask_eroded & (maxfeat == distinctiveness) & (distinctiveness >= thresh)).unsqueeze(0)[:, :, 2:]
    
    return kpts_pt(kpts, (D, H, W), align_corners=True)