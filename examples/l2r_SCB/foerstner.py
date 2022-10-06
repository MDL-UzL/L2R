import torch
import torch.nn.functional as F

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

def gpu_usage():
    print('gpu usage (current/max): {:.2f} / {:.2f} GB'.format(torch.cuda.memory_allocated()*1e-9, torch.cuda.max_memory_allocated()*1e-9))
import torch
import torch.nn.functional as F

def filter1D(img, weight, dim, padding_mode='replicate'):
    B, C, D, H, W = img.shape
    N = weight.shape[0]
    
    padding = torch.zeros(6,)
    padding[[4 - 2 * dim, 5 - 2 * dim]] = N//2
    padding = padding.long().tolist()
    
    view = torch.ones(5,)
    view[dim + 2] = -1
    view = view.long().tolist()
    
    return F.conv3d(F.pad(img.view(B*C, 1, D, H, W), padding, mode=padding_mode), weight.view(view)).view(B, C, D, H, W)

def smooth(img, sigma):
    device = img.device
    
    sigma = torch.tensor([sigma], device=device)
    N = torch.ceil(sigma * 3.0 / 2.0).long().item() * 2 + 1
    
    weight = torch.exp(-torch.pow(torch.linspace(-(N // 2), N // 2, N, device=device), 2) / (2 * torch.pow(sigma, 2)))
    weight /= weight.sum()
    
    img = filter1D(img, weight, 0)
    img = filter1D(img, weight, 1)
    img = filter1D(img, weight, 2)
    
    return img

def mean_filter(img, r):
    device = img.device
    
    weight = torch.ones((2 * r + 1,), device=device)/(2 * r + 1)
    
    img = filter1D(img, weight, 0)
    img = filter1D(img, weight, 1)
    img = filter1D(img, weight, 2)
    
    return img

def minconv(input):
    device = input.device
    disp_width = input.shape[-1]
    
    disp1d = torch.linspace(-(disp_width//2), disp_width//2, disp_width, device=device)
    regular1d = (disp1d.view(1,-1) - disp1d.view(-1,1)) ** 2
    
    output = torch.min( input.view(-1, disp_width, 1, disp_width, disp_width) + regular1d.view(1, disp_width, disp_width, 1, 1), 1)[0]
    output = torch.min(output.view(-1, disp_width, disp_width, 1, disp_width) + regular1d.view(1, 1, disp_width, disp_width, 1), 2)[0]
    output = torch.min(output.view(-1, disp_width, disp_width, disp_width, 1) + regular1d.view(1, 1, 1, disp_width, disp_width), 3)[0]
    
    output = output - (torch.min(output.view(-1, disp_width**3), 1)[0]).view(-1, 1, 1, 1)

    return output.view_as(input)

def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def kpts_pt(kpts_world, shape, align_corners=None):
    device = kpts_world.device
    D, H, W = shape
   
    kpts_pt_ = (kpts_world.flip(-1) / (torch.tensor([W, H, D], device=device) - 1)) * 2 - 1
    if not align_corners:
        kpts_pt_ *= (torch.tensor([W, H, D], device=device) - 1)/torch.tensor([W, H, D], device=device)
    
    return kpts_pt_

def kpts_world(kpts_pt, shape, align_corners=None):
    device = kpts_pt.device
    D, H, W = shape
    
    if not align_corners:
        kpts_pt /= (torch.tensor([W, H, D], device=device) - 1)/torch.tensor([W, H, D], device=device)
    kpts_world_ = (((kpts_pt + 1) / 2) * (torch.tensor([W, H, D], device=device) - 1)).flip(-1) 
    
    return kpts_world_

def flow_pt(flow_world, shape, align_corners=None):
    device = flow_world.device
    D, H, W = shape
    
    flow_pt_ = (flow_world.flip(-1) / (torch.tensor([W, H, D], device=device) - 1)) * 2
    if not align_corners:
        flow_pt_ *= (torch.tensor([W, H, D], device=device) - 1)/torch.tensor([W, H, D], device=device)
    
    return flow_pt_

def flow_world(flow_pt, shape, align_corners=None):
    device = flow_pt.device
    D, H, W = shape
    
    if not align_corners:
        flow_pt /= (torch.tensor([W, H, D], device=device) - 1)/torch.tensor([W, H, D], device=device)
    flow_world_ = ((flow_pt / 2) * (torch.tensor([W, H, D], device=device) - 1)).flip(-1)
    
    return flow_world_

def get_disp(disp_step, disp_radius, shape, device):
    D, H, W = shape
    
    disp = torch.stack(torch.meshgrid(torch.arange(- disp_step * disp_radius, disp_step * disp_radius + 1, disp_step, device=device),
                                      torch.arange(- disp_step * disp_radius, disp_step * disp_radius + 1, disp_step, device=device),
                                      torch.arange(- disp_step * disp_radius, disp_step * disp_radius + 1, disp_step, device=device)), dim=3).view(1, -1, 3)
    
    disp = flow_pt(disp, (D, H, W), align_corners=True)
    return disp

def get_patch(patch_step, patch_radius, shape, device):
    D, H, W = shape
    
    patch = torch.stack(torch.meshgrid(torch.arange(0, 2 * patch_radius + 1, patch_step, device=device),
                                       torch.arange(0, 2 * patch_radius + 1, patch_step, device=device),
                                       torch.arange(0, 2 * patch_radius + 1, patch_step, device=device)), dim=3).view(1, -1, 3) - patch_radius
    patch = flow_pt(patch, (D, H, W), align_corners=True)
    return patch

def invert_flow(disp, iter=5):
    identity = F.affine_grid(torch.eye(3, 4, dtype=disp.dtype, device=disp.device).unsqueeze(0), (1, 1, *disp.shape[1:4]), align_corners=True)
    disp_inv = torch.zeros_like(disp)
    for i in range(iter):
        new_grid = disp_inv + identity
        disp_inv = - F.grid_sample(disp.permute(0, 4, 1, 2, 3), new_grid, mode='bilinear', padding_mode='border', align_corners=True).permute(0, 2, 3, 4, 1)
    return disp_inv

def cdist(x1, x2, p=2, sqrt=False):
    adjustment = x1.mean(-2, keepdim=True)
    x1 = x1 - adjustment
    x2 = x2 - adjustment
    
    if p == 1:
        dist = torch.abs(x1.unsqueeze(2) - x2.unsqueeze(1)).sum(dim=3)
    elif p == 2:
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x1_pad = torch.ones_like(x1_norm)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        x2_pad = torch.ones_like(x2_norm)
        x1_ = torch.cat([-2. * x1, x1_norm, x1_pad], dim=-1)
        x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
        dist = x1_.matmul(x2_.transpose(-2, -1))
        dist.clamp_min_(1e-30)

        if sqrt:
            dist = dist.sqrt()
        
    return dist

def save_nifti(img, spacing, filename):
    affine = np.eye(4)
    affine[0,0] = -spacing[0]
    affine[1,1] = -spacing[1]
    affine[2,2] = spacing[2]
    nib.save(nib.Nifti1Image(img, affine), filename)

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