import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

def gpu_usage():
    print('gpu usage (current/max): {:.2f} / {:.2f} GB'.format(torch.cuda.memory_allocated()*1e-9, torch.cuda.max_memory_allocated()*1e-9))
    
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
