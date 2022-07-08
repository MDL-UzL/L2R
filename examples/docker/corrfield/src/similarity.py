import math
import torch
import torch.nn.functional as F

from utils import *

def ssd(kpts_fixed, feat_fixed, feat_moving, disp_radius=16, disp_step=2, patch_radius=3, unroll_step_size=2**6):
    device = kpts_fixed.device
    N = kpts_fixed.shape[1]
    _, C, D, H, W = feat_fixed.shape
    
    patch_step = disp_step # same stride necessary for fast implementation
    patch = get_patch(patch_step, patch_radius, (D, H, W), device=device)
    patch_width = round(patch.shape[1] ** (1.0 / 3))
    
    pad = [(patch_width - 1) // 2, (patch_width - 1) // 2 + (1-patch_width % 2)]
    
    disp = get_disp(disp_step, disp_radius + ((pad[0] + pad[1]) / 2), (D, H, W), device=device)
    disp_width = disp_radius * 2 + 1
    
    cost = torch.zeros(1, N, disp_width, disp_width, disp_width, device=device)
    n = math.ceil(N/unroll_step_size)
    for j in range(n):
        j1 = j * unroll_step_size
        j2 = min((j + 1) * unroll_step_size, N)
        
        feat_fixed_patch = F.grid_sample(feat_fixed, kpts_fixed[:, j1:j2, :].view(1, -1, 1, 1, 3) + patch.view(1, 1, -1, 1, 3), mode='nearest', padding_mode='border', align_corners=True)
        feat_moving_disp = F.grid_sample(feat_moving, kpts_fixed[:, j1:j2, :].view(1, -1, 1, 1, 3) + disp.view(1, 1, -1, 1, 3), mode='nearest', padding_mode='border', align_corners=True)
        
        fixed_sum = (feat_fixed_patch ** 2).sum(dim = 3).view(C, (j2-j1), 1, 1, 1)
        moving_sum = (patch_width ** 3) * F.avg_pool3d((feat_moving_disp ** 2).view(C, -1, disp_width + pad[0] + pad[1], disp_width + pad[0] + pad[1], disp_width + pad[0] + pad[1]), patch_width, stride=1).view(C, (j2-j1), disp_width, disp_width, disp_width)
        corr = F.conv3d(feat_moving_disp.view(1, -1, disp_width + pad[0] + pad[1], disp_width + pad[0] + pad[1], disp_width + pad[0] + pad[1]), feat_fixed_patch.view(-1, 1, patch_width, patch_width, patch_width), groups = C * (j2-j1)).view(C, (j2-j1), disp_width, disp_width, disp_width)

        cost[0, j1:j2, :, :, :] = (fixed_sum + moving_sum - 2 * corr).sum(dim = 0) / (patch_width ** 3)
    
    return cost