import torch

from filters import *

def find_rigid_3d(x, y):
    device = x.device
    x_mean = x[:, :3].mean(0)
    y_mean = y[:, :3].mean(0)
    u, s, v = torch.svd(torch.matmul((x[:, :3]-x_mean).t(), (y[:, :3]-y_mean)))
    m = torch.eye(v.shape[0], v.shape[0], device=device)
    m[-1,-1] = torch.det(torch.matmul(v, u.t()))
    rotation = torch.matmul(torch.matmul(v, m), u.t())
    translation = y_mean - torch.matmul(rotation, x_mean)
    T = torch.eye(4, device=device)
    T[:3,:3] = rotation
    T[:3, 3] = translation
    return T

def compute_rigid_transform(kpts_fixed, kpts_moving, iter=5):
    device = kpts_fixed.device
    kpts_fixed = torch.cat((kpts_fixed, torch.ones(1, kpts_fixed.shape[1], 1, device=device)), 2)
    kpts_moving = torch.cat((kpts_moving, torch.ones(1, kpts_moving.shape[1], 1, device=device)), 2)
    idx = torch.arange(kpts_fixed.shape[1]).to(device)[torch.randperm(kpts_fixed.shape[1])[:kpts_fixed.shape[1]//2]]
    for i in range(iter):
        x = find_rigid_3d(kpts_fixed[0, idx, :], kpts_moving[0, idx, :]).t()
        residual = torch.sqrt(torch.sum(torch.pow(kpts_moving[0] - torch.mm(kpts_fixed[0], x), 2), 1))
        _, idx = torch.topk(residual, kpts_fixed.shape[1]//2, largest=False)
    return x.t().unsqueeze(0)