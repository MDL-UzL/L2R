import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import nibabel as nib
import time
import math

import sys
from tqdm.notebook import tqdm



### functions 
def jacobian_determinant_3d(dense_flow):
    B,_,H,W,D = dense_flow.size()
    
    dense_pix = dense_flow*(torch.Tensor([H-1,W-1,D-1])/2).view(1,3,1,1,1).to(dense_flow.device)
    gradz = nn.Conv3d(3,3,(3,1,1),padding=(1,0,0),bias=False,groups=3)
    gradz.weight.data[:,0,:,0,0] = torch.tensor([-0.5,0,0.5]).view(1,3).repeat(3,1)
    gradz.to(dense_flow.device)
    grady = nn.Conv3d(3,3,(1,3,1),padding=(0,1,0),bias=False,groups=3)
    grady.weight.data[:,0,0,:,0] = torch.tensor([-0.5,0,0.5]).view(1,3).repeat(3,1)
    grady.to(dense_flow.device)
    gradx = nn.Conv3d(3,3,(1,1,3),padding=(0,0,1),bias=False,groups=3)
    gradx.weight.data[:,0,0,0,:] = torch.tensor([-0.5,0,0.5]).view(1,3).repeat(3,1)
    gradx.to(dense_flow.device)
    #with torch.no_grad():
    jacobian = torch.cat((gradz(dense_pix),grady(dense_pix),gradx(dense_pix)),0)+torch.eye(3,3).view(3,3,1,1,1).to(dense_flow.device)
    jacobian = jacobian[:,:,2:-2,2:-2,2:-2]
    jac_det = jacobian[0,0,:,:,:]*(jacobian[1,1,:,:,:]*jacobian[2,2,:,:,:]-jacobian[1,2,:,:,:]*jacobian[2,1,:,:,:])-\
    jacobian[1,0,:,:,:]*(jacobian[0,1,:,:,:]*jacobian[2,2,:,:,:]-jacobian[0,2,:,:,:]*jacobian[2,1,:,:,:])+\
    jacobian[2,0,:,:,:]*(jacobian[0,1,:,:,:]*jacobian[1,2,:,:,:]-jacobian[0,2,:,:,:]*jacobian[1,1,:,:,:])

    return jac_det

def dice_coeff(outputs, labels, max_label):
    dice = torch.FloatTensor(max_label-1).fill_(0)
    for label_num in range(1, max_label):
        iflat = (outputs==label_num).view(-1).float()
        tflat = (labels==label_num).view(-1).float()
        intersection = torch.mean(iflat * tflat)
        dice[label_num-1] = (2. * intersection) / (1e-8 + torch.mean(iflat) + torch.mean(tflat))
    return dice


def weighted_sample(seg_fix,N,H,W,D):
    with torch.no_grad():
        weight_s = 1/(torch.bincount(seg_fix.long().reshape(-1))).float().pow(.5)
        weight_s[torch.isinf(weight_s)] = 0
        weight_s[torch.isnan(weight_s)] = 0
        #weight_s[weight_s==1] = 0
        weight_s[0] *= 10
        weight_s /= weight_s.mean()
        mask = F.max_pool3d(F.max_pool3d((seg_fix.view(1,1,H,W,D).cuda()>0).float(),5,stride=1,padding=2),5,stride=1,padding=2)
        indx = mask.view(-1).nonzero()
        w_idx = weight_s[seg_fix.reshape(-1)[indx]].squeeze()
        w_idx[torch.isinf(w_idx)] = 0
        w_idx[torch.isnan(w_idx)] = 0
        w_idx[w_idx<0] = 0
        ident = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,H,W,D))
        print('w_idx',len(w_idx),w_idx.sum(),'N',N)


        randweight = torch.utils.data.WeightedRandomSampler(w_idx,N,replacement=False)
        dataset = torch.utils.data.TensorDataset(torch.arange(len(w_idx)))
        loader = torch.utils.data.DataLoader(dataset, batch_size=N, sampler=randweight)
        for _, sample_batched in enumerate(loader):
            indices = sample_batched[0]
        #
        sample_index = indx[indices]
#        sample_label = seg_fix.reshape(-1)[sample_index]
        sample_xyz = ident.view(-1,3)[sample_index.view(-1),:]
        
    return sample_xyz


def pdist(x, p=2):
    if p==1:
        dist = torch.abs(x.unsqueeze(2) - x.unsqueeze(1)).sum(dim=2)
    elif p==2:
        xx = (x**2).sum(dim=2).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x, x.permute(0, 2, 1))
        dist[:, torch.arange(dist.shape[1]), torch.arange(dist.shape[2])] = 0
    return dist


class TPS:       
    @staticmethod
    def fit(c, f, lambd=0.):
        device = c.device
        
        n = c.shape[0]
        f_dim = f.shape[1]

        U = TPS.u(TPS.d(c, c))
        K = U + torch.eye(n, device=device) * lambd

        P = torch.ones((n, 4), device=device)
        P[:, 1:] = c

        v = torch.zeros((n+4, f_dim), device=device)
        v[:n, :] = f

        A = torch.zeros((n+4, n+4), device=device)
        A[:n, :n] = K
        A[:n, -4:] = P
        A[-4:, :n] = P.t()

        theta = torch.solve(v, A)[0]
        return theta
        
    @staticmethod
    def d(a, b):
        ra = (a**2).sum(dim=1).view(-1, 1)
        rb = (b**2).sum(dim=1).view(1, -1)
        dist = ra + rb - 2.0 * torch.mm(a, b.permute(1, 0))
        dist.clamp_(0.0, float('inf'))
        return torch.sqrt(dist)

    @staticmethod
    def u(r):
        return (r**2) * torch.log(r + 1e-6)

    @staticmethod
    def z(x, c, theta):
        U = TPS.u(TPS.d(x, c))
        w, a = theta[:-4], theta[-4:].unsqueeze(2)
        b = torch.matmul(U, w)
        return (a[0] + a[1] * x[:, 0] + a[2] * x[:, 1] + a[3] * x[:, 2] + b.t()).t()
    
def thin_plate_dense(x1, y1, shape, step, lambd=.0, unroll_step_size=2**12):
    device = x1.device
    D, H, W = shape
    D1, H1, W1 = D//step, H//step, W//step
    
    x2 = F.affine_grid(torch.eye(3, 4, device=device).unsqueeze(0), (1, 1, D1, H1, W1), align_corners=True).view(-1, 3)
    tps = TPS()
    theta = tps.fit(x1[0], y1[0], lambd)
    
    y2 = torch.zeros((1, D1 * H1 * W1, 3), device=device)
    N = D1*H1*W1
    n = math.ceil(N/unroll_step_size)
    for j in range(n):
        j1 = j * unroll_step_size
        j2 = min((j + 1) * unroll_step_size, N)
        y2[0, j1:j2, :] = tps.z(x2[j1:j2], x1[0], theta)
        
    y2 = y2.view(1, D1, H1, W1, 3).permute(0, 4, 1, 2, 3)
    y2 = F.interpolate(y2, (D, H, W), mode='trilinear', align_corners=True).permute(0, 2, 3, 4, 1)
    
    return y2

def knn_graph(kpts, k, include_self=False):
    B, N, D = kpts.shape
    device = kpts.device
    
    dist = pdist(kpts)
    ind = (-dist).topk(k + (1 - int(include_self)), dim=-1)[1][:, :, 1 - int(include_self):]
    A = torch.zeros(B, N, N).to(device)
    A[:, torch.arange(N).repeat(k), ind[0].t().contiguous().view(-1)] = 1
    A[:, ind[0].t().contiguous().view(-1), torch.arange(N).repeat(k)] = 1
    
    return ind, dist*A, A

def laplacian(kpts, k, lambd, sigma=0):
    _, dist, A = knn_graph(kpts, k)
    W = lambd * A.squeeze(0)
    if sigma > 0:
        W = W * torch.exp(- dist.squeeze(0) / (sigma ** 2))
    return (torch.diag(W.sum(1) + 1) - W).unsqueeze(0), W.unsqueeze(0)



