#!/usr/bin/env python

import argparse
import nibabel as nib
import numpy as np
import os
import time
import torch
import torch.nn.functional as F
from scipy.ndimage.interpolation import zoom

from belief_propagation import *
from graphs import *
from foerstner import *
from mindssc import *
from rigid import *
from similarity import *
from thin_plate_spline import *
from utils import *

def compute_marginals(kpts_fix, img_fix, mind_fix, mind_mov, alpha, beta, disp_radius, disp_step, patch_radius):
    cost = alpha * ssd(kpts_fix, mind_fix, mind_mov, disp_radius, disp_step, patch_radius)
    
    k = 32
    while True:
        k *= 2
        try:
            dist = kpts_dist(kpts_fix, img_fix, beta, k)
            edges, level = minimum_spanning_tree(dist)
            break
        except:
            pass
    marginals = tbp(cost, edges, level, dist)
    
    return marginals

def corrfield(img_fix, mask_fix, img_mov, alpha, beta, gamma, delta, step, lambd, factor, sigma, sigma1, symmetric, L, N, Q, R, T):
    device = img_fix.device
    _, _, D, H, W = img_fix.shape
    
    print('Compute fixed MIND features ...', end =" ")
    torch.cuda.synchronize()
    t0 = time.time()
    mind_fix = mindssc(img_fix, delta, sigma1)
    torch.cuda.synchronize()
    t1 = time.time()
    print('finished ({:.2f} s).'.format(t1-t0))
    
    affine_grid = F.affine_grid(torch.eye(3, 4, dtype=img_mov.dtype, device=device).unsqueeze(0), (1, 1, D, H, W), align_corners=True)
     
    dense_flow = affine_grid
    img_mov_warped = img_mov
    for i in range(len(L)):
        print('Stage {}/{}'.format(i + 1, len(L)))
        print('    search radius: {}'.format(L[i]))
        print('      cube length: {}'.format(N[i]))
        print('     quantisation: {}'.format(Q[i]))
        print('     patch radius: {}'.format(R[i]))
        print('        transform: {}'.format(T[i]))
        
        disp = get_disp(Q[i], L[i], (D, H, W), device=device)
        
        print('    Compute moving MIND features ...', end =" ")
        torch.cuda.synchronize()
        t0 = time.time()
        mind_mov = mindssc(img_mov_warped, delta, sigma1)
        torch.cuda.synchronize()
        t1 = time.time()
        print('finished ({:.2f} s).'.format(t1-t0))
        print()
        
        torch.cuda.synchronize()
        t0 = time.time()
        if (i == 0) or (N[i]!=N[i-1]):
            if factor == 1:
                kpts_fix = foerstner_kpts(img_fix[:,:1], mask_fix, sigma, N[i])
            else:
                kpts_fix = foerstner_kpts(F.interpolate(img_fix[:,:1], scale_factor=factor, mode='trilinear', align_corners=True, recompute_scale_factor=True), (F.interpolate(mask_fix, scale_factor=factor, mode='trilinear', align_corners=True, recompute_scale_factor=True)>0.5), sigma, N[i])
        torch.cuda.synchronize()
        t1 = time.time()
        print('    {} fixed keypoints extracted ({:.2f} s).'.format(kpts_fix.shape[1], t1-t0))

        print('    Compute forward marginals ...', end =" ")
        torch.cuda.synchronize()
        t0 = time.time()
        marginalsf = compute_marginals(kpts_fix, img_fix[:,:1], mind_fix, mind_mov, alpha, beta, L[i], Q[i], R[i])
        torch.cuda.synchronize()
        t1 = time.time()
        print('finished ({:.2f} s).'.format(t1-t0))

        flow = (F.softmax(-gamma * marginalsf.view(1, kpts_fix.shape[1], -1, 1), dim=2) * disp.view(1, 1, -1, 3)).sum(2)
        
        if symmetric:
            print('    Compute symmetric backward marginals ...', end =" ")
            torch.cuda.synchronize()
            t0 = time.time()
            marginalsb = compute_marginals(kpts_fix + flow, img_mov[:,:1], mind_mov, mind_fix, alpha, beta, L[i], Q[i], R[i])
            torch.cuda.synchronize()
            t1 = time.time()
            print('finished ({:.2f} s).'.format(t1-t0))

            marginals = 0.5 * (marginalsf.view(1, kpts_fix.shape[1], -1) + marginalsb.view(1, kpts_fix.shape[1], -1).flip(2))

            flow = (F.softmax(-gamma * marginals.view(1, kpts_fix.shape[1], -1, 1), dim=2) * disp.view(1, 1, -1, 3)).sum(2)
        
        torch.cuda.synchronize()
        t0 = time.time()
        if  T[i] == 'r':
            print('    Find rigid transform ...', end =" ")
            rigid = compute_rigid_transform(kpts_fix, kpts_fix + flow)
            dense_flow_ = F.affine_grid(rigid[:, :3, :] - torch.eye(3, 4, device=device).unsqueeze(0), (1, 1, D, H, W), align_corners=True)
        elif T[i] == 'n':
            print('    Dense thin plate spline interpolation ...', end =" ")
            dense_flow_ = thin_plate_dense(kpts_fix, flow, (D, H, W), step, lambd)
        torch.cuda.synchronize()
        t1 = time.time()
        print('finished ({:.2f} s).'.format(t1-t0))
        
        dense_flow = F.grid_sample(dense_flow.permute(0,4,1,2,3),  affine_grid + dense_flow_.to(img_mov.dtype), align_corners=True).permute(0,2,3,4,1)
        img_mov_warped = F.grid_sample(img_mov, dense_flow.to(img_mov.dtype), align_corners=True)
        
        print()
        
        del mind_mov
        del marginalsf
        if symmetric:
            del marginalsb
        del flow
        del dense_flow_
        
    dense_flow = dense_flow - affine_grid
        
    flow = F.grid_sample(dense_flow.permute(0, 4, 1, 2, 3), kpts_fix.view(1, 1, 1, -1, 3), align_corners=True).view(1, 3, -1).permute(0, 2, 1)
    
    disp = flow_world(dense_flow.view(1, -1, 3), (D, H, W), align_corners=True).view(1, D, H, W, 3)
    kpts_mov = kpts_world(kpts_fix + flow, (D, H, W), align_corners=True)
    kpts_fix = kpts_world(kpts_fix, (D, H, W), align_corners=True)

    return disp, kpts_fix, kpts_mov