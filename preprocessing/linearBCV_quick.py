import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import argparse
import numpy as np
def parameter_count(model):
    print('# parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

def gpu_usage():
    print('gpu usage (current/max): {:.2f} / {:.2f} GB'.format(torch.cuda.memory_allocated()*1e-9, torch.cuda.max_memory_allocated()*1e-9))

def save_nib(img, path):
    img_nib = nib.Nifti1Image(img.detach().cpu().numpy(), np.eye(4))
    nib.save(img_nib, path)

def pdist(x, p=2):
    if p==1:
        dist = torch.abs(x.unsqueeze(2) - x.unsqueeze(1)).sum(dim=2)
    elif p==2:
        xx = (x**2).sum(dim=2).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x, x.permute(0, 2, 1))
        dist[:, torch.arange(dist.shape[1]), torch.arange(dist.shape[2])] = 0
    return dist

def pdist2(x, y, p=2):
    if p==1:
        dist = torch.abs(x.unsqueeze(2) - y.unsqueeze(1)).sum(dim=3)
    elif p==2:
        xx = (x**2).sum(dim=2).unsqueeze(2)
        yy = (y**2).sum(dim=2).unsqueeze(1)
        dist = xx + yy - 2.0 * torch.bmm(x, y.permute(0, 2, 1))
    return dist

def mindssc(img, delta=1, sigma=1):
    # see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details on the MIND-SSC descriptor
    
    device = img.device
    dtype = img.dtype
    
    # define start and end locations for self-similarity pattern
    six_neighbourhood = torch.Tensor([[0, 1, 1],
                                      [1, 1, 0],
                                      [1, 0, 1],
                                      [1, 1, 2],
                                      [2, 1, 1],
                                      [1, 2, 1]]).long()
    
    # squared distances
    dist = pdist(six_neighbourhood.unsqueeze(0)).squeeze(0)
    
    # define comparison mask
    x, y = torch.meshgrid(torch.arange(6), torch.arange(6),indexing='ij')
    mask = ((x > y).view(-1) & (dist == 2).view(-1))
    
    # build kernel
    idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1,6,1).view(-1,3)[mask, :]
    idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6,1,1).view(-1,3)[mask, :]
    mshift1 = torch.zeros(12, 1, 3, 3, 3).to(dtype).to(device)
    mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:,0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
    mshift2 = torch.zeros(12, 1, 3, 3, 3).to(dtype).to(device)
    mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:,0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
    rpad = nn.ReplicationPad3d(delta)
    smooth = nn.AvgPool3d(sigma*2+1)
    # compute patch-ssd
    ssd = F.avg_pool3d(((F.conv3d(rpad(img), mshift1, dilation=delta) - F.conv3d(rpad(img), mshift2, dilation=delta)) ** 2),sigma*2+1,stride=1,padding=sigma)
    
    # MIND equation
    mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
    mind_var = torch.mean(mind, 1, keepdim=True)
    mind_var = torch.clamp(mind_var, mind_var.mean() * 0.001, mind_var.mean() * 1000)
    mind /= mind_var
    mind = torch.exp(-mind).to(dtype)
    
    #permute to have same ordering as C++ code
    mind = mind[:, torch.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]
    
    return mind


#patch_step=2
def ssd(kpts_fixed, feat_fixed, feat_moving, orig_shape, disp_radius=16, disp_step=2, patch_radius=3, alpha=1.5, unroll_factor=50):
    _, N, _ = kpts_fixed.shape
    device = kpts_fixed.device
    D, H, W = orig_shape
    C = feat_fixed.shape[1]
    dtype = feat_fixed.dtype
    
    patch_step = disp_step # same stride necessary for fast implementation
    patch = torch.stack(torch.meshgrid(torch.arange(0, 2 * patch_radius + 1, patch_step),
                                       torch.arange(0, 2 * patch_radius + 1, patch_step),
                                       torch.arange(0, 2 * patch_radius + 1, patch_step),indexing='ij')).permute(1, 2, 3, 0).contiguous().view(1, 1, -1, 1, 3).float() - patch_radius
    patch = (patch.flip(-1) * 2 / (torch.tensor([W, H, D]) - 1)).to(dtype).to(device)
    #print(patch.shape)
    patch_width = round(patch.shape[2] ** (1.0 / 3))
    
    if patch_width % 2 == 0:
        pad = [(patch_width - 1) // 2, (patch_width - 1) // 2 + 1]
    else:
        pad = [(patch_width - 1) // 2, (patch_width - 1) // 2]
    #print(pad)
    disp = torch.stack(torch.meshgrid(torch.arange(- disp_step * (disp_radius + ((pad[0] + pad[1]) / 2)), (disp_step * (disp_radius + ((pad[0] + pad[1]) / 2))) + 1, disp_step),
                                      torch.arange(- disp_step * (disp_radius + ((pad[0] + pad[1]) / 2)), (disp_step * (disp_radius + ((pad[0] + pad[1]) / 2))) + 1, disp_step),
                                      torch.arange(- disp_step * (disp_radius + ((pad[0] + pad[1]) / 2)), (disp_step * (disp_radius + ((pad[0] + pad[1]) / 2))) + 1, disp_step),indexing='ij')).permute(1, 2, 3, 0).contiguous().view(1, 1, -1, 1, 3).float()
    disp = (disp.flip(-1) * 2 / (torch.tensor([W, H, D]) - 1)).to(dtype).to(device)
    #print(torch.arange(- disp_step * (disp_radius + ((pad[0] + pad[1]) / 2)), (disp_step * (disp_radius + ((pad[0] + pad[1]) / 2))) + 1, disp_step))
    disp_width = disp_radius * 2 + 1
    ssd = torch.zeros(1, N, disp_width ** 3).to(dtype).to(device)
    split = np.array_split(np.arange(N), unroll_factor)
    for i in range(unroll_factor):
        feat_fixed_patch = F.grid_sample(feat_fixed, kpts_fixed[:, split[i], :].view(1, -1, 1, 1, 3).to(dtype) + patch, padding_mode='border', align_corners=True)
        feat_moving_disp = F.grid_sample(feat_moving, kpts_fixed[:, split[i], :].view(1, -1, 1, 1, 3).to(dtype) + disp, padding_mode='border', align_corners=True)        
        corr = F.conv3d(feat_moving_disp.view(1, -1, disp_width + pad[0] + pad[1], disp_width + pad[0] + pad[1], disp_width + pad[0] + pad[1]), feat_fixed_patch.view(-1, 1, patch_width, patch_width, patch_width), groups=C * split[i].shape[0]).view(C, split[i].shape[0], -1)
        patch_sum = (feat_fixed_patch ** 2).squeeze(0).squeeze(3).sum(dim=2, keepdims=True)
        disp_sum = (patch_width ** 3) * F.avg_pool3d((feat_moving_disp ** 2).view(C, -1, disp_width + pad[0] + pad[1], disp_width + pad[0] + pad[1], disp_width + pad[0] + pad[1]), patch_width, stride=1).view(C, split[i].shape[0], -1)
        ssd[0, split[i], :] = ((- 2 * corr + patch_sum + disp_sum)).sum(0)
    
    ssd *= (alpha / (patch_width ** 3))
    
    return ssd,disp


#def least_trimmed_squares(fixed_pts,moving_pts,confidence):
#    idx = torch.arange(fixed_pts.size(0)).to(fixed_pts.device)
#    for i in range(20):
#        x,_ = #torch.solve(moving_pts[idx,:].t().mm(torch.diag(confidence[idx])).mm(moving_pts[idx,:]),moving_pts[idx,:].t().mm(torch.diag(confidence[idx])).mm(fixed_pts[idx,:]))
#        residual = torch.sqrt(torch.sum(torch.pow(moving_pts - torch.mm(fixed_pts, x),2),1))
#        _,idx = torch.topk(residual,fixed_pts.size(0)//2,largest=False)
#    return x
def least_trimmed_squares(fixed_pts,moving_pts,iter=5):
    idx = torch.arange(fixed_pts.size(0)).to(fixed_pts.device)
    for i in range(iter):
        x,_ = torch.solve(moving_pts[idx,:].t().mm(moving_pts[idx,:]),moving_pts[idx,:].t().mm(fixed_pts[idx,:]))
        residual = torch.sqrt(torch.sum(torch.pow(moving_pts - torch.mm(fixed_pts, x),2),1))
        _,idx = torch.topk(residual,fixed_pts.size(0)//2,largest=False)
    return x


def findRigid3d(fixed_pts,moving_pts):
    fixed_demean = (fixed_pts-fixed_pts.mean(0,keepdim=True))[:,:3]
    moving_demean = (moving_pts-moving_pts.mean(0,keepdim=True))[:,:3]
    cov_pts = torch.mm(fixed_demean.t(),moving_demean)
    U,S,V = torch.svd(cov_pts)
    R = torch.eye(4).to(U.device)
    R[:3,:3] = U.mm(V.t())
    t1=torch.mean(moving_pts.t()-torch.inverse(R).mm(fixed_pts.t()),1)[:3]#.t()
    #pts2b = torch.solve(moving_pts.t(),R)[0].t()#torch.inverse(R).mm(moving_pts.t())#
    R[3,:3] = t1
    return R

def least_trimmed_rigid(fixed_pts,moving_pts):#,confidence):
    idx = torch.arange(fixed_pts.size(0)).to(fixed_pts.device)
    for i in range(20):
        x = findRigid3d(fixed_pts[idx,:],moving_pts[idx,:])
        residual = torch.sqrt(torch.sum(torch.pow(moving_pts - torch.mm(fixed_pts, x),2),1))
        _,idx = torch.topk(residual,fixed_pts.size(0)//2,largest=False)
    return x


def compute_datacost_grid(img_fix,mind_fix,mind_mov,grid_step,disp_radius,disp_step):
    H,W,D = img_fix.size()

    grid_pts = F.affine_grid(.925*torch.eye(3,4).unsqueeze(0).cuda(),(1,1,H//grid_step,W//grid_step,D//grid_step),align_corners=True).view(1,-1,3)
    mask_bg = F.grid_sample(F.avg_pool3d(img_fix.view(1,1,H,W,D).cuda(),7,stride=1,padding=3),grid_pts.view(1,-1,1,1,3),align_corners=True)
    grid_pts = grid_pts[:,mask_bg.view(-1)>10,:] #hyper-parameter threshold
    datacost,disp_mesh = ssd(grid_pts, mind_fix, mind_mov, (H,W,D), disp_radius, disp_step, disp_radius+1)#, alpha=1.5, unroll_factor=50):
    device = mind_fix.device
    disp = torch.stack(torch.meshgrid(torch.arange(- disp_step * disp_radius, disp_step * disp_radius  + 1, disp_step),
                              torch.arange(- disp_step * disp_radius, disp_step * disp_radius  + 1, disp_step),
                              torch.arange(- disp_step * disp_radius, disp_step * disp_radius  + 1, disp_step),indexing='ij')).permute(1, 2, 3, 0).contiguous().view(1, -1, 3).float()
    disp = (disp.flip(-1) * 2 / (torch.tensor([D, W, H]) - 1)).to(device)
    ssd_val,ssd_idx = torch.min(datacost.squeeze(),1)
    idx_best = torch.sort(ssd_val,dim=0,descending=False)[1][:grid_pts.size(1)//2]
    disp_best = disp[0,ssd_idx[idx_best],:]
    fixed_pts = torch.cat((grid_pts.cpu()[0,idx_best,:],torch.ones(idx_best.size(0),1)),1)
    moving_pts = torch.cat((grid_pts.cpu()[0,idx_best,:]+disp_best.cpu(),torch.ones(idx_best.size(0),1)),1)
    return fixed_pts,moving_pts
    
def quick_linear(img_fix,img_mov,max_level,rigid_flag=True):
    quant = torch.Tensor([5,4,3,2,1])
    mind_step = torch.floor(0.5*quant+1.0)
    search_hw = torch.Tensor([6,5,4,3,2])
    grids = torch.Tensor([9,8,7,6,5,4])
    H,W,D = img_fix.shape
    #print('mind',mind_step,'quant',quant,'search',search_hw,'grid',grids,)
    with torch.no_grad():
        R0 = torch.eye(4)
        for level in range(max_level):#limit to 3 levels for speed
            H,W,D = img_fix.size()


            mind_fix = mindssc(img_fix.view(1,1,H,W,D).cuda(), delta=int(mind_step[level]), sigma=int(mind_step[level]))
            rigid_grid = F.affine_grid(R0[:,:3].t().unsqueeze(0).cuda(),(1,1,H,W,D),align_corners=True)
            img_mov_warp = F.grid_sample(img_mov.view(1,1,H,W,D).cuda(),rigid_grid,align_corners=True).squeeze()
            mind_mov_warp = mindssc(img_mov_warp.view(1,1,H,W,D).cuda(), delta=int(mind_step[level]), sigma=int(mind_step[level]))


            R0_inv = torch.inverse(R0)
            mind_mov = mindssc(img_mov.view(1,1,H,W,D).cuda(), delta=int(mind_step[level]), sigma=int(mind_step[level]))
            rigid_grid = F.affine_grid(R0_inv[:,:3].t().unsqueeze(0).cuda(),(1,1,H,W,D),align_corners=True)
            img_fix_warp = F.grid_sample(img_fix.view(1,1,H,W,D).cuda(),rigid_grid,align_corners=True).squeeze()
            mind_fix_warp = mindssc(img_fix_warp.view(1,1,H,W,D).cuda(), delta=int(mind_step[level]), sigma=int(mind_step[level]))


            #forward and backward
            fixed_pts1,moving_pts1 = compute_datacost_grid(img_fix,mind_fix,mind_mov_warp,grid_step=int(grids[level]),disp_radius=int(search_hw[level]),disp_step=int(quant[level]))#approx 20 pixel in each dim
            moving_pts2,fixed_pts2 = compute_datacost_grid(img_mov,mind_mov,mind_fix_warp,grid_step=int(grids[level]),disp_radius=int(search_hw[level]),disp_step=int(quant[level])) #approx 20 pixel in each dim
            if(rigid_flag):
                
                R = least_trimmed_rigid(torch.cat((fixed_pts1,fixed_pts2),0),torch.cat((moving_pts1,moving_pts2),0))
            else:
                R = least_trimmed_squares(torch.cat((fixed_pts1,fixed_pts2),0),torch.cat((moving_pts1,moving_pts2),0))
            
            R0  =torch.matmul(R,R0)


            #rigid_grid = F.affine_grid(R0[:,:3].t().unsqueeze(0),(1,1,H,W,D),align_corners=True)
            #rigid_warped_seg = #F.grid_sample(seg_mov.view(1,1,H,W,D),rigid_grid,mode='nearest',align_corners=True).squeeze()

            #d0 = dice_coeff(seg_fix,seg_mov,5); print(d0,d0.mean())
            #d1 = dice_coeff(seg_fix,rigid_warped_seg,5); print(d1,d1.mean())
            #print('level',level,'completed in ',time.time()-t0,'sec')
        #print(R0[:,:3])

    return R0,img_mov_warp

import argparse


