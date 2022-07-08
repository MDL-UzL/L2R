#!/usr/bin/env python

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.notebook import tqdm
import sys
import os
import numpy as np
import nibabel as nib
import time
import math
import json

import features
from features.lraspp3d import train_lraspp3d,create_lraspp3d
from features.mind import MINDSSC
from features.unet import create_unet

from thin_plate_spline import *

base_path = 'data/'

def prepare_data(dataset_json):
    with open(dataset_json, 'r') as f:
        dataset = json.load(f)
    
    H, W, D = dataset['tensorImageShape']['0']
    C = 1
    
    num_classes = len(dataset['labels']['0'])
    
    modality_count = 0
    for i in range(4):
        files = sorted(os.listdir(base_path+'imagesTr/'))
        mod_str = str(i).zfill(4)+'.nii.gz'
        if(sum(mod_str in s for s in files)>0):
            modality_count += 1

    print('there are '+str(modality_count)+' modalities')

    label_exist = (os.path.exists(base_path+'labelsTr/'))
    print('label_exist',label_exist)

    masks_exist = (os.path.exists(base_path+'masksTr/'))
    print('masks_exist',masks_exist)
    
    all_paired = False
    img_files = sorted(os.listdir(base_path+'imagesTr/'))
    ids = []
    for f in img_files:
        if('.nii.gz' in f):
            ids.append(int(f.split('_')[-2]))
    ids = np.asarray(ids)
    print('#ids',len(ids),'#unique',len(np.unique(ids)))
    all_paired = len(ids)==len(np.unique(ids))
    print('all_paired',all_paired)
        
    return num_classes, modality_count, label_exist, masks_exist

def main(dataset_json):
    
    with open(dataset_json, 'r') as f:
        dataset = json.load(f)
    
    H, W, D = dataset['tensorImageShape']['0']
    num_pairs = dataset['numRegistration_val']
    pairs = dataset['registration_val']
    
    num_classes, modality_count, label_exist, mask_exist = prepare_data(dataset_json)
    
    if(label_exist):
        models = []
        for modality in range(modality_count):
            mod_str = str(modality).zfill(4)+'.nii.gz'
            print('loading segmentation model for modality',mod_str,'#label',num_classes)
            network = create_lraspp3d(num_classes)
            network.load_state_dict(torch.load('models/network_'+str(modality).zfill(4)+'.pth'))
            network.eval()
            network.cuda()
            models.append(network)
            
    inshape = (H//2,W//2,D//2)
    unet_model = create_unet(10,inshape)
    unet_model.load_state_dict(torch.load('models/vxmplus_heatmap_nl.pth')['unet_model'])
    unet_model.eval()
    unet_model.cuda()
    heatmap = nn.Sequential(nn.ConvTranspose3d(64,16,5),
                            nn.InstanceNorm3d(16),nn.ReLU(),
                            nn.Conv3d(16,32,3,padding=1),
                            nn.InstanceNorm3d(32),nn.ReLU(),
                            nn.Conv3d(32,32,3,padding=1),
                            nn.Upsample(size=(11,11,11),mode='trilinear'),
                            nn.Conv3d(32,32,3,padding=1),
                            nn.InstanceNorm3d(64),
                            nn.ReLU(),
                            nn.Conv3d(32,32,3,padding=1),
                            nn.InstanceNorm3d(32),
                            nn.ReLU(),
                            nn.Conv3d(32,16,3,padding=1),
                            nn.InstanceNorm3d(16),
                            nn.ReLU(),
                            nn.Conv3d(16,1,3,padding=1))
    heatmap.load_state_dict(torch.load('models/vxmplus_heatmap_nl.pth')['heatmap'])
    heatmap.eval()
    heatmap.cuda()
    
    mesh = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda()*.3,(1,1,11,11,11)).half()
    patch = F.affine_grid(0.05*torch.eye(3,4).unsqueeze(0).cuda().half(),(1,1,3,3,3))
    
    for i in range(num_pairs):
        pair = pairs[i]
        fix_path = os.path.join(base_path,pair['fixed'])
        mov_path = os.path.join(base_path,pair['moving'])
        fix_id = os.path.basename(fix_path).split('_')[1]
        mov_id = os.path.basename(mov_path).split('_')[1]
        disp_path = os.path.join('output','disp_{}_{}.nii.gz'.format(fix_id, mov_id))
        
        fix = torch.from_numpy(nib.load(fix_path).get_fdata()).cuda().unsqueeze(0).unsqueeze(0).float()/500
        fix_mask = torch.from_numpy(nib.load(fix_path.replace('images','masks')).get_fdata()).cuda().unsqueeze(0).unsqueeze(0).float()
        mov = torch.from_numpy(nib.load(mov_path).get_fdata()).cuda().unsqueeze(0).unsqueeze(0).float()/500
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                fix_seg = models[0](fix).argmax(1, keepdim=True)
                mov_seg = models[1](mov).argmax(1, keepdim=True)
                
                mind_fix = F.avg_pool3d(MINDSSC(fix,2,2),2)
                mind_mov = F.avg_pool3d(MINDSSC(mov,2,2),2)
                
                input_fix = F.avg_pool3d(F.one_hot(fix_seg[0].long(),num_classes).float().permute(0,4,1,2,3),2)           
                input_mov = F.avg_pool3d(F.one_hot(mov_seg[0].long(),num_classes).float().permute(0,4,1,2,3),2)
                
                affine = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,H//2,W//2,D//2))
                nnunet_fix = F.grid_sample(input_fix,affine)
                nnunet_mov = F.grid_sample(input_fix,affine)

                keypts_rand = 2*torch.rand(2048*24,3).cuda()-1
                val = F.grid_sample(fix_mask,keypts_rand.view(1,-1,1,1,3),align_corners=False)
                idx1 = torch.nonzero(val.squeeze()==1).reshape(-1)
                keypts_fix = keypts_rand[idx1[:1024*2]]

                input = torch.cat((nnunet_fix,nnunet_mov),1).cuda()
                output = unet_model(input)

                sample_xyz = keypts_fix
                sampled = F.grid_sample(output,sample_xyz.cuda().view(1,-1,1,1,3),mode='bilinear',align_corners=False)
                disp_pred = heatmap(sampled.permute(2,1,0,3,4))

                pred_xyz = torch.sum(torch.softmax(disp_pred.view(-1,11**3,1),1)*mesh.view(1,11**3,3),1)
                
        dense_flow_ = thin_plate_dense(keypts_fix.unsqueeze(0), pred_xyz.unsqueeze(0), (H, W, D), 4, 0.1)
        dense_flow = dense_flow_.flip(4).permute(0,4,1,2,3)*torch.tensor([H-1,W-1,D-1]).cuda().view(1,3,1,1,1)/2
        
        disp_hr = dense_flow

        if(True):
            grid_sp = 2

            mind_fix = mind_fix.cuda().half()
            mind_mov = mind_mov.cuda().half()

            disp_lr = F.interpolate(disp_hr,size=(H//grid_sp,W//grid_sp,D//grid_sp),mode='trilinear',align_corners=False)
            net = nn.Sequential(nn.Conv3d(3,1,(H//grid_sp,W//grid_sp,D//grid_sp),bias=False))
            net[0].weight.data[:] = disp_lr.float().cpu().data/grid_sp
            net.cuda()
            optimizer = torch.optim.Adam(net.parameters(), lr=1)
            grid0 = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,H//grid_sp,W//grid_sp,D//grid_sp),align_corners=False)
            #run Adam optimisation with diffusion regularisation and B-spline smoothing
            lambda_weight = .65# with tps: .5, without:0.7
            for iter in range(50):#80
                optimizer.zero_grad()
                disp_sample = F.avg_pool3d(F.avg_pool3d(F.avg_pool3d(net[0].weight,3,stride=1,padding=1),\
                                                        3,stride=1,padding=1),3,stride=1,padding=1).permute(0,2,3,4,1)
                reg_loss = lambda_weight*((disp_sample[0,:,1:,:]-disp_sample[0,:,:-1,:])**2).mean()+\
                lambda_weight*((disp_sample[0,1:,:,:]-disp_sample[0,:-1,:,:])**2).mean()+\
                lambda_weight*((disp_sample[0,:,:,1:]-disp_sample[0,:,:,:-1])**2).mean()
                scale = torch.tensor([(H//grid_sp-1)/2,(W//grid_sp-1)/2,(D//grid_sp-1)/2]).cuda().unsqueeze(0)
                grid_disp = grid0.view(-1,3).cuda().float()+((disp_sample.view(-1,3))/scale).flip(1).float()
                patch_mov_sampled = F.grid_sample(mind_mov.float(),grid_disp.view(1,H//grid_sp,W//grid_sp,D//grid_sp,3).cuda(),\
                                                  align_corners=False,mode='bilinear')#,padding_mode='border')
                sampled_cost = (patch_mov_sampled-mind_fix).pow(2).mean(1)*12
                loss = sampled_cost.mean()
                (loss+reg_loss).backward()
                optimizer.step()
            fitted_grid = disp_sample.permute(0,4,1,2,3).detach()
            disp_hr = F.interpolate(fitted_grid*grid_sp,size=(H,W,D),mode='trilinear',align_corners=False)

        disp_smooth = F.avg_pool3d(F.avg_pool3d(F.avg_pool3d(disp_hr,3,padding=1,stride=1),\
                                                3,padding=1,stride=1),3,padding=1,stride=1)


        disp_hr = disp_smooth#torch.flip(disp_smooth/torch.tensor([H-1,W-1,D-1]).view(1,3,1,1,1).cuda()*2,[1])
        
        nib.save(nib.Nifti1Image(disp_hr[0].permute(1,2,3,0).cpu().numpy(), np.eye(4)), disp_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'vxm++ args')
    parser.add_argument('dataset_json', help='path to dataset_json')
    args = parser.parse_args()

    main(args.dataset_json)