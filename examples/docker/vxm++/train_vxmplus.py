#!/usr/bin/env python

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
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

base_path = 'data/'

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
    
    img_all = []
    seg_all = []
    mask_all = []

    intensity_div = 500
    for modality in range(modality_count):
        mod_str = str(modality).zfill(4)+'.nii.gz'
        validation = []
        dataset_json = os.path.join(base_path,'AbdomenMRCT_dataset.json')
        with open(dataset_json, 'r') as f:
            dataset = json.load(f)
            for pair in dataset['registration_val']:
                for element in pair.values():
                    if(mod_str in element):
                        validation.append(element.split('/')[-1])

        img_count = sum(mod_str in s for s in img_files)
        print('reading modality',mod_str,'#img_count',img_count,'#validation',len(validation))

        img = torch.zeros(img_count-len(validation),C,H,W,D)
        if(label_exist):
            seg = torch.zeros(img_count-len(validation),H,W,D)
        else:
            seg = None
        if(masks_exist):
            mask = torch.zeros(img_count-len(validation),H,W,D)
        else:
            mask = None

        count = 0
        with tqdm(total=img_count, file=sys.stdout) as pbar:


            for f in img_files:
                if(f in validation):
                    continue
                if(mod_str in f):
                    img[count] = torch.from_numpy(nib.load(base_path+'imagesTr/'+f).get_fdata()).float()/intensity_div
                    if(label_exist):
                        seg[count] = torch.from_numpy(nib.load(base_path+'labelsTr/'+f).get_fdata()).float()
                    if(masks_exist):
                        mask[count] = torch.from_numpy(nib.load(base_path+'labelsTr/'+f).get_fdata()).float()
                    count += 1
                    pbar.update(1)

        img_all.append(img)
        seg_all.append(seg)
        mask_all.append(mask)
        
    return num_classes, modality_count, label_exist, masks_exist, img_all, seg_all, mask_all
    
def main(dataset_json, num_iterations):
    
    with open(dataset_json, 'r') as f:
        dataset = json.load(f)
    
    H, W, D = dataset['tensorImageShape']['0']
    C = 1
    
    num_classes, modality_count, label_exist, mask_exist, img_all, seg_all, mask_all = prepare_data(dataset_json)
    
    if(label_exist):
        models = []
        for modality in range(modality_count):
            mod_str = str(modality).zfill(4)+'.nii.gz'
            print('loading segmentation model for modality',mod_str,'#label',num_classes)
            network = create_lraspp3d(num_classes)
            network.load_state_dict(torch.load('models/network_'+str(modality).zfill(4)+'.pth'))
            models.append(network)
            
    if(label_exist):
        semantic_features = []
        for modality in range(modality_count):
            mod_str = str(modality).zfill(4)+'.nii.gz'
            print('inferencing segmentation model for modality',mod_str,'#label',num_classes)

            unet_all = torch.zeros(img_all[modality].shape[0],H,W,D)
            network = models[modality]
            network.cuda()
            for i in tqdm(range(img_all[modality].shape[0])):
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        unet_all[i] = network(img_all[modality][i:i+1].cuda()).argmax(1).squeeze(0).cpu()
            semantic_features.append(unet_all)
            
    if(label_exist):
        weight = torch.bincount(seg_all[0].long().reshape(-1))
        for i in range(1,modality_count):
            weight += torch.bincount(seg_all[i].long().reshape(-1))

        weight = weight.float().pow(-.35).cuda()
        weight /= weight.mean() 
        print('label_weight',weight)
        
    mind_all = []
    for modality in range(modality_count):
        mod_str = str(modality).zfill(4)+'.nii.gz'
        print('extracting MIND features (half-res) for modality',mod_str)

        mind = torch.zeros(img_all[modality].shape[0],12,H//2,W//2,D//2)
        for i in tqdm(range(img_all[modality].shape[0])):
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        mind[i] = F.avg_pool3d(MINDSSC(img_all[modality][i:i+1].cuda(),2,2),2).cpu()
        mind_all.append(mind)
        
    print(H,W,D)
    inshape = (H//2,W//2,D//2)


    unet_model = create_unet(10,inshape)
    unet_model.cuda()
    heatmap = nn.Sequential(nn.ConvTranspose3d(64,16,5),nn.InstanceNorm3d(16),nn.ReLU(),nn.Conv3d(16,32,3,padding=1),nn.InstanceNorm3d(32),nn.ReLU(),                       nn.Conv3d(32,32,3,padding=1),nn.Upsample(size=(11,11,11),mode='trilinear'),nn.Conv3d(32,32,3,padding=1),nn.InstanceNorm3d(64),nn.ReLU(),                        #nn.Conv3d(32,32,3,padding=1),nn.InstanceNorm3d(32),nn.ReLU(),\
                           nn.Conv3d(32,32,3,padding=1),nn.InstanceNorm3d(32),nn.ReLU(),nn.Conv3d(32,16,3,padding=1),nn.InstanceNorm3d(16),nn.ReLU(),nn.Conv3d(16,1,3,padding=1))

    heatmap.cuda()
    print()
    
    mesh = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda()*.3,(1,1,11,11,11)).half()
    patch = F.affine_grid(0.05*torch.eye(3,4).unsqueeze(0).cuda().half(),(1,1,3,3,3))
    optimizer = torch.optim.Adam(list(unet_model.parameters())+list(heatmap.parameters()),lr=0.001)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,2*700,0.5)
    t0 = time.time()
    run_loss = torch.zeros(2*4900)
    for i in range(2*4900):
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                idx0 = torch.randperm(img_all[0].shape[0])[:1]
                idx1 = torch.randperm(img_all[1].shape[0])[:1]
                input_fix = F.avg_pool3d(F.one_hot(semantic_features[0][idx0].cuda().long(),num_classes).float().permute(0,4,1,2,3),2)           
                input_mov = F.avg_pool3d(F.one_hot(semantic_features[1][idx1].cuda().long(),num_classes).float().permute(0,4,1,2,3),2)            

                A1 = (torch.randn(3,4)*.035+torch.eye(3,4)).cuda()
                affine = F.affine_grid(A1.unsqueeze(0),(1,1,H//2,W//2,D//2))

                mask_fix = F.grid_sample(F.interpolate(mask_all[0][idx0].unsqueeze(1).cuda().half(),scale_factor=.5,mode='trilinear'),affine.half())
                nnunet_fix = F.grid_sample(input_fix.cuda().half(),affine.half())

                keypts_fix = weighted_sample(F.interpolate(input_fix,scale_factor=2,mode='trilinear').argmax(1),512,H,W,D)

                A2 = (torch.randn(3,4)*.035+torch.eye(3,4)).cuda()
                affine = F.affine_grid(A2.unsqueeze(0),(1,1,H//2,W//2,D//2))
                nnunet_mov = F.grid_sample(input_mov.cuda().half(),affine.half())

        with torch.cuda.amp.autocast():
            input = torch.cat((nnunet_fix,nnunet_mov),1).cuda()
            output = unet_model(input)

            sample_xyz = keypts_fix
            sampled = F.grid_sample(output,sample_xyz.cuda().view(1,-1,1,1,3)+patch.view(1,1,-1,1,3),mode='bilinear')
            disp_pred = heatmap(sampled.permute(2,1,0,3,4).view(keypts_fix.shape[0],-1,3,3,3))

            pred_xyz = torch.sum(torch.softmax(disp_pred.view(-1,11**3,1),1)*mesh.view(1,11**3,3),1)

            soft_pred = torch.softmax(disp_pred.view(-1,11**3,1),1)
            mesh_nnunet_mov = torch.sum(soft_pred.view(1,1,-1,11**3,1)*F.grid_sample(nnunet_mov,sample_xyz.cuda().view(1,-1,1,1,3)+mesh.view(1,1,-1,1,3),mode='bilinear'),3)
            sampled_nnunet_fix = F.grid_sample(nnunet_fix,sample_xyz.cuda().view(1,-1,1,1,3)).squeeze(2).squeeze(-1)
            loss = nn.MSELoss()(weight.view(1,-1,1,1,1)*mesh_nnunet_mov,weight.view(1,-1,1,1,1)*sampled_nnunet_fix)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        run_loss[i] = loss.item()
        if(i%150==19):
            print(i,run_loss[i-18:i-1].mean(),time.time()-t0,'sec')
            break
            
    unet_model.cpu(); heatmap.cpu()
    torch.save({'unet_model':unet_model.state_dict(),'heatmap':heatmap.state_dict()},'models/vxmplus_heatmap_nl.pth')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'vxm++ args')
    parser.add_argument('dataset_json', help='path to dataset_json')
    parser.add_argument('num_iterations', help='number of training iterations')
    args = parser.parse_args()

    main(args.dataset_json, int(args.num_iterations))