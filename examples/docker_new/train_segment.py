#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import nibabel as nib
import time
from time import sleep
import sys
from tqdm import tqdm

import os


from utils import *
import features
from features.segmodels import *


if __name__ == "__main__":
    gpu_id = 0
    num_iterations = 5000
    task_name = "AbdomenCTCT"
    print(f"Arguments count: {len(sys.argv)}")
    for i, arg in enumerate(sys.argv):
        if(i==1):
            print(f"task_name {i:>6}: {arg}")
            task_name  = str(arg)
        elif(i==2):
            print(f"num_iterations {i:>6}: {arg}")
            num_iterations  = int(arg)
        elif(i==3):
            print(f"GPU ID {i:>6}: {arg}")
            gpu_id = int(arg)    
        else:
            print(f" argument {i:>6}: {arg}")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    print(torch.cuda.get_device_name())
    
    
    ##preload data

    #task_name = "AbdomenCTCT"#"OASIS"#
    data_path='../Learn2Reg_Dataset_release_v1.1/'+task_name+'/'
    with open(os.path.join(data_path,task_name+'_dataset.json')) as f:
        dataset_info=json.load(f)

    val_list=sorted(list(set([x['fixed'] for x in dataset_info['registration_val']] 
                  + [x['moving'] for x in dataset_info['registration_val']])))
    validation_ = dataset_info['registration_val']
    training_ = [x for x in dataset_info['training'] if x['image'] not in val_list]
    if(len(training_)>300):
        training_ = training_[::3]

    if(len(training_)>150):
        training_ = training_[::2]

    H,W,D = dataset_info['tensorImageShape']['0']
    num_val=len(val_list); num_train=len(training_)
    print('Training:',len(training_),'; Validation',len(val_list))


    ##training:
    img_all = torch.zeros(num_train,1,H//2,W//2,D//2)
    seg_all = torch.zeros(num_train,H,W,D).long()
    print('memory allocated')
    t0=time.time()
    for ii,i in enumerate(tqdm(training_)):
        seg_all[ii] = torch.from_numpy(nib.load(os.path.join(data_path,i['label'])).get_fdata()).long()
        img_all[ii] = F.avg_pool3d(torch.from_numpy(nib.load(os.path.join(data_path,i['image'])).get_fdata()).float().cuda().unsqueeze(0).unsqueeze(0)/500,2).cpu()
    t1 = time.time()
    print('training data loaded in %.2f s' % (t1-t0))

    ##validation
    seg_val = torch.zeros(num_val,H,W,D).long()
    img_val = torch.zeros(num_val,1,H//2,W//2,D//2)

    t0 = time.time()
    for ii,i in enumerate(tqdm(val_list)):
        seg_val[ii] = torch.from_numpy(nib.load(os.path.join(data_path,i.replace('image','label'))).get_fdata()).long()
        img_val[ii] = F.avg_pool3d(torch.from_numpy(nib.load(os.path.join(data_path,i)).get_fdata()).float().cuda().unsqueeze(0).unsqueeze(0)/500,2).cpu()
    t1 = time.time()
    print('validation data loaded in %.2f s' % (t1-t0))


    label = torch.from_numpy(nib.load(os.path.join(data_path,training_[0]['label'])).get_fdata()).long()#.argmax()
    max_label = int(label.reshape(-1).max()+1)
    print('#num_label',max_label)



    resnet = UNet_Segment(max_label)
    resnet.cuda()
    print()

    for repeat in range(2):
        #repeat twice
        ##Segmentation pre-training
        resnet.cuda()
        half_iterations = num_iterations//2

        optimizer = torch.optim.Adam(resnet.parameters(),lr=0.001)
        run_loss = torch.zeros(half_iterations)
        t0 = time.time()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,half_iterations//3,2)
        scaler = torch.cuda.amp.GradScaler()

        weight = torch.bincount(seg_all.long().reshape(-1)).float().pow(-.35).cuda()
        weight /= weight.mean() 

        idx_rand = torch.randperm(seg_all.shape[0])
        idx_train = idx_rand[:3*seg_all.shape[0]//4]
        with tqdm(total=half_iterations, file=sys.stdout, colour="red") as pbar:


            for i in range(half_iterations):
                idx = idx_train[torch.randperm(len(idx_train))][:2]
                optimizer.zero_grad()
                r_xyz = (torch.rand(3)*torch.tensor([H-128,W-128,D-128])).long()
                with torch.cuda.amp.autocast():
                    affine_matrix = (0.07*torch.randn(2,3,4)+torch.eye(3,4).unsqueeze(0)).cuda()
                    augment_grid = F.affine_grid(affine_matrix,(2,1,H,W,D),align_corners=False)[:,r_xyz[0]:r_xyz[0]+128,r_xyz[1]:r_xyz[1]+128,r_xyz[2]:r_xyz[2]+128,:]
                    input = F.grid_sample(img_all[idx].cuda(),augment_grid,align_corners=False)
                    target = F.grid_sample(seg_all[idx].cuda().unsqueeze(1).float(),                                       augment_grid,mode='nearest',align_corners=False).squeeze(1).long()
                    output = nn.Upsample(scale_factor=2,mode='trilinear')(resnet(F.avg_pool3d(input,1))[:,:])
                    loss = nn.CrossEntropyLoss(weight)(output,target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                run_loss[i] = loss.item()
                str1 = f"iter: {i}, loss: {'%0.3f'%(run_loss[i-28:i-1].mean())}, runtime: {'%0.3f'%(time.time()-t0)} sec, GPU max/memory: {'%0.2f'%(torch.cuda.max_memory_allocated()*1e-9)} GByte"
                pbar.set_description(str1)
                pbar.update(1)


    torch.save({'resnet':resnet.state_dict()},'models/'+task_name+'_segment.pth')

    #semantic_features:
    print('inferencing segmentation model')

    unet_val = torch.zeros(img_val.shape[0],H,W,D)
    resnet.eval()
    for i in tqdm(range(img_val.shape[0])):
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                affine = torch.eye(3,4).unsqueeze(0).cuda()#+torch.randn(4,3,4).cuda()*0#scale_affine
                affine2 = F.affine_grid(affine,(1,1,H,W,D))

                img_fix = F.grid_sample(img_val[i:i+1].cuda().half(),affine2[:1])


                feat = resnet(img_fix)

                predict = F.interpolate(feat,scale_factor=2,mode='trilinear').argmax(1).contiguous()

                labels_out = predict#largest_components(predict)



                unet_val[i] = labels_out
    d_val = torch.zeros(img_val.shape[0],max_label-1)
    for i in tqdm(range(img_val.shape[0])):
        with torch.no_grad():
            d_val[i] = dice_coeff(unet_val[i].cuda().contiguous(),seg_val[i].cuda().contiguous(),max_label)
    print('validation dice (unet)',d_val.mean())
    print(d_val.mean(1))

