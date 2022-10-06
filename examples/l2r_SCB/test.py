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

import argparse
import json
import numpy as np
import nibabel as nib
import os



import sys

import os


from utils import *
import features
from features.segmodels import *
from features.unet import *


#only use for non-CT
def nonCTnorm(img):
    mask = img>0
    mean_intensity = img[mask].mean()
    std_intensity = img[mask].std()
    img = (img - mean_intensity) / (std_intensity + 1e-8)
    img[mask == 0] = 0
    return img

def run_seg(val_list,data_path,task_name,dataset_info,H,W,D):
    num_val = len(val_list)
    img_val = torch.zeros(num_val,1,H//2,W//2,D//2).pin_memory()
    #img_val_hr = torch.zeros(num_val,1,H,W,D)#.pin_memory()

    t0 = time.time()
    for ii,i in enumerate(val_list):
        #seg_val[ii] = torch.from_numpy(nib.load(os.path.join(data_path,i['label']).get_fdata()).long()
        img = torch.from_numpy(nib.load(os.path.join(data_path,i['image'])).get_fdata()).float().cuda().unsqueeze(0).unsqueeze(0)
        if not('CT' in dataset_info['modality']):
            img = nonCTnorm(img)
        else:
            img /= 500
        img_val[ii] = F.avg_pool3d(img,2).cpu()


    #label = torch.from_numpy(nib.load(os.path.join(data_path,training_[0]['label'])).get_fdata()).long()#.argmax()
    max_label = int(len(dataset_info['labels']))
    num_classes = max_label
    print('#num_label',max_label)


    models = torch.load('models/'+task_name+'_complete.pth')
    resnet = UNet_Segment(max_label)
    resnet.load_state_dict(models['resnet'])
    resnet.cuda()
    resnet.eval()

    print(H,W,D)
    inshape = (H//2,W//2,D//2)



    torch.cuda.empty_cache()

    print('inferencing segmentation model')


    unet_val = torch.zeros(img_val.shape[0],H,W,D)

    for i in tqdm(range(img_val.shape[0])):
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                affine = torch.eye(3,4).unsqueeze(0).cuda()#+torch.randn(4,3,4).cuda()*0#scale_affine
                affine2 = F.affine_grid(affine,(1,1,H,W,D))

                img_fix = F.grid_sample(img_val[i:i+1].cuda().half(),affine2[:1])


                feat = resnet(img_fix)


                predict = F.interpolate(feat,scale_factor=2,mode='trilinear').argmax(1).contiguous()


                unet_val[i] = predict


    return unet_val,img_val

def run_reg(unet_val,img_val,val_list,validation_,data_path,task_name,dataset_info,H,W,D):
    
    max_label = int(len(dataset_info['labels']))
    num_classes = max_label
    print('#num_label',max_label)

    inshape = (H//2,W//2,D//2)


    models = torch.load('models/'+task_name+'_complete.pth')
    
    
    
    unet_model = create_unet(num_classes*2,inshape)#
    heatmap = nn.Sequential(nn.ConvTranspose3d(64,16,5),nn.InstanceNorm3d(16),nn.ReLU(),nn.Conv3d(16,32,3,padding=1),nn.InstanceNorm3d(32),nn.ReLU(),                       nn.Conv3d(32,32,3,padding=1),nn.Upsample(size=(11,11,11),mode='trilinear'),nn.Conv3d(32,32,3,padding=1),nn.InstanceNorm3d(64),nn.ReLU(),                        #nn.Conv3d(32,32,3,padding=1),nn.InstanceNorm3d(32),nn.ReLU(),\
                           nn.Conv3d(32,32,3,padding=1),nn.InstanceNorm3d(32),nn.ReLU(),nn.Conv3d(32,16,3,padding=1),nn.InstanceNorm3d(16),nn.ReLU(),nn.Conv3d(16,1,3,padding=1))

    heatmap.cuda()
    regressor = nn.Sequential(nn.Conv3d(64,32,3,padding=1),nn.InstanceNorm3d(32),nn.ReLU(),nn.Conv3d(32,3,3,padding=1),nn.Tanh())
    models = torch.load('models/'+task_name+'_complete.pth')


    unet_model.load_state_dict(models['unet_model'])
    unet_model.cuda()
    unet_model.eval()
    regressor.load_state_dict(models['regressor'])
    regressor.cuda()
    regressor.eval()

    heatmap.load_state_dict(models['heatmap'])
    heatmap.cuda()
    heatmap.eval()
    
    weight = torch.bincount(unet_val[0].long().reshape(-1))
    for i in range(1,unet_val.shape[0]):
        weight += torch.bincount(unet_val[i].long().reshape(-1))

    weight = weight.float().pow(-.35).cuda()
    weight /= weight.mean() 
    print('label_weight',weight)

    unet_model.cuda(); regressor.cuda(); heatmap.cuda()
    unet_model.eval(); regressor.eval(); heatmap.eval()


    mesh = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda()*.3,(1,1,11,11,11)).half()
    patch = F.affine_grid(0.05*torch.eye(3,4).unsqueeze(0).cuda().half(),(1,1,3,3,3))


    heatmap_pred = False
    torch.manual_seed(0)
    for i in tqdm(range(len(validation_))):
        idx0 = val_list.index(validation_[i]['fixed'])
        idx0 = torch.arange(idx0,idx0+1)
        ii0 = int(validation_[i]['fixed'].split('_')[1])
        ii1 = int(validation_[i]['moving'].split('_')[1])
        
        idx1 = val_list.index(validation_[i]['moving'])
        idx1 = torch.arange(idx1,idx1+1)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                input_fix = img_val[idx0].cuda()#
                input_fix2 = F.avg_pool3d(F.one_hot(unet_val[idx0].cuda().long(),num_classes).float().permute(0,4,1,2,3),2)           
                input_mov2 = F.avg_pool3d(F.one_hot(unet_val[idx1].cuda().long(),num_classes).float().permute(0,4,1,2,3),2)           
                input_mov = img_val[idx1].cuda()#F.avg_pool3d(F.one_hot(unet_val[idx1].cuda().long(),num_classes).float().permute(0,4,1,2,3),2)            

                A1 = (torch.randn(3,4)*.0+torch.eye(3,4)).cuda()
                affine = F.affine_grid(A1.unsqueeze(0),(1,1,H//2,W//2,D//2))

                nnunet_fix = F.grid_sample(input_fix.cuda().half(),affine.half())
                nnunet_fix2 = F.grid_sample(input_fix2.cuda().half(),affine.half())

                A2 = (torch.randn(3,4)*.0+torch.eye(3,4)).cuda()
                affine = F.affine_grid(A2.unsqueeze(0),(1,1,H//2,W//2,D//2))
                nnunet_mov = F.grid_sample(input_mov.cuda().half(),affine.half())
                nnunet_mov2 = F.grid_sample(input_mov2.cuda().half(),affine.half())


                input = torch.cat((nnunet_fix2,nnunet_mov2),1).cuda()

                output = unet_model(input)

                if(heatmap_pred):
                    keypts_fix = weighted_sample(F.interpolate(nnunet_fix2,scale_factor=2,mode='trilinear').argmax(1),2*1024,H,W,D)



                    sample_xyz = keypts_fix
                    sampled = F.grid_sample(output,sample_xyz.cuda().view(1,-1,1,1,3)+patch.view(1,1,-1,1,3),mode='bilinear')
                    disp_pred = heatmap(sampled.permute(2,1,0,3,4).view(keypts_fix.shape[0],-1,3,3,3))

                    pred_xyz = torch.sum(torch.softmax(disp_pred.view(-1,11**3,1),1)*mesh.view(1,11**3,3),1)

                    soft_pred = torch.softmax(disp_pred.view(-1,11**3,1),1)
                    mesh_nnunet_mov = torch.sum(soft_pred.view(1,1,-1,11**3,1)*F.grid_sample(nnunet_mov,sample_xyz.cuda().view(1,-1,1,1,3)+mesh.view(1,1,-1,1,3),mode='bilinear'),3)
                    dense_flow_ = thin_plate_dense(keypts_fix.unsqueeze(0), pred_xyz.unsqueeze(0), (H, W, D), 4, 0.1)
                else:
                    dense_flow_ = .3*F.avg_pool3d(F.avg_pool3d(F.interpolate(regressor(unet_model(input)),scale_factor=2,mode='trilinear'),5,stride=1,padding=2),5,stride=1,padding=2)
                    dense_flow_ = F.interpolate(dense_flow_,scale_factor=2,mode='trilinear').permute(0,2,3,4,1)


                dense_flow = dense_flow_.flip(4).permute(0,4,1,2,3)*torch.tensor([H-1,W-1,D-1]).cuda().view(1,3,1,1,1)/2

            grid_sp = 2

            patch_mind_fix = 5*(input_fix2*weight.view(1,-1,1,1,1).cuda()).half()
            patch_mind_mov = 5*(input_mov2*weight.view(1,-1,1,1,1).cuda()).half()


            #create optimisable displacement grid
            disp_lr = F.interpolate(dense_flow,size=(H//grid_sp,W//grid_sp,D//grid_sp),mode='trilinear',align_corners=False)


        net = nn.Sequential(nn.Conv3d(3,1,(H//grid_sp,W//grid_sp,D//grid_sp),bias=False))
        net[0].weight.data[:] = 1*disp_lr.float().cpu().data/grid_sp
        net.cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=1)
        #torch.cuda.synchronize()
        #t0 = time.time()
        grid0 = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,H//grid_sp,W//grid_sp,D//grid_sp),align_corners=False)
        torch.cuda.synchronize()
        t0 = time.time()
        #run Adam optimisation with diffusion regularisation and B-spline smoothing
        lambda_weight = .7#1.5# with tps: .5, without:0.7
        for iter in range(50):#80
            optimizer.zero_grad()

            disp_sample = F.avg_pool3d(F.avg_pool3d(F.avg_pool3d(net[0].weight,3,stride=1,padding=1),3,stride=1,padding=1),3,stride=1,padding=1).permute(0,2,3,4,1)
            #disp_sample1 = disp_sample + disp_lr.permute(0,2,3,4,1)/grid_sp
            reg_loss = lambda_weight*((disp_sample[0,:,1:,:]-disp_sample[0,:,:-1,:])**2).mean()+        lambda_weight*((disp_sample[0,1:,:,:]-disp_sample[0,:-1,:,:])**2).mean()+        lambda_weight*((disp_sample[0,:,:,1:]-disp_sample[0,:,:,:-1])**2).mean()


            scale = torch.tensor([(H//grid_sp-1)/2,(W//grid_sp-1)/2,(D//grid_sp-1)/2]).cuda().unsqueeze(0)
            grid_disp = grid0.view(-1,3).cuda().float()+((disp_sample.view(-1,3))/scale).flip(1).float()

            patch_mov_sampled = F.grid_sample(patch_mind_mov.float(),grid_disp.view(1,H//grid_sp,W//grid_sp,D//grid_sp,3).cuda(),align_corners=False,mode='bilinear')
            sampled_cost = (patch_mov_sampled-patch_mind_fix).pow(2).mean(1)*12
            loss = sampled_cost.mean()
            (loss+reg_loss).backward()
            optimizer.step()

        fitted_grid = disp_sample.permute(0,4,1,2,3).detach()
        disp_hr = F.interpolate(fitted_grid*grid_sp,size=(H,W,D),mode='trilinear',align_corners=False)
        disp_smooth = F.avg_pool3d(F.avg_pool3d(F.avg_pool3d(disp_hr,3,padding=1,stride=1)                                         ,3,padding=1,stride=1),3,padding=1,stride=1)
        disp_path = os.path.join('/data_supergrover2/heinrich/temp288/output','disp_'+str(ii0).zfill(4)+'_'+str(ii1).zfill(4)+'.nii.gz')
        nib.save(nib.Nifti1Image(disp_smooth.permute(0,2,3,4,1).squeeze(0).cpu().numpy(), np.eye(4)), disp_path)
    return disp_smooth
#        disp = disp_smooth.cuda().float().permute(0,2,3,4,1)/torch.tensor([H-1,W-1,D-1]).cuda().view(1,1,1,1,3)*2
#        disp = disp.flip(4)


   #     disp_field = F.interpolate(disp_smooth,scale_factor = 0.5,mode='trilinear',align_corners=False)

   #     x1 = disp_field[0,0,:,:,:].cpu().float().data.numpy().astype('float16')
   #     y1 = disp_field[0,1,:,:,:].cpu().float().data.numpy().astype('float16')
   #     z1 = disp_field[0,2,:,:,:].cpu().float().data.numpy().astype('float16')

        #np.savez_compressed('outputs/'+task_name+'/disp_'+str(int(ii0)).zfill(4)+'_'+str(int(ii1)).zfill(4)+'.npz',np.stack((x1,y1,z1),0))



def main(dataset_json):

    torch.backends.cuda.matmul.allow_tf32 = True


    with open(dataset_json, 'r') as f:
        dataset = json.load(f)
    
    file_list = dataset_json.split('/')[:-1]
    task_name = dataset_json.split('/')[-1].split('_')[0]
    print('task_name',task_name)
    data_path = os.path.join(*file_list)
    print('data_path',data_path)
    H, W, D = dataset['tensorImageShape']['0']
    val_list = dataset['test']
    print(len(val_list))
    print(val_list[0])
    unet_val,img_val = run_seg(val_list,data_path,task_name,dataset,H,W,D)
    
    #num_pairs = dataset['numRegistration_test']
    validation_ = dataset['registration_test']
    val_list_img = []
    for i in val_list:
        val_list_img.append(i['image'])
    run_reg(unet_val,img_val,val_list_img,validation_,data_path,task_name,dataset,H,W,D)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'l2r_demo args')
    parser.add_argument('dataset_json', help='path to dataset_json')
    parser.add_argument("-t","--test_data", dest="test", action='store_true', default=False, help='use test data')
    args = parser.parse_args()
    main(args.dataset_json)


