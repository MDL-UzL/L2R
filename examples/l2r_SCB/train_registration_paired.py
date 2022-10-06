import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import nibabel as nib
import time

import sys
from tqdm.notebook import tqdm

import os
import logging

gpu_id = 0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
print(torch.cuda.get_device_name())

from utils import *
import features
from features.segmodels import *
from features.unet import *



if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s",\
        handlers=[logging.FileHandler("output/debug.log"),logging.StreamHandler()])

    gpu_id = 0
    num_iterations = 5000
    task_name = "NLST"
    print(f"Arguments count: {len(sys.argv)}")
    for i, arg in enumerate(sys.argv):
        if(i==1):
            print(f"json_path {i:>6}: {arg}")
            json_path = str(arg)
        elif(i==2):
            print(f"num_iterations {i:>6}: {arg}")
            num_iterations = int(arg)
        elif(i==3):
            print(f"GPU ID {i:>6}: {arg}")
            gpu_id = int(arg)    
        else:
            print(f" argument {i:>6}: {arg}")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)


    ##preload data

    #task_name = "NLST"#"OASIS"#
    data_path=os.path.dirname(json_path)
    with open(json_path) as f:
        dataset_info=json.load(f)

    task_name=dataset_info['name']
    val_list=sorted(list(set([x['fixed'] for x in dataset_info['registration_val']] 
                  + [x['moving'] for x in dataset_info['registration_val']])))
    validation_ = dataset_info['registration_val']
    training_ = [x for x in dataset_info['training'] if x['image'] not in val_list]

    logging.info(f"Task Name: {task_name}, Iterations: {num_iterations}, GPU: {gpu_id}")

    H,W,D = dataset_info['tensorImageShape']['0']
    num_val=len(val_list); num_train=len(training_)
    logging.info('Training '+str(len(training_))+'; Validation '+str(len(val_list)))




    ##training:
    import csv

    img_all = torch.zeros(num_train,2,H//2,W//2,D//2)
    mask_all = torch.zeros(num_train,2,H,W,D).long()

    keypts_all = []

    #print(H,W,D)

    t0=time.time()
    for ii,i in enumerate(tqdm(training_)):
        mask_all[ii,0] = torch.from_numpy(nib.load(os.path.join(data_path,i['mask'])).get_fdata()).long()
        img_all[ii,0] = F.avg_pool3d(torch.from_numpy(nib.load(os.path.join(data_path,i['image'])).get_fdata()).float().cuda().unsqueeze(0).unsqueeze(0)/500,2).cpu()
        mask_all[ii,1] = torch.from_numpy(nib.load(os.path.join(data_path,i['mask'].replace('_0000','_0001'))).get_fdata()).long()
        img_all[ii,1] = F.avg_pool3d(torch.from_numpy(nib.load(os.path.join(data_path,i['image'].replace('_0000','_0001'))).get_fdata()).float().cuda().unsqueeze(0).unsqueeze(0)/500,2).cpu()


        corrfield1 = torch.empty(0,3)
        corrfield2 = torch.empty(0,3)

        with open(os.path.join(data_path,i['keypoints'].replace('.nii.gz','.csv')), newline='') as csvfile:
            fread = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in fread:
                corrfield1 = torch.cat((corrfield1,torch.from_numpy(np.array(row).astype('float32')).float().view(1,3)),0)
        with open(os.path.join(data_path,i['keypoints'].replace('.nii.gz','.csv').replace('_0000','_0001')), newline='') as csvfile:
            fread = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in fread:
                corrfield2 = torch.cat((corrfield2,torch.from_numpy(np.array(row).astype('float32')).float().view(1,3)),0)

        keypts_fix = torch.stack((corrfield1[:,2+0]/(D-1)*2-1,corrfield1[:,1+0]/(W-1)*2-1,corrfield1[:,0+0]/(H-1)*2-1),1).cuda()
        keypts_mov = torch.stack((corrfield2[:,2+0]/(D-1)*2-1,corrfield2[:,1+0]/(W-1)*2-1,corrfield2[:,0+0]/(H-1)*2-1),1).cuda()
        keypts_all.append(torch.stack((keypts_fix,keypts_mov),0))

    t1 = time.time()
    logging.info('training data loaded in %.2f s' % (t1-t0))



    unet_model = create_unet(2,(H//2,W//2,D//2))#
    unet_model.cuda()
    heatmap = nn.Sequential(nn.ConvTranspose3d(64,16,5),nn.InstanceNorm3d(16),nn.ReLU(),nn.Conv3d(16,32,3,padding=1),nn.InstanceNorm3d(32),nn.ReLU(),                       nn.Conv3d(32,32,3,padding=1),nn.Upsample(size=(11,11,11),mode='trilinear'),nn.Conv3d(32,32,3,padding=1),nn.InstanceNorm3d(64),nn.ReLU(),                        #nn.Conv3d(32,32,3,padding=1),nn.InstanceNorm3d(32),nn.ReLU(),\
                           nn.Conv3d(32,32,3,padding=1),nn.InstanceNorm3d(32),nn.ReLU(),nn.Conv3d(32,16,3,padding=1),nn.InstanceNorm3d(16),nn.ReLU(),nn.Conv3d(16,1,3,padding=1))

    heatmap.cuda()
    
    

    mesh = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda()*.3,(1,1,11,11,11)).half()
    patch = F.affine_grid(0.05*torch.eye(3,4).unsqueeze(0).cuda().half(),(1,1,3,3,3))

    optimizer = torch.optim.Adam(list(unet_model.parameters())+list(heatmap.parameters()),lr=0.001)#0.001
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,num_iterations//5,0.5)
    t0 = time.time()
    run_tre = torch.zeros(num_iterations)
    run_loss = torch.zeros(num_iterations)


    with tqdm(total=num_iterations, file=sys.stdout) as pbar:
        for i in range(num_iterations):
            ii = torch.randperm(len(training_))[:1]
            keypts_fix = keypts_all[int(ii)][0].cuda()
            keypts_mov = keypts_all[int(ii)][1].cuda()
            mind_fix = img_all[ii,:1].cuda().half()
            mind_mov = img_all[ii,1:].cuda().half()
            fixed_mask = mask_all[ii,:1].view(1,1,H,W,D).cuda().half()
            moving_mask = mask_all[ii,1:].view(1,1,H,W,D).cuda().half()

            #Affine augmentation of images *and* keypoints 
            if(i%2==0):
                A = (torch.randn(3,4)*.035+torch.eye(3,4)).cuda()
                affine = F.affine_grid(A.unsqueeze(0),(1,1,H//2,W//2,D//2))
                keypts_fix = torch.solve(torch.cat((keypts_fix,torch.ones(keypts_fix.shape[0],1).cuda()),1).t(),\
                                         torch.cat((A,torch.tensor([0,0,0,1]).cuda().view(1,-1)),0))[0].t()[:,:3]
                fixed_mask = F.grid_sample(mask_all[ii,:1].view(1,1,H,W,D).cuda().half(),affine.half())
                fixed_img = F.grid_sample(img_all[ii,:1].view(1,1,H//2,W//2,D//2).cuda().half(),affine.half())
                with torch.cuda.amp.autocast():
                    mind_fix = (fixed_mask.float()*fixed_img.float()).half()
                    #mind_fix = F.avg_pool3d(mind_fix_,grid_sp,stride=grid_sp)

            if(i%2==1):
                A = (torch.randn(3,4)*.035+torch.eye(3,4)).cuda()
                affine = F.affine_grid(A.unsqueeze(0),(1,1,H//2,W//2,D//2))
                keypts_mov = torch.solve(torch.cat((keypts_mov,torch.ones(keypts_mov.shape[0],1).cuda()),1).t(),\
                                         torch.cat((A,torch.tensor([0,0,0,1]).cuda().view(1,-1)),0))[0].t()[:,:3]
                moving_mask = F.grid_sample(mask_all[ii,1:].view(1,1,H,W,D).cuda().half(),affine.half())
                moving_img = F.grid_sample(img_all[ii,1:].view(1,1,H//2,W//2,D//2).cuda().half(),affine.half())
                with torch.cuda.amp.autocast():
                    mind_mov = (moving_mask.float()*moving_img.float()).half()
                    #mind_mov = F.avg_pool3d(mind_mov_,grid_sp,stride=grid_sp)

            disp_gt = keypts_mov-keypts_fix

            scheduler.step()
            optimizer.zero_grad()
            idx = torch.randperm(keypts_fix.shape[0])[:512]

            with torch.cuda.amp.autocast():
                #VoxelMorph requires some padding
                input = torch.cat((mind_fix,mind_mov),1)
                output = unet_model(input)#[:,:,4:-4,4:-4,2:-2]

                sample_xyz = keypts_fix[idx]#keypts_all_fix[int(ii)][idx]#fix
                #todo nearest vs bilinear
                sampled = F.grid_sample(output,sample_xyz.cuda().view(1,-1,1,1,3)+patch.view(1,1,-1,1,3),mode='bilinear')
                #sampled = F.grid_sample(output,sample_xyz.cuda().view(1,-1,1,1,3),mode='bilinear')
                disp_pred = heatmap(sampled.permute(2,1,0,3,4).view(sample_xyz.shape[0],-1,3,3,3))


                pred_xyz = torch.sum(torch.softmax(disp_pred.view(-1,11**3,1),1)*mesh.view(1,11**3,3),1)
                loss = nn.MSELoss()(pred_xyz,disp_gt[idx])


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            run_loss[i] = 100*loss.item()
            run_tre[i] = ((disp_gt[idx]-pred_xyz)*100).pow(2).sum(1).sqrt().mean()


            str1 = f"iter: {i}, loss: {'%0.3f'%run_loss[i-20:i-1].mean()}, kpTRE: {'%0.3f'%run_tre[i-20:i-1].mean()}"
            pbar.set_description(str1)
            pbar.update(1)

                
            if(i%50==49):
                logging.root.handlers = [logging.FileHandler("output/debug.log")]
                logging.info(str1)
                logging.root.handlers = [logging.FileHandler("output/debug.log"),logging.StreamHandler()]
            
    unet_model.cpu(); unet_model.eval()
    heatmap.cpu(); heatmap.eval()
    torch.save({'unet_model':unet_model.state_dict(),'heatmap':heatmap.state_dict()},'models/'+task_name+'_complete.pth')