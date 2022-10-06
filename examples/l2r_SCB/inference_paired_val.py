
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

gpu_id = 0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
print(torch.cuda.get_device_name())


from utils import *
import features
from features.segmodels import *
from features.unet import *
from foerstner import foerstner_kpts
from features.mind import *
import csv


if __name__ == "__main__":
    gpu_id = 0
    task_name = "NLST"
    print(f"Arguments count: {len(sys.argv)}")
    for i, arg in enumerate(sys.argv):
        if(i==1):
            print(f"task_name {i:>6}: {arg}")
            task_name = str(arg)
        elif(i==2):
            print(f"GPU ID {i:>6}: {arg}")
            gpu_id = int(arg)    
        else:
            print(f" argument {i:>6}: {arg}")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    print(torch.cuda.get_device_name())
    


    ##preload data

    data_path='../temp/HIDDENDATA/'+task_name+'/'
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


    unet_model = create_unet(2,(H//2,W//2,D//2))#
    unet_model.cuda()
    heatmap = nn.Sequential(nn.ConvTranspose3d(64,16,5),nn.InstanceNorm3d(16),nn.ReLU(),nn.Conv3d(16,32,3,padding=1),nn.InstanceNorm3d(32),nn.ReLU(),                       nn.Conv3d(32,32,3,padding=1),nn.Upsample(size=(11,11,11),mode='trilinear'),nn.Conv3d(32,32,3,padding=1),nn.InstanceNorm3d(64),nn.ReLU(),                        #nn.Conv3d(32,32,3,padding=1),nn.InstanceNorm3d(32),nn.ReLU(),\
                           nn.Conv3d(32,32,3,padding=1),nn.InstanceNorm3d(32),nn.ReLU(),nn.Conv3d(32,16,3,padding=1),nn.InstanceNorm3d(16),nn.ReLU(),nn.Conv3d(16,1,3,padding=1))

    heatmap.cuda()


    unet_model.cpu(); heatmap.cpu()
    models = torch.load('models/'+task_name+'_complete.pth')
    unet_model.load_state_dict(models['unet_model'])
    heatmap.load_state_dict(models['heatmap'])

    unet_model.cuda(); heatmap.cuda()
    unet_model.eval(); heatmap.eval()

    ##validation:

    img_val = torch.zeros(len(validation_),2,H,W,D)
    mask_val = torch.zeros(len(validation_),2,H,W,D).long()

    keypts_val = []

    print(H,W,D)

    t0=time.time()
    for ii,i in enumerate(tqdm(validation_)):

        mask_val[ii,0] = torch.from_numpy(nib.load(os.path.join(data_path,i['fixed'].replace('images','masks'))).get_fdata()).long()
        img_val[ii,0] = (torch.from_numpy(nib.load(os.path.join(data_path,i['fixed'])).get_fdata()).float().cuda().unsqueeze(0).unsqueeze(0)/500).cpu()
        mask_val[ii,1] = torch.from_numpy(nib.load(os.path.join(data_path,i['moving'].replace('images','masks'))).get_fdata()).long()
        img_val[ii,1] = (torch.from_numpy(nib.load(os.path.join(data_path,i['moving'])).get_fdata()).float().cuda().unsqueeze(0).unsqueeze(0)/500).cpu()


        corrfield1 = torch.empty(0,3)
        corrfield2 = torch.empty(0,3)

        hidden_path = data_path#'../temp/l2r_v11b/'
        with open(os.path.join(hidden_path,i['fixed'].replace('images','keypoints').replace('.nii.gz','.csv')), newline='') as csvfile:
            fread = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in fread:
                corrfield1 = torch.cat((corrfield1,torch.from_numpy(np.array(row).astype('float32')).float().view(1,3)),0)
        with open(os.path.join(hidden_path,i['moving'].replace('images','keypoints').replace('.nii.gz','.csv')), newline='') as csvfile:
            fread = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in fread:
                corrfield2 = torch.cat((corrfield2,torch.from_numpy(np.array(row).astype('float32')).float().view(1,3)),0)

        keypts_fix = torch.stack((corrfield1[:,2+0]/(D-1)*2-1,corrfield1[:,1+0]/(W-1)*2-1,corrfield1[:,0+0]/(H-1)*2-1),1).cuda()
        keypts_mov = torch.stack((corrfield2[:,2+0]/(D-1)*2-1,corrfield2[:,1+0]/(W-1)*2-1,corrfield2[:,0+0]/(H-1)*2-1),1).cuda()
        keypts_val.append(torch.stack((keypts_fix,keypts_mov),0))

    t1 = time.time()
    print('validation data loaded in %.2f s' % (t1-t0))





    mesh = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda()*.3,(1,1,11,11,11)).half()
    patch = F.affine_grid(0.05*torch.eye(3,4).unsqueeze(0).cuda().half(),(1,1,3,3,3))

    run_tre = torch.zeros(img_val.shape[0])
    print('starting Voxelmorph++ registration')# of '+fixed_file+' and '+moving_file)

    std_log = torch.empty(0,2)

    for idx in range(img_val.shape[0]):

        mask_fix = mask_val[idx:idx+1,:1].cuda().float()
        mask_mov = mask_val[idx:idx+1,1:].cuda().float()
        img_fix = img_val[idx:idx+1,:1].cuda()
        img_mov = img_val[idx:idx+1,1:].cuda()

        grid_sp = 2

        img_fix2 = F.avg_pool3d((mask_fix*img_fix).view(1,1,H,W,D).float(),grid_sp,stride=grid_sp)

        img_mov2 = F.avg_pool3d((mask_mov*img_mov).view(1,1,H,W,D).float(),grid_sp,stride=grid_sp)


        with torch.no_grad():
            with torch.cuda.amp.autocast():

                mind_fix_ = mask_fix.view(1,1,H,W,D).cuda().half()*\
                MINDSSC(img_fix.view(1,1,H,W,D).cuda(),1,2).half()
                mind_mov_ = mask_mov.view(1,1,H,W,D).cuda().half()*\
                MINDSSC(img_mov.view(1,1,H,W,D).cuda(),1,2).half()
                mind_fix = F.avg_pool3d(mind_fix_,grid_sp,stride=grid_sp)
                mind_mov = F.avg_pool3d(mind_mov_,grid_sp,stride=grid_sp)



        keypts_fix = keypts_val[idx][0]
        keypts_mov = keypts_val[idx][1]






        torch.cuda.synchronize()
        t0 = time.time()


        with torch.no_grad():
            with torch.cuda.amp.autocast():

                img_fix2 = img_fix2.cuda().half()
                img_mov2 = img_mov2.cuda().half()



                keypts_fix = keypts_fix.cuda()

                #keypts_mov = keypts_mov.cuda()
                #keypts_rand = 2*torch.rand(2048*24,3).cuda()-1
                #val = F.grid_sample(mask_fix.cuda(),keypts_rand.view(1,-1,1,1,3),align_corners=False)
                #idx1 = torch.nonzero(val.squeeze()==1).reshape(-1)

                #keypts_fix = keypts_rand[idx1[:1024*2]]


                input = torch.cat((img_fix2,img_mov2),1).cuda()
                output = unet_model(input)

                sample_xyz = keypts_fix
                #sampled = F.grid_sample(output,sample_xyz.cuda().view(1,-1,1,1,3),mode='bilinear',align_corners=False)
                sampled = F.grid_sample(output,sample_xyz.cuda().view(1,-1,1,1,3)+patch.view(1,1,-1,1,3),mode='bilinear')
                #disp_pred = heatmap(sampled.permute(2,1,0,3,4))

                disp_pred = heatmap(sampled.permute(2,1,0,3,4).view(keypts_fix.shape[0],-1,3,3,3))

                pred_xyz = torch.sum(torch.softmax(disp_pred.view(-1,11**3,1),1)*mesh.view(1,11**3,3),1)


        dense_flow_ = thin_plate_dense(keypts_fix.unsqueeze(0), pred_xyz.unsqueeze(0), (H, W, D), 4, 0.1)
        dense_flow = dense_flow_.flip(4).permute(0,4,1,2,3)*torch.tensor([H-1,W-1,D-1]).cuda().view(1,3,1,1,1)/2

        torch.cuda.synchronize()
        t1 = time.time()
        t_inf = t1-t0; t0=t1;


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
            lambda_weight = .7# with tps: .5, without:0.7
            for iter in range(60):#80
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


        disp_hr = torch.flip(disp_smooth/torch.tensor([H-1,W-1,D-1]).view(1,3,1,1,1).cuda()*2,[1])
        keypts_fix = keypts_val[idx][0]
        keypts_mov = keypts_val[idx][1]

        disp_gt = keypts_mov-keypts_fix

        pred_xyz = F.grid_sample(disp_hr.float(),keypts_fix.cuda().view(1,-1,1,1,3),mode='bilinear',\
                                 align_corners=False).squeeze().t()

        tre0 = (disp_gt*100).pow(2).sum(1).sqrt()
        tre1 = ((disp_gt-pred_xyz)*100).pow(2).sum(1).sqrt()
        #print('TRE',tre0.mean().item(),tre1.mean().item())

        run_tre[idx] = tre1.mean()


        torch.cuda.synchronize()
        t1 = time.time()
        t_adam = t1-t0; t0=t1

        #print('run time','%0.3f'%t_inf,'sec (net)','%0.3f'%t_adam,'sec (adam)')

        jdet = jacobian_determinant_3d(disp_hr)
        jac_det = torch.clamp(jdet + 3,0.000000001, 1000000000)

        log_jac_det = torch.log(jac_det[2:-2, 2:-2, 2:-2])


        std_log = torch.cat((std_log,torch.tensor([log_jac_det.std(),(jac_det<0).float().mean()]).view(1,-1)),0)
        #    std_log = torch.cat((std_log,torch.tensor([log_jac_det.std(),(jac_det<0).float().mean()]).view(1,-1)),0)

        #print(torch.tensor([log_jac_det.std(),(jac_det<0).float().mean()]).view(1,-1))
        disp_field = F.interpolate(disp_smooth,scale_factor = 0.5,mode='trilinear',align_corners=False)

        x1 = disp_field[0,0,:,:,:].cpu().float().data.numpy().astype('float16')
        y1 = disp_field[0,1,:,:,:].cpu().float().data.numpy().astype('float16')
        z1 = disp_field[0,2,:,:,:].cpu().float().data.numpy().astype('float16')

        np.savez_compressed('outputs/'+task_name+'/disp_'+str(int(val_list[idx].split('_')[1])).zfill(4)+'_'+str(int(val_list[idx].split('_')[1])).zfill(4)+'.npz',np.stack((x1,y1,z1),0))

    print('KP-TRE',run_tre.mean())
    print('std-log',std_log.mean(0))
#if(disp_file is not None):
#    torch.save(fitted_grid.data.cpu(),disp_file)




