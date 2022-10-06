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
from features.unet import *
import logging



#only use for non-CT
def nonCTnorm(img):
    mask = img>0
    mean_intensity = img[mask].mean()
    std_intensity = img[mask].std()
    img = (img - mean_intensity) / (std_intensity + 1e-8)
    img[mask == 0] = 0
    return img



if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s",\
        handlers=[logging.FileHandler("output/debug.log"),logging.StreamHandler()])
    
    gpu_id = 0
    num_iterations = 5000
    dataset_filename = "../Learn2Reg_Dataset_release_v1.1/AbdomenCTCT_dataset.json"
    print(f"Arguments count: {len(sys.argv)}")
    for i, arg in enumerate(sys.argv):
        if(i==1):
            print(f"dataset_filename {i:>6}: {arg}")
            dataset_filename = str(arg)
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
    print(torch.cuda.get_device_name())
    
    data_path = os.path.join(*dataset_filename.split('/')[:-1])
    if(dataset_filename[0]=='/'):
        folder = '/'+folder+'/'
#    print(folder)
    dataset_file = dataset_filename.split('/')[-1]
    task_name = dataset_file.split('_')[0]
    ##preload data
    logging.info('reading dataset from '+data_path+'/'+task_name+'_dataset.json')
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
    logging.info('memory allocated')
    t0=time.time()
    for ii,i in enumerate(tqdm(training_)):
        seg_all[ii] = torch.from_numpy(nib.load(os.path.join(data_path,i['label'])).get_fdata()).long()
        img = torch.from_numpy(nib.load(os.path.join(data_path,i['image'])).get_fdata()).float().cuda().unsqueeze(0).unsqueeze(0)
        if not('CT' in dataset_info['modality']):
            img = nonCTnorm(img)
        else:
            img /= 500
        img_all[ii] = F.avg_pool3d(img,2).cpu()
        #img_all[ii] = F.avg_pool3d(torch.from_numpy(nib.load(os.path.join(data_path,i['image'])).get_fdata()).float().cuda().unsqueeze(0).unsqueeze(0)/500,2).cpu()
    t1 = time.time()
    logging.info('training data loaded in %.2f s' % (t1-t0))

    ##validation
    seg_val = torch.zeros(num_val,H,W,D).long()
    img_val = torch.zeros(num_val,1,H//2,W//2,D//2)

    t0 = time.time()
    for ii,i in enumerate(tqdm(val_list)):
        seg_val[ii] = torch.from_numpy(nib.load(os.path.join(data_path,i.replace('image','label'))).get_fdata()).long()
        img = torch.from_numpy(nib.load(os.path.join(data_path,i)).get_fdata()).float().cuda().unsqueeze(0).unsqueeze(0)
        if not('CT' in dataset_info['modality']):
            img = nonCTnorm(img)
        else:
            img /= 500
        img_val[ii] = F.avg_pool3d(img,2).cpu()
        #img_val[ii] = F.avg_pool3d(torch.from_numpy(nib.load(os.path.join(data_path,i)).get_fdata()).float().cuda().unsqueeze(0).unsqueeze(0)/500,2).cpu()
    t1 = time.time()
    logging.info('validation data loaded in %.2f s' % (t1-t0))


    label = torch.from_numpy(nib.load(os.path.join(data_path,training_[0]['label'])).get_fdata()).long()#.argmax()
    max_label = int(label.reshape(-1).max()+1)
    logging.info('#num_label '+str(max_label))



    logging.info('loading precomputed segmentation network')
    resnet = UNet_Segment(max_label)
    
    
    resnet.load_state_dict(torch.load('models/'+task_name+'_segment.pth')['resnet'])
    resnet.eval()
    resnet.cuda()
    print()
    
    #semantic_features:
    logging.info('inferencing segmentation model')

    unet_all = torch.zeros(img_all.shape[0],H,W,D)

    for i in tqdm(range(img_all.shape[0])):
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                affine = torch.eye(3,4).unsqueeze(0).cuda()
                affine2 = F.affine_grid(affine,(1,1,H,W,D),align_corners=False)
                img_fix = F.grid_sample(img_all[i:i+1].cuda().half(),affine2[:1],align_corners=False)
                feat = resnet(img_fix)
                predict = F.interpolate(feat,scale_factor=2,mode='trilinear').argmax(1).contiguous()
                labels_out = predict#largest_components(predict)
                unet_all[i] = labels_out
                

    unet_val = torch.zeros(img_val.shape[0],H,W,D)
    resnet.eval()
    for i in tqdm(range(img_val.shape[0])):
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                affine = torch.eye(3,4).unsqueeze(0).cuda()#+torch.randn(4,3,4).cuda()*0#scale_affine
                affine2 = F.affine_grid(affine,(1,1,H,W,D),align_corners=False)
                img_fix = F.grid_sample(img_val[i:i+1].cuda().half(),affine2[:1],align_corners=False)
                feat = resnet(img_fix)
                predict = F.interpolate(feat,scale_factor=2,mode='trilinear').argmax(1).contiguous()
                labels_out = predict
                unet_val[i] = labels_out
                
     #allocate memory
    unetlabel_all = []
    label_all = []
    stride = 1
    if(img_all.shape[0]>50):
        stride = 2 #save memory to avoid allocation errors
    t0 = time.time()
    for i in tqdm(range(img_all.shape[0])):
        unet1 = F.avg_pool3d(F.avg_pool3d(F.one_hot(unet_all[i:i+1].cuda().long(),max_label).float().permute(0,4,1,2,3),2),stride).cpu()
        unetlabel_all.append(unet1.half().pin_memory())
        label1 = F.avg_pool3d(F.avg_pool3d(F.one_hot(seg_all[i:i+1].long().cuda(),max_label).float().permute(0,4,1,2,3),2),stride).cpu()
        label_all.append(label1.half().pin_memory())
    t1 = time.time()
    print(t1-t0,'time memory')


    unetlabel_val = []
    label_val = []
    stride = 1
    if(img_all.shape[0]>50):
        stride = 2 #save memory to avoid allocation errors
    t0 = time.time()
    for i in tqdm(range(img_val.shape[0])):
        unet1 = F.avg_pool3d(F.avg_pool3d(F.one_hot(unet_val[i:i+1].cuda().long(),max_label).float().permute(0,4,1,2,3),2),stride).cpu()
        unetlabel_val.append(unet1.half().pin_memory())
        label1 = F.avg_pool3d(F.avg_pool3d(F.one_hot(seg_val[i:i+1].long().cuda(),max_label).float().permute(0,4,1,2,3),2),stride).cpu()
        label_val.append(label1.half().pin_memory())
    t1 = time.time()
    print(t1-t0,'time memory')
    
    weight = torch.bincount(seg_all[0].long().reshape(-1))
    for i in range(1,seg_all.shape[0]):
        weight += torch.bincount(seg_all[i].long().reshape(-1))

    weight = weight.float().pow(-.35).cuda()
    weight /= weight.mean() 
    print('label_weight',weight)
    
    logging.info('setting up registration network')
    num_classes = max_label
    print(H,W,D)
    inshape = (H//2,W//2,D//2)


    unet_model = create_unet(num_classes*2,inshape)#
    unet_model.cuda()
    heatmap = nn.Sequential(nn.ConvTranspose3d(64,16,5),nn.InstanceNorm3d(16),nn.ReLU(),nn.Conv3d(16,32,3,padding=1),nn.InstanceNorm3d(32),nn.ReLU(),                       nn.Conv3d(32,32,3,padding=1),nn.Upsample(size=(11,11,11),mode='trilinear'),nn.Conv3d(32,32,3,padding=1),nn.InstanceNorm3d(64),nn.ReLU(),                        #nn.Conv3d(32,32,3,padding=1),nn.InstanceNorm3d(32),nn.ReLU(),\
                           nn.Conv3d(32,32,3,padding=1),nn.InstanceNorm3d(32),nn.ReLU(),nn.Conv3d(32,16,3,padding=1),nn.InstanceNorm3d(16),nn.ReLU(),nn.Conv3d(16,1,3,padding=1))

    heatmap.cuda()
    regressor = nn.Sequential(nn.Conv3d(64,32,3,padding=1),nn.InstanceNorm3d(32),nn.ReLU(),nn.Conv3d(32,3,3,padding=1),nn.Tanh())
    regressor.cuda()

   

    mesh = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda()*.3,(1,1,11,11,11),align_corners=False).half()
    patch = F.affine_grid(0.05*torch.eye(3,4).unsqueeze(0).cuda().half(),(1,1,3,3,3),align_corners=False)





    for repeat in range(2):
        #repeat twice
        ##Segmentation pre-training
        resnet.cuda()
        half_iterations = num_iterations//2

        
        unet_model.cuda(); regressor.cuda(); heatmap.cuda()


        optimizer = torch.optim.Adam(list(unet_model.parameters())+list(heatmap.parameters())+list(regressor.parameters()),lr=0.002)

        scaler = torch.cuda.amp.GradScaler()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,half_iterations//5,0.5)
        run_loss = torch.zeros(half_iterations)
        run_regular = torch.zeros(half_iterations)
        run_dice = torch.zeros(half_iterations)
        t1 = 0
        with tqdm(total=half_iterations, file=sys.stdout) as pbar:
            for i in range(half_iterations):

                torch.cuda.synchronize()
                t0 = time.time()
                optimizer.zero_grad()


                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        ridx = torch.randperm(img_all.shape[0])[:2]
                        idx0 = ridx[:1]
                        idx1 = ridx[1:2]
                        input_fix = img_all[idx0].cuda()#
                        input_fix2 = unetlabel_all[int(idx0)].cuda()
                        input_mov = img_all[idx1].cuda()#
                        input_mov2 = unetlabel_all[int(idx1)].cuda()
                        label0 = label_all[int(idx0)].cuda()
                        label1 = label_all[int(idx1)].cuda()

                        A1 = (torch.randn(3,4)*.035+torch.eye(3,4)).cuda()
                        affine = F.affine_grid(A1.unsqueeze(0),(1,1,H//2,W//2,D//2),align_corners=False)


                        nnunet_fix = F.grid_sample(input_fix.cuda().half(),affine.half(),\
                                                   padding_mode='border',align_corners=False)
                        nnunet_fix2 = F.grid_sample(input_fix2.cuda().half(),affine.half(),\
                                                    padding_mode='border',align_corners=False)
                        label_fix = F.grid_sample(label0.half(),affine.half(),padding_mode='border',align_corners=False)


                        keypts_fix = weighted_sample(F.interpolate(nnunet_fix2,scale_factor=2,\
                                                                   mode='trilinear').argmax(1),2*512,H,W,D)


                        A2 = (torch.randn(3,4)*.035+torch.eye(3,4)).cuda()
                        affine = F.affine_grid(A2.unsqueeze(0),(1,1,H//2,W//2,D//2),align_corners=False)
                        nnunet_mov = F.grid_sample(input_mov.cuda().half(),affine.half(),\
                                                   padding_mode='border',align_corners=False)
                        nnunet_mov2 = F.grid_sample(input_mov2.cuda().half(),affine.half(),\
                                                    padding_mode='border',align_corners=False)
                        label_mov = F.grid_sample(label1.half(),affine.half(),\
                                                  padding_mode='border',align_corners=False)

                torch.cuda.synchronize()
                t1 += (time.time()-t0)


                with torch.cuda.amp.autocast():
        #            input = torch.cat((nnunet_fix,nnunet_mov),1).cuda()
                    input = torch.cat((nnunet_fix2,nnunet_mov2),1).cuda()

                    output = unet_model(input)

                    sample_xyz = keypts_fix
                    sampled = F.grid_sample(output,sample_xyz.cuda().view(1,-1,1,1,3)+patch.view(1,1,-1,1,3),mode='bilinear')
                    disp_pred = heatmap(sampled.permute(2,1,0,3,4).view(keypts_fix.shape[0],-1,3,3,3))
                    pred_xyz = .3*F.avg_pool3d(F.avg_pool3d(regressor(output),5,stride=1,padding=2),5,stride=1,padding=2)

                    soft_pred = torch.softmax(disp_pred.view(-1,11**3,1),1)
                    mesh_nnunet_mov = torch.sum(soft_pred.view(1,1,-1,11**3,1)*F.grid_sample(label_mov,sample_xyz.cuda().view(1,-1,1,1,3)+mesh.view(1,1,-1,1,3),mode='bilinear'),3)
                    sampled_nnunet_fix = F.grid_sample(label_fix,sample_xyz.cuda().view(1,-1,1,1,3)).squeeze(2).squeeze(-1)
                    loss1 = nn.MSELoss()(weight.view(1,-1,1,1,1)*mesh_nnunet_mov,weight.view(1,-1,1,1,1)*sampled_nnunet_fix)
                    #loss1 += jacobian_determinant_3d(pred_xyz).std()


                    warped = F.grid_sample(label_mov,F.interpolate(pred_xyz,scale_factor=2,mode='trilinear').permute(0,2,3,4,1)+F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,H//2,W//2,D//2)),padding_mode='border')
                    loss2 = 100*nn.MSELoss()(weight.view(1,-1,1,1,1)*label_fix,weight.view(1,-1,1,1,1)*warped)
                    #reg_norm = (L_graph.squeeze().mm(pred_xyz)).norm()/400
                    loss = loss2+loss1
                run_dice[i] = (dice_coeff(label_fix.argmax(1).squeeze().contiguous(),warped.argmax(1).squeeze().contiguous(),max_label).mean())


                run_regular[i] = 20*(loss2.item())
                run_loss[i] = 100*(loss1.item())#-reg_norm
                str1 = f"iter: {i}, loss: {'%0.3f'%run_loss[i-20:i-1].mean()}, warping: {'%0.3f'%run_regular[i-20:i-1].mean()}, dice: {'%0.3f'%run_dice[i-20:i-1].mean()}, time(preprocess): {'%0.3f'%(t1)} sec"
                pbar.set_description(str1)
                pbar.update(1)
                
                if(i%50==49):
                    logging.root.handlers = [logging.FileHandler("output/debug.log")]
                    logging.info(str1)
                    logging.root.handlers = [logging.FileHandler("output/debug.log"),logging.StreamHandler()]



                scaler.scale(loss).backward()


                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
        resnet.cpu(); unet_model.cpu(); heatmap.cpu(); regressor.cpu()
        resnet.eval(); heatmap.eval(); unet_model.eval(); regressor.eval()


        logging.info('saving model')
        torch.save({'resnet':resnet.state_dict(),'heatmap':heatmap.state_dict(),'regressor':regressor.state_dict(),\
                    'unet_model':unet_model.state_dict()},'models/'+task_name+'_complete.pth')
        #
