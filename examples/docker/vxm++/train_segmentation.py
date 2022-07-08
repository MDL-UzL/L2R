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
    num_classes, modality_count, label_exist, mask_exist, img_all, seg_all, mask_all = prepare_data(dataset_json)
    
    if(label_exist):
        for modality in range(modality_count):
            mod_str = str(modality).zfill(4)+'.nii.gz'
            
            print('training segmentation model for modality',mod_str,'#label',num_classes)
            if(os.path.exists('models/network_'+str(modality).zfill(4)+'.pth')):
                pass
            else:
                network = train_lraspp3d(img_all[modality],seg_all[modality],num_classes,num_iterations)
                torch.save(network.state_dict(),'models/network_'+str(modality).zfill(4)+'.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'vxm++ args')
    parser.add_argument('dataset_json', help='path to dataset_json')
    parser.add_argument('num_iterations', help='number of training iterations')
    args = parser.parse_args()

    main(args.dataset_json, int(args.num_iterations))