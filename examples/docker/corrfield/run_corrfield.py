#!/usr/bin/env python

import argparse
import json
import numpy as np
import nibabel as nib
import os
import sys
sys.path.append('./src')
from corrfield import corrfield
import torch

def main(dataset_json):
    
    with open(dataset_json, 'r') as f:
        dataset = json.load(f)
    
    D, H, W = dataset['tensorImageShape']['0']
    num_pairs = dataset['numRegistration_val']
    pairs = dataset['registration_val']
    
    for i in range(num_pairs):
        pair = pairs[i]
        fix_path = os.path.join('data',pair['fixed'])
        mov_path = os.path.join('data',pair['moving'])
        fix_id = os.path.basename(fix_path).split('_')[1]
        mov_id = os.path.basename(mov_path).split('_')[1]
        disp_path = os.path.join('output','disp_{}_{}.nii.gz'.format(fix_id, mov_id))
        
        fix = torch.from_numpy(nib.load(fix_path).get_fdata()).cuda().unsqueeze(0).unsqueeze(0).float()
        fix_mask = torch.from_numpy(nib.load(fix_path.replace('images','masks')).get_fdata()).cuda().unsqueeze(0).unsqueeze(0).float()
        mov = torch.from_numpy(nib.load(mov_path).get_fdata()).cuda().unsqueeze(0).unsqueeze(0).float()
    
        disp, _, _ = corrfield(fix, fix_mask, mov, 2.5, 150, 5, 1, 3, 0, 1, 1.4, 0.8, True, [16,8], [6,3], [2,1], [3,2], ['n','n'])
        nib.save(nib.Nifti1Image(disp[0].cpu().numpy(), np.eye(4)), disp_path)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'zerofield args')
    parser.add_argument('dataset_json', help='path to dataset_json')
    args = parser.parse_args()

    main(args.dataset_json)