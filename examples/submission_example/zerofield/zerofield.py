#!/usr/bin/env python

import argparse
import json
import numpy as np
import nibabel as nib
import os

def main(dataset_json):

    with open(dataset_json, 'r') as f:
        dataset = json.load(f)
    
    D, H, W = dataset['tensorImageShape']['0']


    num_pairs = dataset['numRegistration_test']
    pairs = dataset['registration_test']
        
    for i in range(num_pairs):
        pair = pairs[i]
        fix_path = os.path.join('data',pair['fixed'])
        mov_path = os.path.join('data',pair['moving'])
        fix_id = os.path.basename(fix_path).split('_')[1]
        mov_id = os.path.basename(mov_path).split('_')[1]
        disp_path = os.path.join('output','disp_{}_{}.nii.gz'.format(fix_id, mov_id))
        
        fix = nib.load(fix_path)
        mov = nib.load(mov_path)
        disp = np.zeros((D, H, W, 3))
        
        nib.save(nib.Nifti1Image(disp, np.eye(4)), disp_path)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'zerofield args')
    parser.add_argument('dataset_json', help='path to dataset_json')
    parser.add_argument("-t","--test_data", dest="test", action='store_true', default=False, help='use test data')
    args = parser.parse_args()
    main(args.dataset_json)