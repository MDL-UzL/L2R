#evaluation.py
import json
import nibabel as nib
import os.path
from pathlib import Path
import pandas
import argparse

from utils import *
from scipy.ndimage import map_coordinates, zoom
from surface_distance import *

import os



def main(args):
    d_opt=vars(args)

    DEFAULT_INPUT_PATH=Path(d_opt['input_path'])
    DEFAULT_GROUND_TRUTH_PATH=Path(d_opt['gt_path'])
    DEFAULT_CONFIG_JSON_PATH=Path(d_opt['config_path'])
    DEFAULT_EVALUATION_OUTPUT_FILE_PATH=Path(d_opt['output_path'])
    
    #read evaluation_config.json
    with open(DEFAULT_CONFIG_JSON_PATH, 'r') as f:
        data = json.load(f)

    name=data['task_name']
    expected_shape=np.array(data['expected_shape'])
    evaluation_methods=list(data['evaluation_methods'].keys())
    eval_pairs=data['eval_pairs']
    len_eval_pairs=len(eval_pairs)

    
    #Check if files are complete beforehand
    for idx, pair in enumerate(eval_pairs):
        disp_name='disp_{}_{}'.format(pair['fixed'][-11:-7], pair['moving'][-11:])
        disp_path=os.path.join(DEFAULT_INPUT_PATH, disp_name)
        if os.path.isfile(disp_path):
            continue
        else:
            raise_missing_file_error(disp_name)

    #Dataframe for Case results
    cases_results=pandas.DataFrame()


    print(f"Evaluate for: {evaluation_methods}")
    for idx, pair in enumerate(eval_pairs):
        ## get paths
        #print('case',idx)
        case_results={}

        fix_label_path=os.path.join(DEFAULT_GROUND_TRUTH_PATH, pair['fixed'].replace('images','labels'))
        mov_label_path=os.path.join(DEFAULT_GROUND_TRUTH_PATH, pair['moving'].replace('images','labels'))
        #with nii.gz
        disp_path=os.path.join(DEFAULT_INPUT_PATH, 'disp_{}_{}'.format(pair['fixed'][-11:-7], pair['moving'][-11:]))
        disp_field=nib.load(disp_path).get_fdata()

        shape = np.array(disp_field.shape)
        if not (shape==expected_shape).all():
            raise_shape_error(disp_name, shape, expected_shape)

        ## load files 
        if any([True for eval_ in ['dice','dice_secret','hd95','hd95_secret', 'tre'] if eval_ in evaluation_methods]):
            spacing_fix=nib.load(os.path.join(DEFAULT_GROUND_TRUTH_PATH, pair['fixed'])).header.get_zooms()[:3]
            spacing_mov=nib.load(os.path.join(DEFAULT_GROUND_TRUTH_PATH, pair['moving'])).header.get_zooms()[:3]

        if any([True for eval_ in ['dice','dice_secret','hd95','hd95_secret'] if eval_ in evaluation_methods]):
            fixed_seg=nib.load(fix_label_path).get_fdata()
            moving_seg=nib.load(mov_label_path).get_fdata()
            warped_seg=warp(fixed_seg,moving_seg,disp_field,spacing_fix,spacing_mov)
                

        ### SDlogJ
        if 'sdlogj' in evaluation_methods:
            jac_det = (jacobian_determinant(disp_field[np.newaxis, :, :, :, :].transpose((0,4,1,2,3))) + 3).clip(0.000000001, 1000000000)
            log_jac_det = np.log(jac_det)
            case_results['LogJacDetStd']=log_jac_det.std()

        ### DSC
        if 'dice' in evaluation_methods:
            labels = data['evaluation_methods']['dice']
            dice = compute_dice(fixed_seg,moving_seg,warped_seg,labels)
            case_results['DiceCoefficient']=dice
        
        ### DSC_Secret
        if 'dice_secret' in evaluation_methods:
            labels = data['evaluation_methods']['dice_secret']
            dice_secret = compute_dice(fixed_seg,moving_seg,warped_seg,labels)
            case_results['DiceCoefficient_Secret']=dice_secret

        ### HD95
        if 'hd95' in evaluation_methods:
            labels = data['evaluation_methods']['hd95']
            hd95 = compute_hd95(fixed_seg,moving_seg,warped_seg,labels)
            case_results['HausdorffDistance95']=hd95
        
        ### HD95_Secret
        if 'hd95_secret' in evaluation_methods:
            labels = data['evaluation_methods']['hd95_secret']
            hd95_secret = compute_hd95(fixed_seg,moving_seg,warped_seg,labels)
            case_results['HausdorffDistance95_Secret']=hd95_secret
        
        ### TRE
        if 'tre' in evaluation_methods:
            lms_fix_path = os.path.join(DEFAULT_GROUND_TRUTH_PATH, pair['fixed'].replace('images', 'landmarks').replace('.nii.gz','.csv'))
            lms_mov_path = os.path.join(DEFAULT_GROUND_TRUTH_PATH, pair['moving'].replace('images', 'landmarks').replace('.nii.gz','.csv'))
            fix_lms = np.loadtxt(lms_fix_path, delimiter=',')
            mov_lms = np.loadtxt(lms_mov_path, delimiter=',')
            tre = compute_tre(fix_lms, mov_lms, disp_field ,spacing_fix, spacing_mov)
            case_results['TRE']=tre.mean()
        
        print(f'case_results [{idx}]: {case_results}')
        cases_results=pandas.concat([cases_results, pandas.DataFrame(case_results, index=[0])], ignore_index=True)

    aggregated_results={}
    aggregated_results['mean'] = cases_results.mean().to_dict()
    aggregated_results['std'] = cases_results.std().to_dict()
    aggregated_results['30'] = cases_results.quantile(.3).to_dict()
    final_results={
        name: {
            "case": cases_results.to_dict(),
            "aggregates": aggregated_results
    }}

    #print(f'aggregated_results [{name}]: {aggregated_results}')
    print(json.dumps(aggregated_results, indent=4))
    
    with open(os.path.join(DEFAULT_EVALUATION_OUTPUT_FILE_PATH), 'w') as f:
        json.dump(final_results, f, indent=4)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description='L2R Evaluation script\n'\
    'Docker PATHS:\n'\
    'DEFAULT_INPUT_PATH = Path("/input/")\n'\
    'DEFAULT_GROUND_TRUTH_PATH = Path("/opt/evaluation/ground-truth/")\n'\
    'DEFAULT_EVALUATION_OUTPUT_FILE_PATH = Path("/output/metrics.json")')
    parser.add_argument("-i","--input", dest="input_path", help="path to deformation_field", default="test")
    parser.add_argument("-d","--data", dest="gt_path", help="path to data", default="ground-truth")
    parser.add_argument("-o","--output", dest="output_path", help="path to write results(e.g. 'results/metrics.json')", default="metrics.json")
    parser.add_argument("-c","--config", dest="config_path", help="path to config json-File (e.g. 'evaluation_config.json')", default='ground-truth/evaluation_config.json')   
    args= parser.parse_args()
    main(args)
    