#evaluation.py
import json
import nibabel as nib
import os.path
from pathlib import Path
import pandas
import argparse

from utils import *
from surface_distance import *

import os



def main(args):
    d_opt=vars(args)
    verbose=d_opt['verbose']
    DEFAULT_INPUT_PATH=Path(d_opt['input_path'])
    DEFAULT_GROUND_TRUTH_PATH=Path(d_opt['gt_path'])
    DEFAULT_CONFIG_JSON_PATH=Path(d_opt['config_path'])
    DEFAULT_EVALUATION_OUTPUT_FILE_PATH=Path(d_opt['output_path'])
    
    #read evaluation_config.json
    
    with open(DEFAULT_CONFIG_JSON_PATH, 'r') as f:
        data = json.load(f)

    name=data['task_name']
    expected_shape=np.array(data['expected_shape'])
    evaluation_methods_metrics=[tmp['metric'] for tmp in data['evaluation_methods']]
    if 'masked_evaluation' in data:
        use_mask = data['masked_evaluation']
    else:
        use_mask = False
    eval_pairs=data['eval_pairs']
    len_eval_pairs=len(eval_pairs)

    
    #Check if files are complete beforehand
    for idx, pair in enumerate(eval_pairs):
        disp_name='disp_{}_{}'.format(pair['fixed'][-16:-12], pair['moving'][-16:-12]+'.nii.gz')
        disp_path=os.path.join(DEFAULT_INPUT_PATH, disp_name)
        if os.path.isfile(disp_path):
            continue
        else:
            raise_missing_file_error(disp_name)

    #Dataframe for Case results
    cases_results=pandas.DataFrame()

    if verbose:
        print(f"Evaluate {len_eval_pairs} cases for: {[tmp['name'] for tmp in data['evaluation_methods']]}")
    for idx, pair in enumerate(eval_pairs):
        case_results={}

        fix_label_path=os.path.join(DEFAULT_GROUND_TRUTH_PATH, pair['fixed'].replace('images','labels'))
        mov_label_path=os.path.join(DEFAULT_GROUND_TRUTH_PATH, pair['moving'].replace('images','labels'))
        #with nii.gz
        disp_path=os.path.join(DEFAULT_INPUT_PATH, 'disp_{}_{}'.format(pair['fixed'][-16:-12], pair['moving'][-16:-12]+'.nii.gz'))
        disp_field=nib.load(disp_path).get_fdata()

        shape = np.array(disp_field.shape)
        if not np.all(shape==expected_shape):
            raise_shape_error(disp_name, shape, expected_shape)

        ## load neccessary files 
        if any([True for eval_ in ['tre'] if eval_ in evaluation_methods_metrics]):
            spacing_fix=nib.load(os.path.join(DEFAULT_GROUND_TRUTH_PATH, pair['fixed'])).header.get_zooms()[:3]
            spacing_mov=nib.load(os.path.join(DEFAULT_GROUND_TRUTH_PATH, pair['moving'])).header.get_zooms()[:3]

        if any([True for eval_ in ['dice','hd95'] if eval_ in evaluation_methods_metrics]):
            fixed_seg=nib.load(fix_label_path).get_fdata()
            moving_seg=nib.load(mov_label_path).get_fdata()
            D,H,W = fixed_seg.shape
            identity = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
            warped_seg = map_coordinates(moving_seg, identity + disp_field.transpose(3,0,1,2), order=0)
        
        
        if use_mask:
            mask_path= os.path.join(DEFAULT_GROUND_TRUTH_PATH, pair['fixed'].replace('images','masks'))
            if os.path.exists(mask_path):
                mask=nib.load(mask_path).get_fdata()
                mask_ready=True
            else:
                print(f'Tried to use mask but did not find {mask_path}. Will evaluate without mask.')
                mask_ready=False

                
        ## iterate over designated evaluation metrics
        for _eval in data['evaluation_methods']:
            _name=_eval['name']

            ### SDlogJ
            
            if 'sdlogj' == _eval['metric']:
                jac_det = (jacobian_determinant(disp_field[np.newaxis, :, :, :, :].transpose((0,4,1,2,3))) + 3).clip(0.000000001, 1000000000)
                log_jac_det = np.log(jac_det)
                if use_mask and mask_ready:
                    case_results[_name]=np.ma.MaskedArray(log_jac_det, 1-mask[2:-2, 2:-2, 2:-2]).std()
                else:
                    case_results[_name]=log_jac_det.std()



            ### DSC
            if 'dice' == _eval['metric']:
                labels = _eval['labels']
                dice = compute_dice(fixed_seg,moving_seg,warped_seg,labels)
                case_results[_name]=dice

            ### HD95
            if 'hd95' == _eval['metric']:
                labels = _eval['labels']
                hd95 = compute_hd95(fixed_seg,moving_seg,warped_seg,labels)
                case_results[_name]=hd95
        
            ### TRE
            if 'tre' == _eval['metric']:
                destination = _eval['dest']
                lms_fix_path = os.path.join(DEFAULT_GROUND_TRUTH_PATH, pair['fixed'].replace('images', destination).replace('.nii.gz','.csv'))
                lms_mov_path = os.path.join(DEFAULT_GROUND_TRUTH_PATH, pair['moving'].replace('images', destination).replace('.nii.gz','.csv'))
                fix_lms = np.loadtxt(lms_fix_path, delimiter=',')
                mov_lms = np.loadtxt(lms_mov_path, delimiter=',')
                tre = compute_tre(fix_lms, mov_lms, disp_field ,spacing_fix, spacing_mov)
                case_results[_name]=tre.mean()

        if verbose:
            print(f'case_results [{idx}]: {case_results}')
        cases_results=pandas.concat([cases_results, pandas.DataFrame(case_results, index=[0])], ignore_index=True)

        
        
    aggregated_results = {}   
    for col in cases_results.columns:
        aggregated_results[col] = {'30': cases_results[col].quantile(.3),
                                  'std': cases_results[col].std(),
                                  'mean': cases_results[col].mean()}
    final_results={
        name: {
            "case": cases_results.to_dict(),
            "aggregates": aggregated_results
    }}

    #print(f'aggregated_results [{name}]: {aggregated_results}')
    if verbose:
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
    parser.add_argument("-v","--verbose", dest="verbose", action='store_true', default=False)
    args= parser.parse_args()
    main(args)
    