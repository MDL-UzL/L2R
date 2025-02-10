"""
Evaluation Script for the L2R Challenge.
For details, please visit: https://github.com/MDL-UzL/L2R/
"""


import os
import json
import argparse
import nibabel as nib
from datetime import datetime
from utils import *
from surface_distance import *
from collections import OrderedDict


def evaluate_L2R(INPUT_PATH, GT_PATH, OUTPUT_PATH, JSON_PATH, verbose=False):
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    #check if metrics are well-defined
    if len(set([x['name'] for x in data['evaluation_methods']])) != len([x['name'] for x in data['evaluation_methods']]):
        raise ValueError("Evaluation method names have to be unique.")

    # Get necessary information from config file
    name = data['task_name']
    expected_shape = np.array(data['expected_shape'])
    metrics_dict = {x['name']: x for x in data['evaluation_methods']}
    evaluation_methods_metrics = [x['metric'] for x in metrics_dict.values()]
    eval_pairs = data['eval_pairs']
    len_eval_pairs = len(eval_pairs)
    if 'masked_evaluation' in data:
        use_mask = data['masked_evaluation']
    else:
        use_mask = False

    # Check if all files are available
    missing_files = []
    available_files = {}
    
    for pair in eval_pairs:
        fix_subject, fix_modality = pair['fixed'][-16:-12], pair['fixed'][-11:-7]
        mov_subject, mov_modality = pair['moving'][-16:-12], pair['moving'][-11:-7]

        disp_short_name = f"disp_{fix_subject}_{mov_subject}"
        disp_long_name = f"disp_{fix_subject}_{fix_modality}_{mov_subject}_{mov_modality}"

        #Allow short handles 
        short_name_allowed = (fix_modality == '0000' and mov_modality == '0001')

        found_file = None

        # Check short name if allowed
        if short_name_allowed and file_exists(INPUT_PATH, disp_short_name):
            found_file = disp_short_name

        # Check long name (always allowed)
        if found_file is None and file_exists(INPUT_PATH, disp_long_name):
            found_file = disp_long_name

        if found_file:
            available_files[(fix_subject, fix_modality, mov_subject, mov_modality)] = found_file
        else:
            missing_files.append(f"{disp_long_name} (or {disp_short_name} if allowed)")

    if missing_files:
        raise ValueError(f"Missing files: {missing_files}")



    ### Evaluation
    if verbose:
        print(
            f"Evaluate {len_eval_pairs} cases for: {[tmp['name'] for tmp in data['evaluation_methods']]}")
        if use_mask:
            print("Will use masks for evaluation.")
    cases_results = {}


    for idx, pair in enumerate(eval_pairs):
        case_results = {}
        fix_subject, fix_modality = pair['fixed'][-16:-12], pair['fixed'][-11:-7]
        mov_subject, mov_modality = pair['moving'][-16:-12], pair['moving'][-11:-7]

        fix_label_path = os.path.join(
            GT_PATH, pair['fixed'].replace('images', 'labels'))
        mov_label_path = os.path.join(
            GT_PATH, pair['moving'].replace('images', 'labels'))
       

        disp_filename = available_files.get((fix_subject, fix_modality, mov_subject, mov_modality))

        if disp_filename:
            for ext in ['.npz', '.nii.gz']: 
                full_path = os.path.join(INPUT_PATH, disp_filename + ext)
                if os.path.isfile(full_path):
                    disp_field = load_disp(full_path)
                    break 

        

        shape = np.array(disp_field.shape)
        if not np.all(shape == expected_shape):
            raise_shape_error(f'{fix_subject}_{fix_modality}-->{mov_subject}_{mov_modality}', shape, expected_shape) ##error here

        # load neccessary files
        if any([True for eval_ in ['tre'] if eval_ in evaluation_methods_metrics]):
            spacing_fix = nib.load(os.path.join(
                GT_PATH, pair['fixed'])).header.get_zooms()[:3]
            spacing_mov = nib.load(os.path.join(
                GT_PATH, pair['moving'])).header.get_zooms()[:3]

        if any([True for eval_ in ['dice', 'hd95'] if eval_ in evaluation_methods_metrics]):
            fixed_seg = nib.load(fix_label_path).get_fdata()
            moving_seg = nib.load(mov_label_path).get_fdata()
            D, H, W = fixed_seg.shape
            identity = np.meshgrid(np.arange(D), np.arange(
                H), np.arange(W), indexing='ij')
            warped_seg = map_coordinates(
                moving_seg, identity + disp_field.transpose(3, 0, 1, 2), order=0)

        if use_mask:
            mask_path = os.path.join(
                GT_PATH, pair['fixed'].replace('images', 'masks'))
            if os.path.exists(mask_path):
                mask = nib.load(mask_path).get_fdata()
                mask_ready = True
            else:
                print(
                    f'Tried to use mask but did not find {mask_path}. Will evaluate without mask.')
                mask_ready = False

        # iterate over designated evaluation metrics
        for _eval in data['evaluation_methods']:
            _name = _eval['name']
            # mean is one value, detailed is list
            # SDlogJ
            if 'sdlogj' == _eval['metric']:
                jac_det = (jacobian_determinant(disp_field[np.newaxis, :, :, :, :].transpose(
                    (0, 4, 1, 2, 3))) + 3).clip(0.000000001, 1000000000)
                log_jac_det = np.log(jac_det)
                if use_mask and mask_ready:
                    single_value = np.ma.MaskedArray(
                        log_jac_det, 1-mask[2:-2, 2:-2, 2:-2]).std()
                else:
                    single_value = log_jac_det.std()
                num_foldings = (jac_det <= 0).astype(float).sum()

                case_results[_name] = {
                    'mean': single_value, 'detailed': single_value}
                case_results['num_foldings'] = {
                    'mean': num_foldings, 'detailed': num_foldings}

            # DSC
            if 'dice' == _eval['metric']:
                labels = _eval['labels']
                mean, detailed = compute_dice(
                    fixed_seg, moving_seg, warped_seg, labels)
                case_results[_name] = {'mean': mean, 'detailed': detailed}

            # HD95
            if 'hd95' == _eval['metric']:
                labels = _eval['labels']
                mean, detailed = compute_hd95(
                    fixed_seg, moving_seg, warped_seg, labels)
                case_results[_name] = {'mean': mean, 'detailed': detailed}

            # TRE
            if 'tre' == _eval['metric']:
                def get_keypoint_path(image_path):
                    return os.path.join(GT_PATH, image_path.replace('images', destination).replace('.nii.gz', '.csv'))
                ## corrfield correspondences are calculated for corresponding images
                ## therefore, if modalities are different, the keypoint paths have to be changed
                ## if same modalities : keypointsTr / keypointsTs
                ## if different modalities: keypoints01Tr / keypoints02Tr 

                destination = _eval['dest']
                # If different modalities, adjust keypoint paths
                if not (fix_modality == mov_modality or (fix_modality == '0000' and mov_modality == '0001')):
                    modality_suffix = f"{min(fix_modality, mov_modality)}{max(fix_modality, mov_modality)}"
                    destination += modality_suffix


                lms_fix_path = get_keypoint_path(pair['fixed'])
                lms_mov_path = get_keypoint_path(pair['moving'])
                    
                fix_lms = np.loadtxt(lms_fix_path, delimiter=',')
                mov_lms = np.loadtxt(lms_mov_path, delimiter=',')
                tre = compute_tre(fix_lms, mov_lms, disp_field,
                                  spacing_fix, spacing_mov)

                case_results[_name] = {'mean': tre.mean(), 'detailed': tre.tolist()}

        cases_results[f'{fix_subject}_{fix_modality}<--{mov_subject}_{mov_modality}'] = case_results
        if verbose:
            print(
                f"case_results [{idx}] [{fix_subject}_{fix_modality}<--{mov_subject}_{mov_modality}']:")
            for k, v in case_results.items():
                print(f"\t{k: <{20}}: {v['mean']:.5f}")

    aggregated_results = {}

    for metric_name in metrics_dict.keys():

        for k, v in cases_results.items():

            # calculate mean of all cases
            all_means_metric = np.array(
                [cases_results[k][metric_name]['mean'] for k in cases_results.keys()])
            aggregated_results[metric_name] = {'mean': all_means_metric.mean(),
                                          'std': all_means_metric.std(),
                                          ## if metric in [sdlogj, tre] quantile 0.7
                                          ## it metric in [dice, hd95] quantile 0.3
                                         '30': np.quantile(all_means_metric, .7) if metrics_dict[metric_name]['metric'] in ['sdlogj', 'tre'] else np.quantile(all_means_metric, .3)}
                                         
    if verbose:
        # print aggregated results
        print("\n aggregated_results:")
        for k, v in aggregated_results.items():
            print(f"\t{k: <{20}}: {v['mean']:.5f} +- {v['std']:.5f} | 30%: {v['30']:.5f}")

    if os.path.isdir(OUTPUT_PATH):
        OUTPUT_PATH = os.path.join(OUTPUT_PATH, datetime.now().strftime('%Y%m%d_%H%M')+'.json')

    with open(os.path.join(OUTPUT_PATH), 'w', encoding='utf-8') as f:
        json.dump(OrderedDict({'name': name,
                   'aggregates': aggregated_results,
                   'cases': cases_results,
                   'eval_version': '2.0'}), f, indent=4, allow_nan=True)
    
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='L2R Evaluation script\n'
                                     'Docker PATHS:\n'
                                     'DEFAULT_INPUT_PATH = Path("/input/")\n'
                                     'DEFAULT_GROUND_TRUTH_PATH = Path("/opt/evaluation/ground-truth/")\n'
                                     'DEFAULT_EVALUATION_OUTPUT_FILE_PATH = Path("/output/metrics.json")')
    parser.add_argument("-i", "--input", dest="input_path",
                        help="path to deformation_field", default="test")
    parser.add_argument("-d", "--data", dest="gt_path",
                        help="path to data", default="ground-truth")
    parser.add_argument("-o", "--output", dest="output_path",
                        help="path to write results(e.g. 'results/metrics.json')", default="metrics.json")
    parser.add_argument("-c", "--config", dest="config_path",
                        help="path to config json-File (e.g. 'evaluation_config.json')", default='ground-truth/evaluation_config.json')
    parser.add_argument("-v", "--verbose", dest="verbose",
                        action='store_true', default=False)
    args = parser.parse_args()
    evaluate_L2R(args.input_path, args.gt_path, args.output_path,
                 args.config_path, args.verbose)
