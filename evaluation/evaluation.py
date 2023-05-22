"""Evaluation Script for the L2R Challenge.
For details, please visit: https://github.com/MDL-UzL/L2R/
"""


import os
import json
import argparse
import nibabel as nib
from utils import *
from surface_distance import *


def evaluate_L2R(INPUT_PATH, GT_PATH, OUTPUT_PATH, JSON_PATH, verbose=False):
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    name = data['task_name']
    expected_shape = np.array(data['expected_shape'])
    evaluation_methods_metrics = [tmp['metric']
                                  for tmp in data['evaluation_methods']]
    if 'masked_evaluation' in data:
        use_mask = data['masked_evaluation']
    else:
        use_mask = False
    eval_pairs = data['eval_pairs']
    len_eval_pairs = len(eval_pairs)

    # Check if files are available beforehand
    for idx, pair in enumerate(eval_pairs):
        disp_name = f"disp_{pair['fixed'][-16:-12]}_{pair['moving'][-16:-12]}.nii.gz"
        disp_path = os.path.join(INPUT_PATH, disp_name)
        if os.path.isfile(disp_path):
            continue
        else:
            raise_missing_file_error(disp_name)
    if verbose:
        print(
            f"Evaluate {len_eval_pairs} cases for: {[tmp['name'] for tmp in data['evaluation_methods']]}")
    if use_mask and verbose:
        print("Will use masks for evaluation.")
    cases_results = {}
    for idx, pair in enumerate(eval_pairs):
        case_results = {}
        pairs_string = [pair['fixed'][-16:-12], pair['moving'][-16:-12]]

        fix_label_path = os.path.join(
            GT_PATH, pair['fixed'].replace('images', 'labels'))
        mov_label_path = os.path.join(
            GT_PATH, pair['moving'].replace('images', 'labels'))
        # with nii.gz
        disp_path = os.path.join(INPUT_PATH, 'disp_{}_{}'.format(
            pairs_string[0], pairs_string[1]+'.nii.gz'))
        disp_field = nib.load(disp_path).get_fdata()

        shape = np.array(disp_field.shape)
        if not np.all(shape == expected_shape):
            raise_shape_error(disp_name, shape, expected_shape)

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
                destination = _eval['dest']
                lms_fix_path = os.path.join(GT_PATH, pair['fixed'].replace(
                    'images', destination).replace('.nii.gz', '.csv'))
                lms_mov_path = os.path.join(GT_PATH, pair['moving'].replace(
                    'images', destination).replace('.nii.gz', '.csv'))
                fix_lms = np.loadtxt(lms_fix_path, delimiter=',')
                mov_lms = np.loadtxt(lms_mov_path, delimiter=',')
                tre = compute_tre(fix_lms, mov_lms, disp_field,
                                  spacing_fix, spacing_mov)
                mean = tre.mean()
                detailed = tre.tolist()
                case_results[_name] = {'mean': mean, 'detailed': detailed}

        cases_results[pairs_string[0]+'-->'+pairs_string[1]] = case_results

        if verbose:
            print(
                f"case_results [{idx}] [{pair['fixed'][-16:-12]} --> {pair['moving'][-16:-12]}]:")
            for k, v in case_results.items():
                print(f"\t{k: <{20}}: {v['mean']:.5f}")

    aggregated_results = {}
    metrics = list(list(cases_results.values())[2].keys())
    for metric in metrics:
        for k, v in cases_results.items():
            # calculate mean of all cases
            all_means_metric = np.array(
                [cases_results[k][metric]['mean'] for k in cases_results.keys()])
            aggregated_results[metric] = {'mean': all_means_metric.mean(),
                                          'std': all_means_metric.std(),
                                          '30': np.quantile(all_means_metric, .3)}

    with open(os.path.join(OUTPUT_PATH), 'w', encoding='utf-8') as f:
        json.dump({'name': name,
                   'aggregates': aggregated_results,
                   'cases': cases_results,
                   'eval_version': '2.0'}, f, indent=4, allow_nan=True)


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
