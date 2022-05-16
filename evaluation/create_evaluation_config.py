import json
import os
import argparse



def main(args):
    _version=0.1
    d_opt=vars(args)
    pairs_config=args.pairs_config
    output_dir=args.output_dir
    #evaluation_methods
    evals={}
    for eval in ['sdlogj','tre']:
        if d_opt[eval] == True:
            evals[eval]=True
    for eval in ['dice', 'dice_secret', 'hd95', 'hd95_secret']:
        if d_opt[eval] != None:
            if len(d_opt[eval])==1:
                evals[eval]=list(range(1,d_opt[eval][0]+1))
            else:
                evals[eval]=d_opt[eval]

    ##eval pairs
    with open(pairs_config, 'r') as f:
        json_data=json.load(f)
    if args.test:
        eval_pairs = json_data['registration_test']
        teststring='_TEST_'
    else:
        eval_pairs = json_data['registration_val']
        teststring='_VAL_'

    task_name=json_data['name']
    if args.expected_shape is not None:
        print('ff')
        expected_shape = d_opt['expected_shape']
    else:
        expected_shape=json_data['tensorImageShape']["0"]
        expected_shape.append(3)

    #write config
    evaluation_config={ 'task_name':task_name,
                        'evaluation_methods':evals,
                        'expected_shape':expected_shape,
                        'eval_pairs':eval_pairs,
                        'eval_config_version':_version
                        }

    with open(os.path.join(output_dir,task_name+teststring+'evaluation_config.json'), 'w') as f:
        json.dump(evaluation_config,f, indent=4)

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Create evaluation.json from {TASKNAME}_dataset.json")
    parser.add_argument("-c","--config", dest="pairs_config", help="path to {TASKNAME}_dataset.json", required=True)
    parser.add_argument("-o","--output", dest="output_dir", help="path to write evaluation.json", required=False, default='.')
    parser.add_argument("-s","--shape", dest='expected_shape', help="expected shape of deformation field", required=False, nargs='+', type=int)
    parser.add_argument("--test", dest="test", action=argparse.BooleanOptionalAction, default=False) #Otherwise Validation

    parser.add_argument("--SDlogJ", dest="sdlogj", help="Evaluate SDlogJ", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--DSC", dest="dice", help="Specify the labels. If input is single int, is is assumed to be max label", nargs='+', type=int)
    parser.add_argument("--DSC_secret", dest="dice_secret", help="Specify the labels. If input is int, is is assumed to be max label", nargs='+', type=int)
    parser.add_argument("--HD95", dest="hd95", help="Specify the labels. If input is int, is is assumed to be max label", nargs='+', type=int)
    parser.add_argument("--HD95_secret", dest="hd95_secret", help="Specify the labels. If input is int, is is assumed to be max label", nargs='+', type=int)
    parser.add_argument("--TRE", dest="tre", action=argparse.BooleanOptionalAction, default=False)

    args= parser.parse_args()
    main(args)