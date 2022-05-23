import json
import os
import argparse



def main(args):
    _version=0.1
    d_opt=vars(args)
    pairs_config=args.pairs_config
    output_dir=args.output_dir
    #evaluation_methods
    evals=[]

    if args.sdlogj:
        evals.append({"name": "LogJacDetStd",
                     "metric": "sdlogj"})
    if args.dice is not None:
        for _instance in args.dice:
            if len(_instance) >2:
                evals.append({"name": _instance[0],
                     "metric": "dice",
                     "labels": _instance[1:]})
            else:
                evals.append({"name": _instance[0],
                     "metric": "dice",
                     "labels": list(range(1,int(_instance[1])+1))})
    if args.hd95 is not None:
        for _instance in args.hd95:
            if len(_instance) >2:
                evals.append({"name": _instance[0],
                     "metric": "hd95",
                     "labels": _instance[1:]})
            else:
                evals.append({"name": _instance[0],
                     "metric": "hd95",
                     "labels": list(range(1,int(_instance[1])+1))})
    if args.tre is not None:
        for _instance in args.tre:
            evals.append({"name": _instance[0],
                       "metric": "tre",
                       "dest": _instance[1]})
        
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
        expected_shape = args.expected_shape
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
    parser.add_argument("--DSC", dest="dice", help="--DSC [name] [labels] /If input is single int, is is assumed to be max label",action='append',  nargs='+')
    parser.add_argument("--HD95", dest="hd95", help="--HD95 [name] [labels] /If input is int, is is assumed to be max label",action='append', nargs='+')
    parser.add_argument("--TRE", dest="tre", help="--TRE [name] [directory] ", action='append',  nargs='+',)

    args= parser.parse_args()
    main(args)