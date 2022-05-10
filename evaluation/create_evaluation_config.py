import json
import pandas
import os
import argparse



def main(args):
    _version=0.1
    d_opt=vars(args)
    csv_input=d_opt['csv_input']
    output_dir=d_opt['output_dir']
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
    eval_pairs=[]
    pairs = pandas.read_csv(csv_input)
    for i,val in pairs.iterrows():
        eval_pairs.append({
            'fixed':val['fixed'],
            'moving':val['moving']
            })

    #write config
    evaluation_config={ 'task_name':d_opt['task_name'],
                        'evaluation_methods':evals,
                        'expected_shape':d_opt['expected_shape'],
                        'eval_pairs':eval_pairs,
                        'eval_config_version':_version
                        }

    with open(os.path.join(output_dir,d_opt['task_name']+'_evaluation_config.json'), 'w') as f:
        json.dump(evaluation_config,f, indent=4)

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Create evaluation.json from pairs_{val/test}.csv ")
    parser.add_argument("-c","--csv", dest="csv_input", help="path to pairs_{test/val}.csv", required=True)
    parser.add_argument("-s","--shape", dest='expected_shape', help="expected shape of deformation field", required=True, nargs='+', type=int)
    parser.add_argument("-n","--name", dest="task_name", help='Task/Evaluation name (e.g. OASIS)', required=True)
    parser.add_argument("-o","--output", dest="output_dir", help="path to write evaluation.json", required=False, default='.')

    parser.add_argument("--SDlogJ", dest="sdlogj", help="Evaluate SDlogJ", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--DSC", dest="dice", help="Specify the labels. If input is single int, is is assumed to be max label", nargs='+', type=int)
    parser.add_argument("--DSC_secret", dest="dice_secret", help="Specify the labels. If input is int, is is assumed to be max label", nargs='+', type=int)
    parser.add_argument("--HD95", dest="hd95", help="Specify the labels. If input is int, is is assumed to be max label", nargs='+', type=int)
    parser.add_argument("--HD95_secret", dest="hd95_secret", help="Specify the labels. If input is int, is is assumed to be max label", nargs='+', type=int)
    parser.add_argument("--TRE", dest="tre", action=argparse.BooleanOptionalAction, default=False)

    args= parser.parse_args()
    main(args)