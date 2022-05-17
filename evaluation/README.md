# L2R/evaluation
Repository for L2R evaluation files

Structure:
* L2RTest: docker image for grand-challenge.org
* evaluation_configs: config files for evaluations
* zero_def_fields: zero deformation fields for all current datasets


HowTo:
1) Create evaluation_config.json with create_evaluation_config.py (or use from evaluation_configs)
```
python create_evaluation_config.py -o evaluation_configs/ -c ../tmp/AbdomenCTCT_dataset.json --SDlogJ  --DSC DSC 13 --HD95 HD95 13  
python create_evaluation_config.py -o evaluation_configs/ -c ../tmp/AbdomenMRCT_dataset.json --SDlogJ --DSC DSC 4 --HD95 HD95 4
python create_evaluation_config.py -o evaluation_configs/ -c ../tmp/LungCT_dataset.json --SDlogJ --TRE TRE_lm landmarks --TRE TRE_kp keypoints
python create_evaluation_config.py -o evaluation_configs/ -c ../tmp/CuRIOUS_dataset.json --SDlogJ --TRE TRE_lm landmarks
python create_evaluation_config.py -o evaluation_configs/ -c ../tmp/HippocampusMR_dataset.json --SDlogJ --DSC DSC 2 --HD95 HD95 2
python create_evaluation_config.py -o evaluation_configs/ -c ../tmp/OASIS_dataset.json --SDlogJ --DSC DSC 35 --HD95 HD95 35
```

3) Run evaluation.py
```
 python evaluation.py -i zero_def_fields/HippocampusMR/ 
  -d .../HippocampusMR/ 
  -o output_test/Hippo.json 
  -c evaluation_configs/HippocampusMR_VAL_evaluation_config.json 
  -v
```

