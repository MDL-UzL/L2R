# L2R/evaluation
Repository for L2R evaluation files

Structure:
* L2RTest: docker image for grand-challenge.org
* evaluation_configs: config files for evaluations
* zero_def_fields: zero_def_fields for all current datasets


HowTo:
1) Create evaluation_config.json with create_evaluation_config.py (or use from evaluation_configs)
```
python create_evaluation_config.py -c pairs_val.csv 
  -s 192 160 256 3 
  -n AbdomenCTCT_Val
  -o evaluation_configs/ 
  --SDlogJ 
  --DSC 13 
  --HD95 13
```
3) Run evaluation.py
```
 python evaluation.py -i zero_def_fields/HippocampusMR/ 
  -d .../HippocampusMR/ 
  -o output_test/Hippo.json 
  -c evaluation_configs/HippocampusMR_Val_evaluation_config.json 
  -v
```
