#!/bin/bash
JSON_PATH=$1
GPU=$2


task=$(cat ${JSON_PATH} | python3 -c "import sys,json; print(json.load(sys.stdin)['name'])")
pairings=$(cat ${JSON_PATH} | python3 -c "import sys,json; print(json.load(sys.stdin)['pairings'])")


if [ $pairings = unpaired ]; then
    echo "Inferring ${task} (label-based training)"
    python inference_test.py ${JSON_PATH} ${GPU}
elif [ $pairings = paired ]; then
    echo "Inferring ${task} (keypoint-based training)"
    python inference_paired_test.py ${JSON_PATH} ${GPU}
else 
    echo "$pairings Error determining pairings"
fi