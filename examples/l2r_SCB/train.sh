#!/bin/bash
JSON_PATH=$1
NUM_ITERATIONS=$2
GPU=$3

echo Task $(cat ${JSON_PATH} | python3 -c "import sys,json; print(json.load(sys.stdin)['name'])"), Iter ${NUM_ITERATIONS}, GPU ${GPU}

pairings=$(cat ${JSON_PATH} | python3 -c "import sys,json; print(json.load(sys.stdin)['pairings'])")

if [ $pairings = unpaired ]; then
    echo "Training label-based networks"
    python train_segment.py ${JSON_PATH} ${NUM_ITERATIONS} ${GPU}
    python train_registration.py ${JSON_PATH} ${NUM_ITERATIONS} ${GPU}
elif [ $pairings = paired ]; then
    echo "Training paired network with keypoint loss"
    python train_registration_paired.py ${JSON_PATH} ${NUM_ITERATIONS} ${GPU}
else 
    echo $pairings Error determining pairings
fi