#!/bin/sh

dataset_json=$1
num_iterations=$2

./train_segmentation.py $dataset_json $num_iterations
./train_vxmplus.py $dataset_json $num_iterations