#!/bin/bash

# get params
rank="$1"
echo "$rank"
filepath="$2"
echo "$filepath"
env="$3"
echo "$env"
type="$4"
echo "$type"

# do something
source activate "$env"
python --version
# execute algorithm and write result to json file
nohup python train.py --local_rank="$rank"  --path="$filepath" --type="$type" &
