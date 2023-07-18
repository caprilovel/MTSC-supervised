#! /bin/bash

for i in {0..24}
do
    python main.py --model_index 1 --dataset_index $i --patch_size 1
done