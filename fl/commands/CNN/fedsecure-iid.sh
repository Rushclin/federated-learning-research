#!/bin/sh

# IID 
py main.py \
    --exp_name "FedAvg_PLANT_VILLAGE_TwoCNN_IID" --seed 42 --device cpu \
    --dataset PLANT_VILLAGE \
    --split_type iid --test_size 0 \
    --model_name TwoCNN --resize 256 --hidden_size 200 \
    --algorithm fedsecure --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 precision f1 recall \
    --K 100 --R 250 --E 1 --C 0.1 --B 5 --beta1 0 \
    --optimizer SGD --lr 0.1 --lr_decay 0.99 --lr_decay_step 25 --criterion CrossEntropyLoss \
    --num_classes 4 --in_channels 3 