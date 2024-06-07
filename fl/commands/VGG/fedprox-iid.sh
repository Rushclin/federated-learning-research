#!/bin/sh

## IID 
python3 main.py \
    --exp_name "FedProx_PLANT_VILLAGE_VGG13BN_IID" --seed 42 --device cpu \
    --dataset PLANT_VILLAGE \
    --split_type iid --test_size 0 \
    --model_name VGG13BN --resize 256 --hidden_size 200 \
    --algorithm fedprox --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 precision f1 recall \
    --K 10 --R 20 --E 1 --C 0.1 --B 5 --beta1 0 \
    --optimizer SGD --lr 0.1 --lr_decay 0.999 --lr_decay_step 10 --criterion CrossEntropyLoss \
    --num_classes 15 --in_channels 3