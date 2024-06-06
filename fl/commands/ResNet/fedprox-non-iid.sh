#!/bin/sh

## NON-IID
python3 main.py \
    --exp_name "FedProx_PLANT_VILLAGE_ResNet34_NON_IID" --seed 42 --device cpu \
    --dataset PLANT_VILLAGE \
    --split_type non-iid --test_size 0 \
    --model_name ResNet34 --resize 256 --hidden_size 200 \
    --algorithm fedavg --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 precision f1 recall \
    --K 10 --R 20 --E 1 --C 0.1 --B 5--beta1 0 \
    --optimizer SGD --lr 0.01 --lr_decay 0.999 --lr_decay_step 10 --criterion CrossEntropyLoss \
    --num_classes 4 --in_channels 3 \
    --use_tb --tb_host "127.0.0.1" --tb_port 6006 \

