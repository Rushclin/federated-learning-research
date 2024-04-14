#!/bin/sh

py pred.py \
    --model_name ResNet34 --resize 256 --hidden_size 200 \
    --num_classes 4 --in_channels 3 \

