#!/bin/sh

py pred.py \
    --model_name TwoCNN --resize 256 --hidden_size 200 \
    --num_classes 4 --in_channels 3 \

