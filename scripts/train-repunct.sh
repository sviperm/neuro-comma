#!/bin/bash

python src/train.py \
    --model-name repunct-model \
    --pretrained-model DeepPavlov/rubert-base-cased-sentence \
    --targets O COMMA PERIOD \
    --train-data data/repunct/train \
    --val-data data/repunct/test \
    --test-data data/repunct/test \
    --store-best-weights \
    --epoch 7 \
    --batch-size 4 \
    --augment-rate 0.15 \
    --labml \
    --seed 1 \
    --cuda 

