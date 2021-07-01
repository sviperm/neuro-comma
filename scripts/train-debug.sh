#!/bin/bash

TOKENIZERS_PARALLELISM=false \
python src/train.py \
    --model-name debug-model \
    --train-data data/debug-data/train \
    --val-data data/debug-data/valid \
    --pretrained-model DeepPavlov/rubert-base-cased-sentence \
    --store-best-weights \
    --epoch 3 \
    --targets O M
