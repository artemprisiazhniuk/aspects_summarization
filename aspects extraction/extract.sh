#!/bin/bash

dataset="$1"
limit="$2"
model="bert"

mkdir -p ./data/preprocessed/seeds/"$dataset"/yake_"$model"/"$limit"

python ./extract_aspects_yake_bert.py \
    --train ./data/"$dataset"/new_"$limit".jsonl \
    --seeds_dir ./data/preprocessed/seeds/"$dataset"/yake_"$model"/"$limit" \
    --name ./data/preprocessed/"$dataset" \
    --batch_size 200 \
    --aspects 6 \
    --num_seed_words 30 \
    --lemmatize 
