#!/bin/bash

python ./extract_aspects.py \
    --name ./data/preprocessed/test_dump \
    --batch_size 200 \
    --aspects 6 \
    --lemmatize 
    