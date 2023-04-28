#!/usr/bin/bash

# train sentencepiece tokenizer
# cd ../data/sentencepiece
# python SemAE/src/utils/train-spm.py ../dataset/train.jsonl spm_ru_unigram_32k.model
# cd ../../SemAE

python src/train.py --run_id dataset_run --gpu 0

python src/inference.py --model models/dataset_run_10_model.pt --sample_sentences --run_id dataset_general_run --gpu 0
python src/inference.py --model models/dataset_run_10_model.pt --sample_sentences --run_id dataset_general_run --gpu 0 --entity_id entity_id --summary_data ../data/space_ru/test.json