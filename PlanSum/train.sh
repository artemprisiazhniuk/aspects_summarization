#!/usr/bin/bash

./preprocess.py # reformat data for PlanSum

python src/train_condense.py -mode=train # train condense.model

python src/train_condense.py -mode=create # create train.plan.jsonl

python src/train_abstract.py -mode=train # train abstract.model

python src/train_abstract.py -mode=eval # eval on 'dataset'
python src/train_abstract.py -mode=eval -test_file="../data/space_ru/test.plansum.json" -multi_ref=1 # eval on 'space_ru'

python src/train_abstract.py -mode=train -train_file="../data/space_ru/dev.plansum.json" -multi_ref_train=1 -dev_file="../data/space_ru/test.plansum.json" -multi_ref=1 -warmup=0 -num_epochs=10 # finetune abstract.model

python src/train_abstract.py -mode=eval # eval on 'dataset'
python src/train_abstract.py -mode=eval -test_file="../data/space_ru/test.plansum.json" -multi_ref=1 # eval on 'space_ru'