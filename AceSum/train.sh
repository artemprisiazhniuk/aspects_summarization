#!/usr/bin/bash

./preprocess.py

python src/create_mil_data.py ../data/dataset # create train.mil.jsonl and dev.mil.json

python src/train_mil.py -mode=train -model_name=mil # train mil.model

python src/create_sum_data.py -mode=train -load_model=models/mil.model # create train.sum.jsonl
python src/create_sum_data.py -mode=eval-general -load_model=models/mil.model # create dev.sum.jsonl, test.sum.jsonl
python src/create_sum_data.py -mode=eval-general -load_model=models/mil.model -dataset=space_ru # create dev.sum.jsonl, test.sum.jsonl

python src/train_sum.py -mode=train-general -model_name=sum # train sum.model

python src/train_sum.py -model=validation-general -load_model=models/sum.model # evaluate sum.model
python src/train_sum.py -model=validation-general -load_model=models/sum.model -gen_test_file=../data/space_ru/test.sum.general.jsonl # evaluate sum.model

python src/train_sum.py -mode=train-general -model_name=sum -load_model=models/sum.model -train_file=../data/space_ru/dev.sum.general.jsonl -gen_dev_file=../data/space_ru/test.sum.general.jsonl -no_train_steps=10000 -no_warmup_steps=0 -check_every=2000 -ckpt_every=2000 # finetune on space_ru dev split

python src/train_sum.py -model=validation-general -load_model=models/sum.model # evaluate sum.model
python src/train_sum.py -model=validation-general -load_model=models/sum.model -gen_test_file=../data/space_ru/test.sum.general.jsonl # evaluate sum.model