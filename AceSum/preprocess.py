#!/usr/bin/env python

import json
from tqdm import tqdm

data_dir = '../data'
    
with open(f'{data_dir}/dataset/train.jsonl', 'r', encoding='utf-8') as f:
    data = json.load(f)

with open(f'{data_dir}/dataset/train.acesum.jsonl', 'w', encoding='utf-8') as f:        
    for j, h in enumerate(tqdm(data)):
        json_string = json.dumps(h, ensure_ascii=False)
        f.write(json_string + '\n')

print(f'preprocessed train.jsonl')

with open(f'{data_dir}/dataset/dev.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

with open(f'{data_dir}/dataset/dev.acesum.json', 'w', encoding='utf-8') as f:
    for j, h in enumerate(tqdm(data)):            
        json_string = json.dumps(h, ensure_ascii=False)
        f.write(json_string + '\n')

print(f'preprocessed dev.json')
