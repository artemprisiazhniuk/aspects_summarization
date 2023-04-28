#!/usr/bin/env python

import json
from tqdm import tqdm

data_dir = '../data'

for dataset in ['dataset', 'space_ru']:
    if dataset == 'dataset':
        id_alias = 'id'
    else:
        id_alias = 'entity_id'
    
    if dataset == 'dataset':
        with open(f'{data_dir}/{dataset}/train.jsonl', 'r', encoding='utf-8') as f:
            data = json.load(f)

        with open(f'{data_dir}/{dataset}/train.plansum.jsonl', 'w', encoding='utf-8') as f:
            f.write('[\n')
            
            for j, h in enumerate(tqdm(data)):
                new_dict = {}

                rs = []
                for r in h['reviews']:
                    rs.append([' '.join(r['sentences']), r['rating']])
                
                new_dict['id'] = h[id_alias]
                new_dict['reviews'] = rs
                json_string = json.dumps(new_dict, ensure_ascii=False, indent=2)
                if j < len(data)-1:
                    json_string += ','
                f.write(json_string + '\n')

            f.write(']\n')
        print(f'preprocessed {dataset}/train.jsonl')

    for split in ['dev', 'test']:
        with open(f'{data_dir}/{dataset}/{split}.json', 'r', encoding='utf-8') as f:
            if dataset == 'dataset':
                data = json.load(f)
            else:
                data = []
                for jline in f:
                    data.append(json.loads(jline))

        with open(f'{data_dir}/{dataset}/{split}.plansum.json', 'w', encoding='utf-8') as f:
            f.write('[\n')
            
            for j, h in enumerate(tqdm(data)):
                new_dict = {}

                rs = []
                for r in h['reviews']:
                    rs.append([' '.join(r['sentences']), r['rating']])
                
                new_dict['id'] = h[id_alias]
                new_dict['reviews'] = rs
                if dataset == 'dataset':
                    new_dict['summary'] = h['summary']
                else:
                    new_dict['summary'] = h['summaries']['general']

                json_string = json.dumps(new_dict, ensure_ascii=False, indent=2)
                if j < len(data)-1:
                    json_string += ','
                f.write(json_string + '\n')

            f.write(']\n')
        print(f'preprocessed {dataset}/{split}.json')
