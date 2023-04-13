import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split


cities = ['amsterdam', 'barcelona', 'berlin', 'istanbul', 'lisbon', 'london', 'madrid', 'paris', 'rome', 'tbilisi', 'yerevan']

# train split
with open(f'data/dataset/train.jsonl', 'w', encoding='utf-8') as fi:
    fi.write('[\n')
    for i, city in enumerate(tqdm(cities)):
        with open(f'data/dataset/cities_train/{city}_train.jsonl', 'r', encoding='utf-8') as fo:
            data = json.load(fo)
        for j, h in enumerate(tqdm(data)):            
            json_string = json.dumps(h, ensure_ascii=False, indent=2)
            if (i < len(cities)-1) or (j < len(data)-1):
                json_string += ','
            fi.write(json_string + '\n')

    fi.write(']\n')


with open(f'./data/dataset/train.plansum.jsonl', 'w', encoding='utf-8') as fi:
    fi.write('[\n')
    for i, city in enumerate(tqdm(cities)):
        with open(f'./data/dataset/cities_train/{city}_train.jsonl', 'r', encoding='utf-8') as fo:
            data = json.load(fo)
        for j, h in enumerate(tqdm(data)):
            new_dict = {}

            rs = []
            for r in h['reviews']:
                rs.append([' '.join(r['sentences']), r['rating']])
            
            new_dict['id'] = h['id']
            new_dict['reviews'] = rs
            json_string = json.dumps(new_dict, ensure_ascii=False, indent=2)
            if (i < len(cities)-1) or (j < len(data)-1):
                json_string += ','
            fi.write(json_string + '\n')

    fi.write(']\n')


# dev/test split
with open('./data/dataset/golden.json', 'r', encoding='utf-8') as f:
    golden = json.load(f)

dev, test = train_test_split(golden, test_size=0.5, shuffle=True, random_state=42)

with open('./data/dataset/dev.json', 'w', encoding='utf-8') as f:
    json.dump(dev, f, ensure_ascii=False, indent=2)

with open('./data/dataset/test.json', 'w', encoding='utf-8') as f:
    json.dump(test, f, ensure_ascii=False, indent=2)

with open('./data/dataset/dev.plansum.json', 'w', encoding='utf-8') as f:
    f.write('[\n')
    for j, h in enumerate(tqdm(dev)):
        new_dict = {}

        rs = []
        for r in h['reviews']:
            rs.append([' '.join(r['sentences']), r['rating']])
        
        new_dict['id'] = h['id']
        new_dict['reviews'] = rs
        new_dict['summary'] = h['summary']

        json_string = json.dumps(new_dict, ensure_ascii=False, indent=2)
        if j < len(dev)-1:
            json_string += ','
        f.write(json_string + '\n')

    f.write(']\n')

with open('./data/dataset/test.plansum.json', 'w', encoding='utf-8') as f:
    f.write('[\n')
    for j, h in enumerate(tqdm(test)):
        new_dict = {}

        rs = []
        for r in h['reviews']:
            rs.append([' '.join(r['sentences']), r['rating']])
        
        new_dict['id'] = h['id']
        new_dict['reviews'] = rs
        new_dict['summary'] = h['summary']

        json_string = json.dumps(new_dict, ensure_ascii=False, indent=2)
        if j < len(test)-1:
            json_string += ','
        f.write(json_string + '\n')

    f.write(']\n')