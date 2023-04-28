import json
import numpy as np
import os
from tqdm import tqdm
import random

def create_data(filedir):
  print(filedir)

  # get aspects and keywords
  seeds_filedir = '../data/seeds/' + filedir.split('/')[-1]
  files = os.listdir(seeds_filedir)
  keywords_dict = {}
  for file in files:
    print(seeds_filedir + '/' + file)
    f = open(seeds_filedir + '/' + file, 'r', encoding='utf-8')
    keywords = []
    for _ in range(5):
      keyword = f.readline().strip().split()[-1]
      keywords.append(keyword)

    f.close()
    aspect = file.replace('.txt', '')
    keywords_dict[aspect] = keywords

  aspects = list(keywords_dict.keys()) # + ['general']

  instance_dict = {}
  f = open(filedir + '/train.acesum.jsonl', 'r', encoding='utf-8')
  for line in tqdm(f):
    inst = json.loads(line.strip())

    domain = filedir.split('/')[-1]
    if domain not in instance_dict:
      instance_dict[domain] = {}

    reviews = inst['reviews']
    for review in reviews:
      sentences = review['sentences']

      # sanity check
      if len(sentences) > 35 or len(sentences) < 1:
        continue
      if max([len(sentence.split()) for sentence in sentences]) > 35:
        continue

      review = ' '.join(sentences).split()
      
      # check whether aspect keywords in review
      class_list = []
      for aspect in aspects:
        keywords = keywords_dict[aspect]
        includes = int(any([keyword in review for keyword in keywords]))
        class_list.append(includes)

      # add review to corresponding aspect buckets
      instance_tuple = (sentences, class_list)
      class_list = tuple(class_list)
      if class_list not in instance_dict[domain]:
        instance_dict[domain][class_list] = []
      instance_dict[domain][class_list].append(instance_tuple)

  f.close()

  for domain in instance_dict:
    lengths = [len(instance_dict[domain][key]) for key in instance_dict[domain]]
    print(lengths)
    min_length = sorted(lengths)[1]

    for i in range(len(aspects)):
      c = [0] * len(aspects)
      c[i] = 1
      print(c)
      print(len(instance_dict[domain][tuple(c)]))

    print('mininum instances per tuple', min_length)

    data = []
    for key in instance_dict[domain]:
      instances = instance_dict[domain][key]
      random.shuffle(instances)
      data += instances[:min_length]

    print('total data', len(data))
    random.shuffle(data)
    max_text_length = 0

    domain_aspects = aspects

    f = open(filedir + '/train.mil.jsonl', 'w', encoding='utf-8')
    count_dict = {aspect:0 for aspect in domain_aspects}
    for inst in data:
      new_inst = {}
      new_inst['review'] = inst[0]
      max_text_length = max(max_text_length, len(inst[0]))
      class_dict = {}

      for i, aspect in enumerate(domain_aspects):
        class_dict[aspect] = 'yes' if inst[1][i] else 'no'
        if inst[1][i]:
          count_dict[aspect] += 1
      new_inst['aspects'] = class_dict
      f.write(json.dumps(new_inst, ensure_ascii=False) + '\n')

    f.close()

    print('max text length', max_text_length)
    print(count_dict)


def create_data_dev(filedir):
  print(filedir)

  # get aspects and keywords
  seeds_filedir = '../data/seeds/' + filedir.split('/')[-1]
  files = os.listdir(seeds_filedir)
  keywords_dict = {}
  for file in files:
    print()
    f = open(seeds_filedir + '/' + file, 'r', encoding='utf-8')
    keywords = []
    for _ in range(5):
      keyword = f.readline().strip().split()[-1]
      keywords.append(keyword)

    f.close()
    aspect = file.replace('.txt', '')
    keywords_dict[aspect] = keywords

  aspects = list(keywords_dict.keys()) # + ['general']

  instance_dict = {}
  f = open(filedir + '/dev.acesum.json', 'r', encoding='utf-8')
  for line in tqdm(f):
    inst = json.loads(line.strip())

    domain = filedir.split('/')[-1]
    if domain not in instance_dict:
      instance_dict[domain] = {}

    reviews = inst['reviews']
    for review in reviews:
      sentences = review['sentences']

      # sanity check
      if len(sentences) > 35 or len(sentences) < 1:
        continue
      if max([len(sentence.split()) for sentence in sentences]) > 35:
        continue

      review = ' '.join(sentences).split()
      
      # check whether aspect keywords in review
      class_list = []
      for aspect in aspects:
        keywords = keywords_dict[aspect]
        includes = int(any([keyword in review for keyword in keywords]))
        class_list.append(includes)

      # add review to corresponding aspect buckets
      instance_tuple = (sentences, class_list)
      class_list = tuple(class_list)
      if class_list not in instance_dict[domain]:
        instance_dict[domain][class_list] = []
      instance_dict[domain][class_list].append(instance_tuple)

  f.close()

  for domain in instance_dict:
    print('domain', domain)

    lengths = [len(instance_dict[domain][key]) for key in instance_dict[domain]]
    print(lengths)
    min_length = sorted(lengths)[1]

    for i in range(len(aspects)):
      c = [0] * len(aspects)
      c[i] = 1
      print(c)
      print(len(instance_dict[domain][tuple(c)]))

    print('mininum instances per tuple', min_length)

    data = []
    for key in instance_dict[domain]:
      instances = instance_dict[domain][key]
      random.shuffle(instances)
      data += instances[:min_length]

    print('total data', len(data))
    random.shuffle(data)
    max_text_length = 0

    domain_aspects = aspects

    f = open(filedir + '/dev.mil.jsonl', 'w', encoding='utf-8')
    count_dict = {aspect:0 for aspect in domain_aspects}
    for inst in data:
      new_inst = {}
      new_inst['review'] = inst[0]
      max_text_length = max(max_text_length, len(inst[0]))
      class_dict = {}

      for i, aspect in enumerate(domain_aspects):
        class_dict[aspect] = 'yes' if inst[1][i] else 'no'
        if inst[1][i]:
          count_dict[aspect] += 1
      new_inst['aspects'] = class_dict
      f.write(json.dumps(new_inst, ensure_ascii=False) + '\n')

    f.close()

    print('max text length', max_text_length)
    print(count_dict)


import sys
filedir = sys.argv[1]
create_data(filedir)
create_data_dev(filedir)