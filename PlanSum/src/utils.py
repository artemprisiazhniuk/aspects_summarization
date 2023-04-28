import torch
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
import json
import re
from datasets import load_metric


def condense_data(file, adjust_sentiment=0):
  """
    Preprocess dataset for Condense model.
  """
  f = open(file, 'r', encoding='utf-8', errors='ignore')
  data = json.load(f)
  f.close()

  x_data = []
  y_data = [] # sentiment labels

  for instance in tqdm(data):
    for review in instance['reviews']:
      review, sentiment = review
      if sentiment == -1:
        continue
      sentiment -= adjust_sentiment

      review = review.strip()

      x_data.append(review)
      y_data.append(sentiment)

  return x_data, y_data


def check_summary_worthy(x, tokenizer,
                         min_length=50,
                         max_length=90,
                         max_symbols=0,
                         max_tridots=0):
  """
    Check whether the review x is summary-worthy or not.
  """
  x = tokenizer.decode(x)
  tokens = x.split()[1:-1]

  length = 0
  num_symbols = 0
  num_tridots = 0

  for token in tokens:
    if token in ['[CLS]', '[SEP]']:
      continue
    length += 1
    if token == '...':
      num_tridots += 1
    symbol = re.sub(r"[A-Za-zА-Яа-яЁё0-9]", '', token)
    symbol = re.sub(r"[,!.-]", '', symbol)
    if len(symbol) > 0: # and symbol not in ",!.'":
      num_symbols += 1

  return length >= min_length and length <= max_length and num_symbols <= max_symbols and num_tridots <= max_tridots


def abstract_data(file, multi_ref=False):
  """
    Preprocess dataset for Abstract model.
  """
  f = open(file, 'r', encoding='utf-8', errors='ignore')
  data = json.load(f)
  f.close()

  x_data = []
  y_data = []

  for instance in tqdm(data):
    if not multi_ref:
      if 'summary' in instance:
        summary = instance['summary'].strip()
        y_data.append(summary)
    else:
      if 'summary' in instance:
        summary = instance['summary'][0].strip()
        y_data.append(summary)

    reviews = []
    for review in instance['reviews']:
      if not (type(review) == list):
        review = review.replace('MOV', '<movie>').strip()
      else:
        review = review[0].replace('MOV', '<movie>').strip()
      reviews.append(review)

    x_data.append(reviews)

  return x_data, y_data


def pad_text(batch, pad_id=0):
  max_length = max(len(inst) for inst in batch)

  inst_batch = []
  mask_batch = []
  for inst in batch:
    if isinstance(inst, torch.Tensor):
      inst = inst.tolist()
    inst = list(inst)
    mask = [1.0] * len(inst) + [0.0] * (max_length - len(inst))
    inst = inst + [pad_id] * (max_length - len(inst))
    mask_batch.append(mask)
    inst_batch.append(inst)

  inst_batch = torch.Tensor(inst_batch).long().cuda()
  mask_batch = torch.Tensor(mask_batch).float().cuda()

  return inst_batch, mask_batch


def pad_vector(batch, vec_len):
  max_length = max(len(inst) for inst in batch)

  inst_batch = []
  mask_batch = []
  for inst in batch:
    if isinstance(inst, torch.Tensor):
      inst = inst.tolist()
    inst = list(inst)
    mask = [1.0] * len(inst) + [0.0] * (max_length - len(inst))
    inst = inst + [[0.0] * vec_len] * (max_length - len(inst))
    mask_batch.append(mask)
    inst_batch.append(inst)

  inst_batch = np.array(inst_batch)
  mask_batch = np.array(mask_batch)

  inst_batch = torch.Tensor(inst_batch).float().cuda()
  mask_batch = torch.Tensor(mask_batch).float().cuda()

  return inst_batch, mask_batch


def concat_pad(batch):
  new_batch = []

  for inst in batch:
    new_inst = []
    for doc in inst:
      new_inst += doc
    new_batch.append(new_inst)
  
  return pad_text(new_batch)


def get_distance(a, b):
  def hellinger(p, q):
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)

  print(a)
  print(b)
  asp_dist = hellinger(a[0], b[0])
  sen_dist = hellinger(a[1], b[1])

  return asp_dist + sen_dist


def run_condense(x_data, tokenizer, encoder, model):
  """
    Runs the Condense model then aggregates outputs for each batch.
  """
  vocab_size = len(tokenizer)

  tokens_data = []
  token_ids_data = []
  aspect_data = []
  sentiment_data = []

  for x_batch in x_data:
    token_ids, mask = pad_text(x_batch)

    tokens = encoder(token_ids)

    tokens, _, prob_a, prob_s = model.condense(tokens, mask)

    hidden_dim = tokens.size()[-1]

    tokens = tokens.contiguous().view(-1, hidden_dim)
    token_ids = token_ids.view(-1)
    mask = mask.view(-1)

    assert tokens.size()[0] == token_ids.size()[0]
    assert tokens.size()[0] == mask.size()[0]

    tokens = tokens[mask.nonzero().squeeze()] # token len = batch_size*length-mask, hidden_dim 
    token_ids = token_ids[mask.nonzero().squeeze()]

    # token mean fusion
    token_len = token_ids.size()[0]
    sum_tokens = torch.zeros(vocab_size, hidden_dim).cuda()
    cnt_tokens = torch.zeros(vocab_size).cuda()
    tindex = torch.arange(0, token_len)
    sum_tokens[token_ids] += tokens[tindex]
    cnt_tokens[token_ids] += 1
    ave_tokens = sum_tokens / cnt_tokens.unsqueeze(-1)

    sum_token_ids = cnt_tokens.nonzero().squeeze()
    sum_tokens = sum_tokens[sum_token_ids]

    ave_token_ids = cnt_tokens.nonzero().squeeze()
    ave_tokens = ave_tokens[ave_token_ids]

    # prob distribution injective fusion
    aspect = model.get_aspect(prob_a.mean(dim=0, keepdim=True)).squeeze().cpu().detach().numpy()
    sentiment = model.get_sentiment(prob_s.mean(dim=0, keepdim=True)).squeeze().cpu().detach().numpy()

    tokens_data.append(sum_tokens.cpu().detach().numpy())
    token_ids_data.append(sum_token_ids.cpu().detach().numpy())
    aspect_data.append(aspect)
    sentiment_data.append(sentiment)

  return tokens_data, token_ids_data, aspect_data, sentiment_data


def bert_label_smoothing(y_batch, tokenizer, language_model, rate=0.1):
  """
    Apply BERT-based label smoothing.
  """
  y_batch_ = [[yy if yy != len(tokenizer)-1 else 0 for yy in y] for y in y_batch]
  y_batch_, mask = pad_text(y_batch_)

  batch_size, seq_len = y_batch_.size()

  p_batch = language_model(y_batch_, mask)[0].softmax(-1) # b, s, v
  # pad = torch.zeros(batch_size, seq_len, 1).cuda()
  # p_batch = torch.cat([p_batch, pad], -1)

  y_batch, mask = pad_text(y_batch)
  y_batch = F.one_hot(y_batch, num_classes=len(tokenizer))

  output_batch = rate * p_batch + (1-rate) * y_batch

  return output_batch, mask


def get_metrics(golds, preds, use_stemmer=True):
  metric = load_metric("src/rouge")

  result = metric.compute(predictions=preds, references=golds, language="ru", use_stemmer=use_stemmer)
  result = {key: round(value.mid.fmeasure, 4) * 100 for key, value in result.items()}

  return result['rouge1'], result['rouge2'], result['rougeL']