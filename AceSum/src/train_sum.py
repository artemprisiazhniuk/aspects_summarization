import argparse

import json
import numpy as np

import torch
from torch.utils.data import DataLoader

from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM as Model
from transformers import AdamW
from transformers import get_cosine_schedule_with_warmup

import os
from tqdm import tqdm

from calculate_rouge import calculate
from data_pipeline import SummarizationDataset

import wandb


def train_general(args):
  print(args)

  if args.use_wandb:
      wandb.login()
      wandb.init(project='AceSum', name=args.mode, config=vars(args))

  print('Preparing data...')
  
  if args.train_file is None:
    train_file = args.data_dir + '/' + args.dataset + '/train.sum.jsonl'
  else:
    train_file = args.train_file
  dataset = SummarizationDataset(
    train_file, 
    use_keywords=args.use_keywords, use_switch=args.use_switch)
  dataloader = DataLoader(dataset, batch_size=args.batch_size)

  if args.gen_dev_file is None:
    gen_dev_file = args.data_dir + '/' + args.dataset + '/dev.sum.general.jsonl'
  else:
    gen_dev_file = args.gen_dev_file
  gen_dev_dataset = SummarizationDataset(
    gen_dev_file,
    use_keywords=args.use_keywords, use_switch=args.use_switch, shuffle=False)
  gen_dev_dataloader = DataLoader(gen_dev_dataset, batch_size=args.batch_size)
  f = open(gen_dev_file, 'r', encoding='utf-8')
  lines = f.readlines()
  data = [json.loads(line) for line in lines]
  f.close()
  gen_gold_sums = [[summary.lower() for summary in inst['summary']] for inst in data]

  print('Initializing model...')

  tokenizer = AutoTokenizer.from_pretrained(args.model_type)
  special_tokens = ['<rev>', '<key>', '<sum>', '<switch>']
  if args.use_switch != 'none':
    for i in range(args.num_aspects):
      special_tokens.append('<pos_%d>' % i)

  tokenizer.add_special_tokens(
    {'additional_special_tokens': special_tokens}
  )

  model = Model.from_pretrained(args.model_type, return_dict=True)
  model.resize_token_embeddings(len(tokenizer))
  model.cuda()

  optimizer = AdamW(model.parameters(), lr=args.learning_rate)
  scheduler = get_cosine_schedule_with_warmup(
    optimizer, args.no_warmup_steps, args.no_train_steps)

  step = 0
  best_rouge = 0
  if args.load_model is not None:
    print('Loading model...')
    best_point = torch.load(args.load_model)
    model.load_state_dict(best_point['model'])

  print('Start training...')
  while step < args.no_train_steps:
    losses = []
    for _, (inp_batch, out_batch, switch_batch) in enumerate(tqdm(dataloader)):
      model.train()

      inp_batch = list(inp_batch)
      out_batch = list(out_batch)

      batch_encoding = tokenizer.prepare_seq2seq_batch(
        src_texts=inp_batch,
        tgt_texts=out_batch,
        max_length=args.max_length,
        max_target_length=args.max_target_length,
        padding=True,
        truncation=True,
        return_tensors='pt'
      )

      inp_ids = batch_encoding['input_ids'].cuda()
      inp_mask = batch_encoding['attention_mask'].cuda()
      out_ids = batch_encoding['labels'].cuda()
      out_mask = torch.where(out_ids==0, 0, 1).unsqueeze(-1) # batch_size, out_len
      out_ids[out_ids==0] = -100
      dec_inp_ids = model._shift_right(out_ids)


      model_outputs = model(input_ids=inp_ids,
                            attention_mask=inp_mask,
                            decoder_input_ids=dec_inp_ids,
                            labels=out_ids,
                            output_hidden_states=True)

      loss = model_outputs.loss
      losses.append(loss.item())

      if args.use_wandb:
        wandb.log({
          'train_loss': loss.item()
        })

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      scheduler.step()

      step += 1
      if step % args.check_every == 0:
        print('Step %d Loss %.4f' % (step, np.mean(losses)))
        model.eval()

        rouge_scores = []

        # generate general summaries
        gen_pred_sums = []
        for _, (inp_batch, out_batch, _) in enumerate(tqdm(gen_dev_dataloader)):
          # bug-fix
          inp_batch = list(inp_batch)
          out_batch = list(out_batch)

          batch_encoding = tokenizer.prepare_seq2seq_batch(
            src_texts=inp_batch,
            tgt_texts=out_batch,
            max_length=args.max_length,
            max_target_length=args.max_target_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
          )

          inp_ids = batch_encoding['input_ids'].cuda()

          preds = model.generate(
            inp_ids,
            min_length=60,
            max_length=args.max_target_length*2,
            num_beams=2,
            no_repeat_ngram_size=2,
            decoder_start_token_id=0,
            repetition_penalty=1,
            length_penalty=1,
          )

          for pred in preds:
            gen_pred_sums.append(tokenizer.decode(pred, skip_special_tokens=True))

        gen_scores = 0
        gen_scores = calculate(gen_gold_sums, gen_pred_sums)
        rouge_scores += list(gen_scores)

        if args.use_wandb:
          wandb.log({
            'ROUGE-1': gen_scores[0],
            'ROUGE-2': gen_scores[1],
            'ROUGE-L': gen_scores[2]
          })

        rouge = np.power(np.product(rouge_scores), 1.0/len(rouge_scores))

        print("ROUGE: %.4f" % rouge)
        print("General Gold:", gen_gold_sums[0])
        print("General Pred:", gen_pred_sums[0])

        if rouge > best_rouge:
          print('Saving...')
          checkpoint = args.model_dir + '/' + args.dataset + '/' + args.model_name + '.best.%d.%.2f' % (step, rouge)
          
          best_rouge = rouge
          torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'step': step,
            'loss': np.mean(losses)
          }, checkpoint)
          if args.use_wandb:
            wandb.save(checkpoint)

      if (step / args.ckpt_every >= 1)  and (step % args.ckpt_every == 0):
        print('Saving...')
        checkpoint = args.model_dir + '/' + args.dataset + '/' + args.model_name + '.%d.%.2f' % (step, np.mean(losses))

        torch.save({
          'model': model.state_dict(),
          'optimizer': optimizer.state_dict(),
          'scheduler': scheduler.state_dict(),
          'step': step,
          'loss': np.mean(losses)
        }, checkpoint)
        losses = []

      if step == args.no_train_steps:
        break

  checkpoint = args.model_dir + '/' + args.dataset + '/' + args.model_name + '.model'
  torch.save({
          'model': model.state_dict(),
          'optimizer': optimizer.state_dict(),
          'scheduler': scheduler.state_dict()
        }, checkpoint)

  if args.use_wandb:
    wandb.finish()


@torch.inference_mode()
def evaluate(args, test_type='general'):
  print(args)

  print('Preparing data...')

  # bug-fix
  if args.gen_test_file is None and test_type == 'general':
    test_file = args.data_dir + '/' + args.dataset + '/test.sum.' + test_type + '.jsonl'
  elif test_type == 'general':
    test_file = args.gen_test_file
  dataset = SummarizationDataset(
    test_file,
    use_keywords=args.use_keywords, use_switch=args.use_switch, shuffle=False)
  dataloader = DataLoader(dataset, batch_size=args.batch_size)

  print('Initializing model...')
  tokenizer = AutoTokenizer.from_pretrained(args.model_type)
  special_tokens= ['<rev>', '<key>', '<sum>', '<switch>']
  if args.use_switch != 'none':
    for i in range(args.num_aspects):
      special_tokens.append('<pos_%d>' % i)

  tokenizer.add_special_tokens(
    {'additional_special_tokens': special_tokens}
  )

  model = Model.from_pretrained(args.model_type, return_dict=True)
  model.resize_token_embeddings(len(tokenizer))
  model.cuda()

  assert args.load_model is not None
  best_point = torch.load(args.load_model)
  model.load_state_dict(best_point['model'])

  os.makedirs('output/' + args.dataset, exist_ok=True)
  f = open('output/' + args.dataset + '/acesum.' + test_type + '.out', 'w', encoding='utf-8')
  for _, (inp_batch, out_batch, _) in enumerate(tqdm(dataloader)):
    model.eval()

    inp_batch = list(inp_batch)
    out_batch = list(out_batch)

    batch_encoding = tokenizer.prepare_seq2seq_batch(
      src_texts=inp_batch,
      tgt_texts=out_batch,
      max_length=args.max_length,
      max_target_length=args.max_target_length,
      padding=True,
      truncation=True,
      return_tensors='pt'
    )

    inp_ids = batch_encoding['input_ids'].cuda()
    inp_mask = batch_encoding['attention_mask'].cuda()

    preds = model.generate(
      inp_ids,
      decoder_start_token_id=0,
      min_length=args.min_target_length,
      max_length=args.max_target_length*2,
      num_beams=args.num_beams,
      no_repeat_ngram_size=args.no_repeat_ngram_size,
      repetition_penalty=args.repetition_penalty,
      length_penalty=args.length_penalty,
    )

    for pred in preds:
      f.write(tokenizer.decode(pred, skip_special_tokens=True) + '\n')
  f.close()


@torch.inference_mode()
def validate_general(args):
  print(args)

  print('Preparing data...')

  if args.gen_test_file is None:
    gen_test_file = args.data_dir + '/' + args.dataset + '/test.sum.general.jsonl'
  else:
    gen_test_file = args.gen_test_file
  gen_dev_dataset = SummarizationDataset(
    gen_test_file,
    use_keywords=args.use_keywords, use_switch=args.use_switch, shuffle=False)
  gen_dev_dataloader = DataLoader(gen_dev_dataset, batch_size=args.batch_size)
  f = open(gen_test_file, 'r', encoding='utf-8')
  lines = f.readlines()
  data = [json.loads(line) for line in lines]
  f.close()
  gen_gold_sums = [[summary.lower() for summary in inst['summary']] for inst in data]
  # gen_gold_sums = [inst['summary'].lower() for inst in data]

  print('Initializing model...')

  tokenizer = AutoTokenizer.from_pretrained(args.model_type)
  special_tokens = ['<rev>', '<key>', '<sum>', '<switch>']
  if args.use_switch != 'none':
    for i in range(args.num_aspects):
      special_tokens.append('<pos_%d>' % i)

  tokenizer.add_special_tokens(
    {'additional_special_tokens': special_tokens}
  )

  model = Model.from_pretrained(args.model_type, return_dict=True)
  model.resize_token_embeddings(len(tokenizer))
  model.cuda()

  if args.load_model is not None:
    print('Loading model...')
    best_point = torch.load(args.load_model)
    model.load_state_dict(best_point['model'])

  print('Start validation...')
  model.eval()

  # generate general summaries
  gen_pred_sums = []
  for _, (inp_batch, out_batch, _) in enumerate(tqdm(gen_dev_dataloader)):
    # bug-fix
    inp_batch = list(inp_batch)
    out_batch = list(out_batch)

    batch_encoding = tokenizer.prepare_seq2seq_batch(
      src_texts=inp_batch,
      tgt_texts=out_batch,
      max_length=args.max_length,
      max_target_length=args.max_target_length,
      padding=True,
      truncation=True,
      return_tensors='pt'
    )

    inp_ids = batch_encoding['input_ids'].cuda()

    preds = model.generate(
      inp_ids,
      min_length=60,
      max_length=args.max_target_length*2,
      num_beams=2,
      no_repeat_ngram_size=2,
      decoder_start_token_id=0,
      repetition_penalty=1,
      length_penalty=1,
    )

    for pred in preds:
      gen_pred_sums.append(tokenizer.decode(pred, skip_special_tokens=True))

  gen_scores = calculate(gen_gold_sums, gen_pred_sums)
  rouge_scores = list(gen_scores)

  print('\n', '-' * 50, '\n')

  print('r1, r2, rL: ', rouge_scores)
  rouge = np.power(np.product(rouge_scores), 1.0/len(rouge_scores))

  print("general ROUGE: %.4f" % rouge)

  print("General Gold:", gen_gold_sums[0])
  print("General Pred:", gen_pred_sums[0])


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('-mode', default='train', type=str)

  parser.add_argument('-dataset', default='dataset', type=str)
  parser.add_argument('-num_aspects', default=6, type=int)
  parser.add_argument('-model_name', default='sum', type=str)
  parser.add_argument('-load_model', default=None, type=str)

  parser.add_argument('-train_file', default=None, type=str)
  parser.add_argument('-asp_dev_file', default=None, type=str)
  parser.add_argument('-gen_dev_file', default=None, type=str)
  parser.add_argument('-asp_test_file', default=None, type=str)
  parser.add_argument('-gen_test_file', default=None, type=str)

  parser.add_argument('-data_dir', default='../data', type=str)
  parser.add_argument('-model_dir', default='models', type=str)

  parser.add_argument('-model_type', default='google/mt5-small', type=str)
  parser.add_argument('-model_dim', default=512, type=int)
  parser.add_argument('-use_keywords', default='input', type=str) # none, input, output
  parser.add_argument('-use_switch', default='input', type=str) # none, input, output

  parser.add_argument('-batch_size', default=4, type=int) # 16 -> 4
  parser.add_argument('-learning_rate', default=1e-6, type=float)
  parser.add_argument('-no_train_steps', default=100_000, type=int) # 500_000 -> 100_000
  parser.add_argument('-no_warmup_steps', default=5_000, type=int) # 20_000 -> 5_000
  parser.add_argument('-check_every', default=10_000, type=int) # 10_000
  parser.add_argument('-ckpt_every', default=10_000, type=int) # 10_000

  parser.add_argument('-max_length', default=512, type=int)

  parser.add_argument('-min_target_length', default=15, type=int)
  parser.add_argument('-max_target_length', default=128, type=int)
  parser.add_argument('-num_beams', default=2, type=int)
  parser.add_argument('-no_repeat_ngram_size', default=3, type=int)
  parser.add_argument('-repetition_penalty', default=1, type=float)
  parser.add_argument('-length_penalty', default=1, type=float)

  parser.add_argument('--use_wandb', action='store_true')

  args = parser.parse_args()
  if args.mode == 'train-general':
    train_general(args)
  elif args.mode == 'eval-general':
    evaluate(args, 'general')
  elif args.mode == 'validation-general':
    validate_general(args)