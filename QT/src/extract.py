import os
import os.path
import json
import argparse
from random import seed

from sklearn.feature_extraction.text import TfidfVectorizer

import torch
from torch.utils.data import DataLoader
from torch.distributions.multinomial import Multinomial

from encoders import *
from quantizers import *
from train import *
from utils.data import *
from utils.summary import truncate_summary, RougeEvaluator

from datasets import load_metric

            
if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='Extracts general summaries with a trained QT model.\n' + \
                        'For usage example, refer to:\n' + \
                        '\thttps://github.com/stangelid/qt')

    data_arg_group = argparser.add_argument_group('Data arguments')
    data_arg_group.add_argument('--summary_data',
            help='summarization benchmark data',
            type=str, default='../data/dataset/test.json')
    data_arg_group.add_argument('--gold_data',
            help='gold data root directory',
            type=str, default='../data/gold')
    data_arg_group.add_argument('--gold_aspects',
            help='aspect categories to evaluate against (default: general only)',
            type=str, default='general')
    data_arg_group.add_argument('--sentencepiece',
            help='sentencepiece model file',
            type=str, default='../data/sentencepiece/spm_ru_unigram_32k.model')
    data_arg_group.add_argument('--max_rev_len',
            help='maximum number of sentences per review (default: 150)',
            type=int, default=150)
    data_arg_group.add_argument('--max_sen_len',
            help='maximum number of tokens per sentence (default: 40)',
            type=int, default=40)
    data_arg_group.add_argument('--split_by',
            help='how to split summary data (use "alphanum" for SPACE, ' + \
                 '"presplit" or "original" otherwise)',
            type=str, default='alphanum')

    summ_arg_group = argparser.add_argument_group('Summarizer arguments')
    summ_arg_group.add_argument('--model',
            help='trained QT model to use',
            type=str, default='')
    summ_arg_group.add_argument('--head',
            help='the output head to use for extraction (default: use all)',
            type=int, default=None)
    summ_arg_group.add_argument('--truncate_clusters',
            help='truncate cluster sampling to top-p % of clusters (if < 1) or top-k (if > 1)',
            type=float, default=0.10)
    summ_arg_group.add_argument('--num_cluster_samples',
            help='number of cluster samples (default: 300)',
            type=int, default=300)
    summ_arg_group.add_argument('--sample_sentences',
            help='enable 2-step sampling (sample sentences within cluster neighbourhood)',
            action='store_true')
    summ_arg_group.add_argument('--truncate_cluster_nn',
            help='truncate sentences that live in a cluster neighborhood (default: 5)',
            type=int, default=5)
    summ_arg_group.add_argument('--num_sent_samples',
            help='number of sentence samples per cluster sample (default: 30)',
            type=int, default=30)
    summ_arg_group.add_argument('--temp',
            help='temperature for sampling sentences within cluster (default: 1)',
            type=int, default=1)


    out_arg_group = argparser.add_argument_group('Output control')
    out_arg_group.add_argument('--outdir',
            help='directory to put summaries',
            type=str, default='outputs')
    out_arg_group.add_argument('--max_tokens',
            help='summary budget in words (default: 75)',
            type=int, default=75)
    out_arg_group.add_argument('--min_tokens',
            help='minimum summary sentence length in words (default: 2)',
            type=int, default=2)
    out_arg_group.add_argument('--cos_thres', 
            help='cosine similarity threshold for extraction (default: 0.75)',
            type=float, default=0.75)
    out_arg_group.add_argument('--no_cut_sents',
            help='don\'t cut last summary sentence',
            action='store_true')
    out_arg_group.add_argument('--no_early_stop',
            help='allow last sentence to go over limit',
            action='store_true')
    out_arg_group.add_argument('--newline_sentence_split',
            help='one sentence per line (don\'t use if evaluating with ROUGE)',
            action='store_true')

    other_arg_group = argparser.add_argument_group('Other arguments')
    other_arg_group.add_argument('--run_id',
            help='unique run id (for outputs)',
            type=str, default='general_run1')
    other_arg_group.add_argument('--no_eval',
            help='don\'t evaluate (just write summaries)',
            action='store_true')
    other_arg_group.add_argument('--gpu',
            help='gpu device to use (default: -1, i.e., use cpu)',
            type=int, default=-1)
    other_arg_group.add_argument('--batch_size',
            help='the maximum batch size (default: 5)',
            type=int, default=5)
    other_arg_group.add_argument('--sfp', 
            help='system filename pattern for pyrouge',
            type=str, default='(.*)')
    other_arg_group.add_argument('--mfp',
            help='model filename pattern for pyrouge',
            type=str, default='#ID#_[012].txt')
    other_arg_group.add_argument('--seed',
            help='random seed',
            type=int, default=1)

    other_arg_group.add_argument('--lang',
            type=str, default='ru')
    other_arg_group.add_argument('--entity_id',
            type=str, default='id')
    args = argparser.parse_args()

    seed(1)

    if args.gpu >= 0:
        device = torch.device('cuda:{0}'.format(args.gpu))
    else:
        device = torch.device('cpu')

    summ_data_path = args.summary_data
    model_path = args.model
    output_path = os.path.join(args.outdir, args.run_id)
    eval_path = args.outdir
    gold_path = args.gold_data
    spm_path = args.sentencepiece

    assert args.model != '', 'Please give model path'

    f = open(summ_data_path, 'r', encoding='utf-8')
    dataset_name = args.data.split('/')[2]
    if dataset_name == 'dataset':
        summ_data = json.load(f)
    else:
        summ_data = []
        for jline in f:
            item = json.loads(jline)
            summ_data.append(item)
    f.close()

    # prepare summarization dataset
    summ_dataset = ReviewSummarizationDataset(summ_data, spmodel=spm_path,
            max_rev_len=args.max_rev_len, max_sen_len=args.max_sen_len, entity_id_alias=args.entity_id)
    vocab_size = summ_dataset.vocab_size
    pad_id = summ_dataset.pad_id()
    bos_id = summ_dataset.bos_id()
    eos_id = summ_dataset.eos_id()
    unk_id = summ_dataset.unk_id()

    # wrapper for collate function
    collator = ReviewCollator(padding_idx=pad_id, unk_idx=unk_id,
                                bos_idx=bos_id, eos_idx=eos_id)

    # split dev/test entities
    summ_dataset.entity_split(split_by=args.split_by)

    # create entity data loaders
    summ_dls = {}
    summ_samplers = summ_dataset.get_entity_batch_samplers(args.batch_size)
    for entity_id, entity_sampler in summ_samplers.items():
        summ_dls[entity_id] = DataLoader(summ_dataset, batch_sampler=entity_sampler,
                collate_fn=collator.collate_reviews_with_ids)

    torch.manual_seed(args.seed)

    model = torch.load(model_path, map_location=device)
    nheads = model.encoder.output_nheads
    codebook_size = model.codebook_size
    d_model = model.d_model
    model.eval()

    all_texts = []
    ranked_entity_sentences = {}
    with torch.no_grad():
        for entity_id, entity_loader in summ_dls.items():
            inputs = []
            texts = []
            assignments = []
            quantized = []
            unquantized = []
            distances = []

            max_ntokens = 0
            for batch in entity_loader:
                src = batch[0].to(device)
                ids = batch[2]
                for full_id in ids:
                    entity_id, review_id = full_id.split('__')
                    texts.extend(summ_dataset.reviews[entity_id][review_id])

                batch_size, nsent, ntokens = src.size()
                max_ntokens = max(max_ntokens, ntokens)

                noq_out, q_out, clusters, dists = model.cluster(src)

                inputs.append(src.reshape(-1, ntokens))
                assignments.append(clusters.reshape(-1, nheads))
                quantized.append(q_out.reshape(-1, nheads, d_model))
                unquantized.append(noq_out.reshape(-1, nheads, d_model))
                distances.append(dists.reshape(-1, nheads, codebook_size))

            # all sentences (for tfidf vectorizer)
            all_texts.extend(texts)

            inputs = [torch.cat((src,
                         torch.ones(src.shape[0],
                           max_ntokens - src.shape[1]).to(device).type_as(src) * pad_id), dim=1)
                        for src in inputs]

            inputs = torch.cat(inputs, dim=0)                         # [S x T]
            assignments = torch.cat(assignments, dim=0)               # [S x H]
            quantized = torch.cat(quantized, dim=0)                   # [S x H x E]
            unquantized = torch.cat(unquantized, dim=0)               # [S x H x E]
            distances = torch.cat(distances, dim=0)                   # [S x H x C]

            nsent = assignments.shape[0]

            if args.head is not None:
                # using only 1 head for extraction
                assignments = assignments[:,args.head].unsqueeze(1)   # [S x 1]
                quantized = quantized[:,args.head].unsqueeze(1)       # [S x 1 x E]
                unquantized = unquantized[:,args.head].unsqueeze(1)   # [S x 1 x E]
                distances = distances[:,args.head].unsqueeze(1)       # [S x 1 x C]
                used_nheads = 1
            else:
                # using all heads
                used_nheads = nheads

            assignments_onehot = \
               torch.zeros(nsent, used_nheads, codebook_size).to(device)   # [S x H x C]
            assignments_onehot.scatter_(2, assignments.unsqueeze(2), 1)
            cluster_sizes = assignments_onehot.sum(dim=0)                  # [H x C]

            # truncating possible clusters to be sampled
            if args.truncate_clusters > 1:
                vals_to_keep, indices_to_keep = \
                        cluster_sizes.topk(int(args.truncate_clusters), dim=1)
                filtered_cluster_sizes = torch.zeros_like(cluster_sizes)
                filtered_cluster_sizes.scatter_(1, indices_to_keep, vals_to_keep)
                cluster_sizes = filtered_cluster_sizes
            elif args.truncate_clusters < 1:
                num_clusters_to_keep = \
                        (torch.sum(cluster_sizes > 0, dim=1)
                         * args.truncate_clusters).round().long()
                sorted_sizes, _ = cluster_sizes.sort(dim=1, descending=True)
                min_sizes = sorted_sizes.gather(1, num_clusters_to_keep.unsqueeze(1))
                indices_to_remove = torch.lt(cluster_sizes, min_sizes)
                filtered_cluster_sizes = torch.zeros_like(cluster_sizes)
                filtered_cluster_sizes.copy_(cluster_sizes)
                filtered_cluster_sizes[indices_to_remove] = 0
                cluster_sizes = filtered_cluster_sizes

            # sampling clusters
            # [H x nSamples]
            sampled_clusters = \
                    cluster_sizes.multinomial(args.num_cluster_samples, replacement=True)

            # [k x H x nSamples]
            sampled_clusters = \
                    sampled_clusters.expand(args.truncate_cluster_nn, -1, -1)

            # defining cluster neighborhoods
            # [k x H x C]
            cluster_knn_sent_dist, cluster_knn_sent_idx = \
                    torch.topk(distances, k=args.truncate_cluster_nn, dim=0, largest=False)

            # gathering all possible sentences to be sampled or ranked
            # [k x H x nSamples]
            sampled_neighbourhoods = cluster_knn_sent_idx.gather(2, sampled_clusters)

            if args.sample_sentences:
                # [k x H x nSamples]
                sampled_neighbourhood_dist = \
                        cluster_knn_sent_dist.gather(2, sampled_clusters)

                # multinomial over sentences
                multi = Multinomial(total_count=args.num_sent_samples,
                        logits=-sampled_neighbourhood_dist.transpose(0, 2) / args.temp)

                # [k x H x C]
                samples = multi.sample().transpose(0, 2)

                # [k x H x nSent]
                scattered_sampled_sentences_counts = \
                        torch.zeros(args.truncate_cluster_nn, used_nheads, nsent).to(device)
                scattered_sampled_sentences_counts.scatter_add_(2, sampled_neighbourhoods, samples)

                # [nSent]
                sampled_sentences_counts = scattered_sampled_sentences_counts.sum(dim=[0,1])
            else:
                # [k*H*nSample]
                sampled_sentences = sampled_neighbourhoods.flatten()

                # ranking by number of times a cluster was sampled
                # [nSent] 
                sampled_sentences_counts = sampled_sentences.bincount(minlength=nsent)

            # getting ranked sentence texts
            ranked_sentence_counts, ranked_sentence_indices = \
                    sampled_sentences_counts.sort(descending=True)
            ranked_sentence_texts = [texts[idx] for idx in ranked_sentence_indices]
            ranked_entity_sentences[entity_id] = ranked_sentence_texts

    # tfidf vectorizer used for cosine threshold
    if args.cos_thres != -1:
        vectorizer = TfidfVectorizer(decode_error='replace', stop_words='english') # no russian stopwords support
        vectorizer.fit(all_texts)
    else:
        vectorizer = None
    
    # write summaries
    os.makedirs(output_path, exist_ok=True)
    if args.newline_sentence_split:
        delim = '\n'
    else:
        delim = '\t'

    summary_dict = {}

    for entity_id, ranked_sentences in tqdm(ranked_entity_sentences.items()):
        summary_sentences = truncate_summary(
            ranked_sentences,
            max_tokens=args.max_tokens,
            cut_sents=(not args.no_cut_sents),
            vectorizer=vectorizer,
            cosine_threshold=args.cos_thres,
            early_stop=(not args.no_early_stop),
            min_tokens=args.min_tokens)

        summary_dict[entity_id] = delim.join(summary_sentences)

    golds = []
    preds = []

    entity_id = args.entity_id
    for h in summ_data:
        if h[entity_id] in summary_dict:
            if 'summary' in h:
                golds.append(h['summary'])
            else:
                golds.append(h['summaries']['general'])
            preds.append(summary_dict[h[entity_id]])

    # evaluate summaries
    metric = load_metric("rouge")

    result = metric.compute(predictions=preds, references=golds, language=args.lang, use_stemmer=True)
    result = {key: round(value.mid.fmeasure, 4) * 100 for key, value in result.items()}

    print('rouge-1: {}, rouge-2: {}, rouge-L: {}'.format(result['rouge1'], result['rouge2'], result['rougeL']))
