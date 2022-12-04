import json
import os
from tqdm import tqdm
import random
import numpy as np
import h5py
import re
import sys
import operator
import argparse
from random import sample, seed
from tqdm import tqdm
import gensim.downloader as api
import torch
from scipy.cluster.vq import kmeans
from sklearn.neighbors import NearestNeighbors
import yake
from transformers import AutoModel, AutoTokenizer


'''
In order to run the script you need to download data from https://drive.google.com/drive/folders/1yofbWkGr5PU474N0dP5aSK1mSTOpiiP6?usp=sharing
and create ./data/preprocessed folder for temporary results
'''


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

def special_len(tup):
    """
    comparison function that will sort a document:
    (a) according to the number of segments
    (b) according to its longer segment
    """
    doc = tup[0] if type(tup) is tuple else tup
    return (len(doc), len(max(doc, key=len)))


def clean_str(string):
    """
    Tokenization/string cleaning.
    """
    string = string.lower()
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"&#34;", " ", string)
    string = re.sub(r"(http://)?www\.[^ ]+", "", string)
    string = re.sub(r"<s>", " ", string)
    string = re.sub(r"[^a-z0-9$\'_]", " ", string)
    string = re.sub(r"_{2,}", "_", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'m", " \'m", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\$+", " $ ", string)
    string = re.sub(r"rrb", " ", string)
    string = re.sub(r"lrb", " ", string)
    string = re.sub(r"rsb", " ", string)
    string = re.sub(r"lsb", " ", string)
    string = re.sub(r"(?<=[a-z])I", " I", string)
    string = re.sub(r"(?<= )[0-9]+(?= )", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def load_data(filename, args):
    """
    Reads the provided files and loads the data splits
    """
    f_domain = open(filename, 'r')

    data = []

    for line in f_domain:
        inst = json.loads(line.strip())
        reviews = inst['reviews'] # reviews for one product

        sub_texts = []
        for review in tqdm(reviews):
            clean_review = clean_str(' '.join(review['sentences']))
            sub_texts.append(clean_review)

        text = '\n'.join(sub_texts)

        custom_kw_extractor = yake.KeywordExtractor(n=2, top=100) # TODO: get 10% of words as keywords
        keywords = custom_kw_extractor.extract_keywords(text)
        keywords = [item[0] for item in keywords]
        data.append(keywords)

    f_domain.close()

    # create vocabulary
    stop_words = args.stop_words
    lemmatize = args.lemmatize

    wid = 1
    word2id = {'<PAD>':0}

    for keywords in tqdm(data):
        if stop_words is not None:
            keywords = [word for word in keywords if word not in stop_words]

        if lemmatize:
            from nltk.stem.wordnet import WordNetLemmatizer
            lemmatizer = WordNetLemmatizer()

            keywords = [lemmatizer.lemmatize(word) for word in keywords]

        for word in keywords:
            if word not in word2id:
                word2id[word] = wid
                wid += 1

    print('Vocabulary size:', len(word2id))

    return word2id


def main():
    parser = argparse.ArgumentParser(
      description =__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--w2v', help='word2vec-style binary file', type=str, default='')
    parser.add_argument('--name', help='name of created dataset', type=str, default='')
    parser.add_argument('--train', help='data in appropriate format', type=str, default='')
    parser.add_argument('--test', help='data in appropriate format', type=str, default='')
    parser.add_argument('--dev', help='data in appropriate format', type=str, default='')
    parser.add_argument('--batch_size', help='number of documents per batch (default: 200)', type=int, default=200)
    parser.add_argument('--padding', help='padding around each segment (default: 2)', type=int, default=2)
    parser.add_argument('--lemmatize', help='Lemmatize words', action='store_true')
    parser.add_argument('--stopfile', help='Stop-word file', type=str, default='')
    parser.add_argument('--min_len', help='minimum allowed words per segment (Default: 2)', type=int, default=2)
    parser.add_argument('--max_len', help='maximum allowed segments per document (Default: 100)', type=int, default=100)
    parser.add_argument('--max_seg', help='maximum allowed words per segment (Default: 100)', type=int, default=100)
    parser.add_argument('--seed', help='random seed (default: 1)', type=int, default=1)
    parser.add_argument('--aspects', help='aspects num (default: 10)', type=int, default=10)
    parser.add_argument('--kmeans_iter', help='kmeans iter (default: 5)', type=int, default=5)
    parser.add_argument('--num_seed_words', help='num seed words (default: 5)', type=int, default=5)
    parser.add_argument('--seeds_dir', help='seed words directory', type=str, default='')
    args = parser.parse_args()

    seed(args.seed)

    if args.stopfile == 'no':
        args.stop_words = None
    elif args.stopfile != '':
        stop_words = set()
        fstop = open(args.stopfile, 'r')
        for line in fstop:
            stop_words.add(line.strip())
        fstop.close()
        args.stop_words = stop_words
    else:
        from nltk.corpus import stopwords
        args.stop_words = set(stopwords.words('english'))

    # loads data
    word2id = load_data(args.train, args)

    print('loaded data')

    # writes vocabulary file
    with open(args.name + '_word_mapping.txt', 'w') as f:
      for word, idx in sorted(word2id.items(), key=operator.itemgetter(1)):
        f.write("%s %d\n" % (word, idx))

    # loads word embeddings
    # wv = api.load('word2vec-google-news-300')
    # vocab_size = len(word2id) + 1
    # embed = np.random.uniform(-0.25, 0.25, (vocab_size, 300))
    # embed[0] = 0
    # for key in tqdm(word2id.keys()):
    #     if key in wv:
    #         embed[word2id[key]] = wv[key]
    # del wv
    checkpoint = 'bert-base-uncased'
    model = AutoModel.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    vocab_size = len(word2id) + 1
    embed = np.random.uniform(-0.25, 0.25, (vocab_size, 768))
    embed[0] = 0
    for key in tqdm(word2id.keys()):
        tok_word = tokenizer([key], return_tensors='pt')
        embedding = model(**tok_word).last_hidden_state.detach().cpu()
        embedding = embedding.mean(dim=1)

        embed[word2id[key]] = embedding

    print('loaded word embeddings')

    filename = args.name + '.hdf5'
    with h5py.File(filename, 'w') as f:
        f['emb'] = np.array(embed)


    # custom part
    id2word = {}
    word2id = {}

    fvoc = open(args.name + '_word_mapping.txt', 'r')
    for line in fvoc:
        # word, id = line.split()
        line = line.split()
        id = line[-1]
        word = ' '.join(line[:-1])
        id2word[int(id)] = word
        word2id[word] = int(id)
    fvoc.close()


    f = h5py.File(args.name + '.hdf5', 'r')
    w_emb_array = f['emb'][()]
    w_emb = torch.from_numpy(w_emb_array)
    vocab_size, emb_size = w_emb.size()
    f.close()

    # kmeans initialization (ABAE)
    print('Running k-means...')
    a_emb, _ = kmeans(w_emb_array, args.aspects, iter=args.kmeans_iter)
    a_emb = torch.from_numpy(a_emb)

    # save aspects embeddings
    torch.save({
        'aspects_emb': a_emb
    }, './data/preprocessed/aspects_embs')

    # find top-k close words and aspect names
    nn = NearestNeighbors(n_neighbors=args.num_seed_words).fit(w_emb_array[1:]) # excluding <PAD> special token
    _, idxs = nn.kneighbors(a_emb)
    idxs += 1

    # TODO: get aspect names from a_emb[i] with some decoder
    # aspect_names = 

    for k in tqdm(range(args.aspects)):
        aspect_name = k # aspect_names[k]
        sw = list(map(lambda x: id2word[x], idxs[k]))
        f = open(f'{args.seeds_dir}/{aspect_name}.txt', 'w')
        print(*sw, sep='\n', end='', file=f)
        f.close()

if __name__ == '__main__':
    main()