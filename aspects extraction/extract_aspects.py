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
    string = re.sub(r"(http://)?www\.[^ ]+", " _url_ ", string)
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
    string = re.sub(r"(?<= )[0-9]+(?= )", "<NUM>", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def line_to_words(line, min_len, max_len, stop_words=None, lemmatize=False):
    """
    Splits a line of text into words, removes stop words and lemmatizes
    """
    line = line.strip()
    clean_line = clean_str(line)
    original = None

    if lemmatize:
        from nltk.stem.wordnet import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()

    words = clean_line.split()

    if stop_words is not None:
        words = [word for word in words if word not in stop_words]

    if lemmatize:
        words = [lemmatizer.lemmatize(word) for word in words]

    if len(words) < min_len:
        return None, None
    if len(words) <= max_len:
        return words, line
    else:
        words = words[:max_len]
        return words, line


def get_vocab(filename, min_len, max_len, max_seg, stop_words, lemmatize):
    """
    Constructs a vocabulary from the provided files
    """
    max_len_actual = 0
    max_seg_actual = 0

    wid = 1
    word2id = {'<PAD>':0}
    seg_cnt = 0
    doc_cnt = 0

    f_domain = open(filename, 'r')
    for line in tqdm(f_domain):
        inst = json.loads(line.strip()) # one product data
        reviews = inst['reviews'] # reviews for one product
        doc_cnt += len(reviews)

        for review in reviews:
            sentences = review['sentences'] # separate sentences of one review
            seg_cnt += len(sentences)

            # sanity check
            if len(sentences) > max_seg or len(sentences) < 1:
                continue
            sent_max_len = max([len(sentence.split()) for sentence in sentences])
            if (sent_max_len > max_len) or (sent_max_len < min_len):
                continue

            for sent in sentences:
                seg, original = line_to_words(sent, min_len, max_len, stop_words, lemmatize)
                if seg is not None: # sentence has words after cleaning
                    max_len_actual = max(max_len_actual, len(seg))
                    for word in seg:
                        if word not in word2id:
                            word2id[word] = wid
                            wid += 1

            max_seg_actual = max(max_seg_actual, len(sentences))

            # review = ' '.join(sentences).split() # full text of one review
    f_domain.close()

    return max_len_actual, max_seg_actual, seg_cnt, doc_cnt, word2id


def load_data(filename, args):
    """
    Reads the provided files and loads the data splits
    """
    padding = args.padding
    min_len = args.min_len
    max_len = args.max_len
    max_seg = args.max_seg
    batch_size = args.batch_size
    stop_words = args.stop_words
    lemmatize = args.lemmatize

    max_len_actual, max_seg_actual, seg_cnt, doc_cnt, word2id = get_vocab(filename,
            min_len, max_len, max_seg, stop_words, lemmatize)

    print('Number of documents:', doc_cnt)
    print('Number of segments:' , seg_cnt)
    print('Max segment length:', max_len_actual)
    print('Max number of segments:', max_seg_actual)
    print('Vocabulary size:', len(word2id))

    f_domain = open(filename, 'r')

    insts = []
    data = []
    data_lbl = []
    data_orig = []

    for line in f_domain:
        inst = json.loads(line.strip()) # one product data

        insts.append(inst)
        data.append([])
        data_lbl.append([])
        data_orig.append([])
    f_domain.close()

    for inst, docs, lbls, orig in tqdm(zip(insts, data, data_lbl, data_orig)):
        reviews = inst['reviews'] # reviews for one product
        for review in reviews:
            sentences = review['sentences'] # separate sentences of one review

            # sanity check
            if len(sentences) > max_seg or len(sentences) < 1:
                continue
            sent_max_len = max([len(sentence.split()) for sentence in sentences])
            if (sent_max_len > max_len) or (sent_max_len < min_len):
                continue

            doc = []
            doc_orig = []
            label = int(review['rating'])

            for sent in sentences:
                if len(doc) >= max_seg:
                    continue
                seg, original = line_to_words(sent, min_len, max_len,
                        stop_words=stop_words, lemmatize=lemmatize)

                if seg is not None:
                    seg_ids = [word2id[word] for word in seg]
                    seg_ids = [0] * padding + seg_ids + [0] * padding
                    doc.append(seg_ids)
                    doc_orig.append(original)

            if len(doc) > 0:
                docs.append(doc)
                lbls.append(label)
                orig.append('\n'.join(doc_orig))
    

    return word2id, data, data_lbl, data_orig


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
    word2id, all_docs, all_label, all_original = load_data(args.train, args)

    assert len(all_docs) > 0, 'loading data failed'

    print('loaded data')

    # writes vocabulary file
    with open(args.name + '_word_mapping.txt', 'w') as f:
      for word, idx in sorted(word2id.items(), key=operator.itemgetter(1)):
        f.write("%s %d\n" % (word, idx))

    # loads word embeddings
    wv = api.load('word2vec-google-news-300')
    vocab_size = len(word2id) + 1
    embed = np.random.uniform(-0.25, 0.25, (vocab_size, 300))
    embed[0] = 0
    for key in tqdm(wv.key_to_index.keys()):
        if key in word2id:
            embed[word2id[key]] = wv[key]
    del wv

    print('loaded word embeddings')

    # shuffles data
    train_data = sorted(sample(
        list(zip(all_docs[0], all_label[0], all_original[0])), len(all_docs[0])),
        key=special_len)

    # if len(all_docs) > 1:
    #     test_data = sorted(sample(
    #         zip(all_docs[1], all_label[1], all_original[1]), len(all_docs[1])),
    #         key=special_len)
    # else:
    #     test_data = []
    test_data = []

    # if len(all_docs) > 2:
    #     dev_data = sorted(sample(
    #         zip(all_docs[2], all_label[2], all_original[2]), len(all_docs[2])),
    #         key=special_len)
    # else:
    #     dev_data = []
    dev_data = []

    filename = args.name + '.hdf5'
    with h5py.File(filename, 'w') as f:
        f['w2v'] = np.array(embed) # array of w2v vectors for words in domain

        # splits data into batches and pads where necessary
        for i, chunk in tqdm(enumerate(chunks(train_data, args.batch_size))):
            docs, lbls, original = map(list, zip(*chunk))

            max_len_batch = 0
            max_seg_batch = len(max(docs, key=len))

            for j in range(len(docs)):
                max_len_batch = max(max_len_batch, len(max(docs[j], key=len)))

            for j in range(len(docs)):
                original[j] += '\n' * (max_seg_batch - len(docs[j]))
                docs[j].extend([[0] * max_len_batch] * (max_seg_batch - len(docs[j])))
                for k in range(len(docs[j])):
                    docs[j][k].extend([0] * (max_len_batch - len(docs[j][k])))

            f['train/' + str(i)] = np.array(docs, dtype=np.int32)
            f['train_label/' + str(i)] = np.array(lbls, dtype=np.int32)
            dt = h5py.special_dtype(vlen=bytes)
            # f.create_dataset('train_original/' + str(i),
            #         (len(original),), dtype=dt, data=original)

        for i, chunk in enumerate(chunks(test_data, args.batch_size)):
            docs, lbls, original = map(list, zip(*chunk))

            max_len_batch = 0
            max_seg_batch = len(max(docs, key=len))

            for j in range(len(docs)):
                max_len_batch = max(max_len_batch, len(max(docs[j], key=len)))

            for j in range(len(docs)):
                original[j] += '\n' * (max_seg_batch - len(docs[j]))
                docs[j].extend([[0] * max_len_batch] * (max_seg_batch - len(docs[j])))
                for k in range(len(docs[j])):
                    docs[j][k].extend([0] * (max_len_batch - len(docs[j][k])))

            f['test/' + str(i)] = np.array(docs, dtype=np.int32)
            f['test_label/' + str(i)] = np.array(lbls, dtype=np.int32)
            dt = h5py.special_dtype(vlen=bytes)
            f.create_dataset('test_original/' + str(i),
                    (len(original),), dtype=dt, data=original)

        for i, chunk in enumerate(chunks(dev_data, args.batch_size)):
            docs, lbls, original = map(list, zip(*chunk))

            max_len_batch = 0
            max_seg_batch = len(max(docs, key=len))

            for j in range(len(docs)):
                max_len_batch = max(max_len_batch, len(max(docs[j], key=len)))

            for j in range(len(docs)):
                original[j] += '\n' * (max_seg_batch - len(docs[j]))
                docs[j].extend([[0] * max_len_batch] * (max_seg_batch - len(docs[j])))
                for k in range(len(docs[j])):
                    docs[j][k].extend([0] * (max_len_batch - len(docs[j][k])))

            f['dev/' + str(i)] = np.array(docs, dtype=np.int32)
            f['dev_label/' + str(i)] = np.array(lbls, dtype=np.int32)
            dt = h5py.special_dtype(vlen=bytes)
            f.create_dataset('dev_original/' + str(i),
                    (len(original),), dtype=dt, data=original)


    # custom part
    id2word = {}
    word2id = {}

    fvoc = open(args.name + '_word_mapping.txt', 'r')
    for line in fvoc:
        word, id = line.split()
        id2word[int(id)] = word
        word2id[word] = int(id)
    fvoc.close()


    f = h5py.File(args.name + '.hdf5', 'r')
    w_emb_array = f['w2v'][()]
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
