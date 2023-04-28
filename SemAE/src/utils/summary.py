from sklearn.metrics.pairwise import cosine_similarity
import string

PUNCT = set(string.punctuation)


# parts of the code has been
# adapted from: https://github.com/stangelid/qt

def truncate_summary(ranked_sentences,
                     max_tokens=75,
                     min_tokens=1,
                     cut_sents=False,
                     early_stop=True,
                     remove_non_alpha=True,
                     vectorizer=None,
                     cosine_threshold=None):
    '''Truncates a summary by iteratively adding sentences
       until the max_tokens limit is passed. 
    '''
    count = 0
    summary = []
    summary_sentence_ids = []

    if vectorizer is not None:
        assert cosine_threshold > 0 and cosine_threshold <= 1, \
                'cosine threshold should be in (0,1]'
        sentence_vecs = vectorizer.transform(ranked_sentences)
        similarities = cosine_similarity(sentence_vecs)

    for i, sentence in enumerate(ranked_sentences):
        if remove_non_alpha and all(c.isdigit() or c in PUNCT
                                    for c in sentence):
            continue

        if len(sentence.split()) < min_tokens:
            continue

        if vectorizer is not None and i > 0:
            similarities_to_existing = similarities[i, summary_sentence_ids]
            if not all(similarities_to_existing < cosine_threshold):
                continue

        summary.append(sentence)
        summary_sentence_ids.append(i)

        count += len(sentence.split())
        if count > max_tokens:
            if cut_sents:
                last_sent = summary[-1].split()
                last_sent = last_sent[:len(last_sent) - count + max_tokens]
                if len(last_sent) > 0:
                    summary[-1] = ' '.join(last_sent)
                else:
                    summary = summary[:-1]
                break
            else:
                summary = summary[:-1]
                if early_stop:
                    break
                else:
                    count -= len(sentence.split())
                    summary_sentence_ids = summary_sentence_ids[:-1]

    return summary
