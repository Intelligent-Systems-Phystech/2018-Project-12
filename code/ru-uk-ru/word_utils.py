import os
import sys
import re
import nltk
import queue
import numpy as np
import pandas as pd

import pickle
import unicodedata
import string

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def BLEU(ref_list, hyp_list):
  chencherry = SmoothingFunction()
  ref_lists = [[r] for r in ref_list]
  return corpus_bleu(ref_lists, hyp_list, smoothing_function=chencherry.method1)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-ZА-ЯҐЄІЇа-яґєії.!?]+", r" ", s)
    return s.strip()

def read_sentences(path):
    lines = []
    with open(path) as f:
        for line in f:
            lines.append(normalizeString(line))
    return lines

def seq_format(seq, max_words):
    nwords = len(seq)
    seq_new = seq + ["<EOS>"]
    seq_new += ["<PAD>" for i in range(max_words - nwords)]
    return seq_new

def freq_filter(seq, lang, freq):
    if freq == -1:
        return True
    for w in seq:
        if lang.word2count[w] < freq:
            return False
    return True

def len_filter(seq, max_len):
    if max_len == -1:
        return True
    else:
        return len(seq) <= max_len

def prepare_list(list, lang, max_words, freq):
    list_seq = [s.split() for s in list]
    list_clean = []
    for s in list_seq:
        if len_filter(s, max_words) and freq_filter(s, lang, freq):
            list_clean.append(s)
        else:
            list_clean.append(None)

    return list_clean

def seq2ind(seq, lang):
    return [lang.word2index[w] for w in seq]

def noise(seq, drop_prob=0.1, shuffle_len=3):
    n = len(seq)
    ind = np.argsort(np.arange(0, n) + np.random.uniform(0, shuffle_len, n))
    drop_mask = np.random.binomial(1, 1 - drop_prob, n).astype(np.bool)
    ind = ind[drop_mask]
    res = []
    for i in ind:
        res.append(seq[i])
    return res

def ind2words(ind_seq, vocab):
    """Translate word indices to words.

        Arguments:
        ind_seq  -- sequence of indices
        lang     -- corresponding vocabulary
    """
    return list(map(lambda x: vocab.index2word[x], ind_seq))

def words2sent(words):
    """Translate word indices to sentence.

        Arguments:
        words  -- sequence of words
        lang     -- corresponding vocabulary
    """
    try:
        end = words.index('<EOS>')
    except ValueError:
        end = len(words)
    return ' '.join([w for w in words[:end] if w not in ['<PAD>', '<SOS>', '<EOS>']])

def ind2sent(ind_seq, vocab):
    return words2sent(ind2words(ind_seq, vocab))

def load_vec(emb_path, max_words=-1):
    vectors = []
    word2id = {}
    it = 0
    with open(emb_path) as f:
        nvec, ndim = [int(k) for k in f.readline().split()]
        for line in f:
            if max_words != -1 and it > max_words:
                break
            it += 1
            orig_word, vect = line.rstrip().split(' ', 1)

            word = normalizeString(orig_word)
            vect = np.fromstring(vect, sep=' ')

            # Words are sorted by frequency, no need to add less
            # frequent version of the same word
            if not (word in word2id):
                vectors.append(vect)
                word2id[word] = len(word2id)

    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id
