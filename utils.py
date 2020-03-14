from __future__ import unicode_literals, print_function, division

from io import open
import torch
import unicodedata
import re
import numpy as np


class Lang:
    def __init__(self, name, data):
        self.name = name
        self.data = data
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 0  # Count SOS and EOS
        self.vocab = set()

        self.create_index()

    def create_index(self):
        for sentence in self.data:
            self.vocab.update(sentence.split(' '))
        self.vocab = sorted(self.vocab)
        self.add_word('<pad>')
        for word in self.vocab:
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicode2ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters


def normalize_string(s):
    s = unicode2ascii(s.lower().strip())
    s = re.sub(r"([?.!,¿])", r" \1 ", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = '<sos>' + s + '<eos>'
    return s


def read_langs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8'). \
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]
    data1 = [pair[0] for pair in pairs]
    data2 = [pair[1] for pair in pairs]
    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        data1, data2 = data2, data1
        input_lang = Lang(lang2, data1)
        output_lang = Lang(lang1, data2)
    else:
        input_lang = Lang(lang1, data1)
        output_lang = Lang(lang2, data2)

    return input_lang, output_lang, pairs, data1, data2


def max_length(tensor):
    return max(len(t) for t in tensor)


def pad_sequences(x, max_len):
    padded = np.zeros(max_len, dtype=np.int64)
    if len(x) > max_len:
        padded[:] = x[:max_len]
    else:
        padded[:len(x)] = x
    return padded


def loss_function(criterion, real, pred):
    """ Only consider non-zero inputs in the loss; mask needed """
    # mask = 1 - np.equal(real, 0) # assign 0 to all above 0 and 1 to all 0s
    # print(mask)
    mask = real.ge(1).type(torch.FloatTensor)

    loss_ = criterion(pred, real) * mask
    return torch.mean(loss_)


def sort_batch(X, y, lengths):
    lengths, indx = lengths.sort(dim=0, descending=True)
    X = X[indx]
    y = y[indx]
    return X.transpose(0, 1), y, lengths  # transpose (batch x seq) to (seq x batch)
