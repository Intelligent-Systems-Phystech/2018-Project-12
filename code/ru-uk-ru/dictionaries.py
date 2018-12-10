from word_utils import *
import torch
import time

class Vocabulary:
    """
    Tracks info about known words, their indices and frequences.
    """

    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0
        self.add_seq(self.get_dummies())

    def add_list(self, list):
        for s in list:
            self.add_sentence(s)

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_seq(self, seq):
        for word in seq:
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word.append(word)
            self.n_words += 1
        else:
            self.word2count[word] += 1

    @staticmethod
    def get_dummies():
        return ["<SOS>", "<EOS>", "<PAD>", "<UNK>"]

    @staticmethod
    def get_dummy_ind(w):
        return Vocabulary.get_dummies().index(w)


class Dataset:

    def __init__(self, lang_info, max_len=-1, min_freq=-1, val_ratio=0.1):
        self.max_len = max_len
        self.min_freq = min_freq

        self.s_list = {}

        if len(lang_info) != 2:
            raise ValueError('Only pairs of languages are supported, but {} was passed.'.format(len(lang_info)))
        self.names = []
        for l, path in lang_info.items():
            self.names.append(l)
            self.s_list[l] = read_sentences(path)

        self.v_list = {}
        for l in self.names:
            self.v_list[l] = Vocabulary(l)
            self.v_list[l].add_list(self.s_list[l])

        self.seq_list = {}
        nsents = []

        seq_list = []
        seq_names = []
        for l in self.names:
            tmp = prepare_list(self.s_list[l], self.v_list[l],
                               max_len, min_freq)
            nsents.append(len(tmp))
            seq_list.append(tmp)
            seq_names.append(l)

        if len(set(nsents)) != 1:
            raise Warning('Numbers of sentences are not equal for the languages.')

        nopair = {l: [] for l in seq_names}
        pair = {l: [] for l in seq_names}
        nfiltered = [0, 0]
        npairs = 0
        for s in zip(*seq_list):
            if s[0] == None and s[1] == None:
                continue
            if s[1] == None:
                nfiltered[0] += 1
                nopair[seq_names[0]].append(s[0])
            elif s[0] == None:
                nfiltered[1] += 1
                nopair[seq_names[1]].append(s[1])
            else:
                nfiltered[0] += 1
                nfiltered[1] += 1
                npairs += 1
                for i in range(2):
                    pair[seq_names[i]].append(s[i])

        wanted_len = int(val_ratio * min(nfiltered))
        if wanted_len > npairs:
            raise Warning('Asked for {} test samples, but only {} can be provided.'.format(wanted_len, npairs))
        res_len = min(npairs, wanted_len)
        self.test_list = {}
        self.seq_list = {}
        self.val_size = res_len
        for l in self.names:
            self.seq_list[l] = pair[l][res_len:] + nopair[l]
            self.test_list[l] = pair[l][:res_len]

        self.seq_tr_list = {}
        for l in self.names:
            self.seq_tr_list[l] = None

        self.emb = {}
        for l in self.names:
            self.emb[l] = None

    def translate(self, translator, info_timeout=30):
        other = dict(zip(self.names, self.names[::-1]))
        start = time.time()
        for l, seq_list in self.seq_list.items():
            n = len(seq_list)
            self.seq_tr_list[l] = []
            for i, s in enumerate(seq_list):
                seq_tr = translator.translate_seq(s, l, other[l])
                self.seq_tr_list[l].append(seq_tr)
                self.v_list[other[l]].add_seq(seq_tr)
                if not i % 100:
                    end = time.time()
                if end - start > info_timeout:
                    start = end
                    print('[{}] {:.1f}% done'.format(l, i / n * 100))

    def get_train(self, batch_size=1):
        X_auto = {}
        Y_auto = {}

        X_cross = {}
        Y_cross = {}
        other = dict(zip(self.names, self.names[::-1]))
        for l, lang in self.v_list.items():
            batch_ind = np.random.choice(range(len(self.seq_list[l])), batch_size, replace=False)
            seq_list_tmp = [self.seq_list[l][i] for i in batch_ind]

            X_auto_tmp = list(map(noise, seq_list_tmp))
            Y_auto_tmp = seq_list_tmp

            batch_ind = np.random.choice(range(len(self.seq_tr_list[l])), batch_size, replace=False)
            seq_list_tmp = [self.seq_list[l][i] for i in batch_ind]
            seq_tr_list_tmp = [self.seq_tr_list[l][i] for i in batch_ind]

            X_cross_tmp = list(map(noise, seq_tr_list_tmp))
            Y_cross_tmp = seq_list_tmp

            vocabs = 3 * [self.v_list[l]] + [self.v_list[other[l]]]
            seq_lists = [X_auto_tmp, Y_auto_tmp, Y_cross_tmp, X_cross_tmp]
            ind_lists = []
            for lang, seq_list in zip(vocabs, seq_lists):
                max_len = max(list(map(len, seq_list)))
                formatted = list(map(lambda x: seq_format(x, max_len), seq_list))
                inds = torch.tensor(list(map(lambda x: seq2ind(x, lang), formatted)))
                ind_lists.append(inds)

            X_auto[l], Y_auto[l], Y_cross[l], X_cross[l] = ind_lists

        return X_auto, Y_auto, X_cross, Y_cross

    def get_test(self, nsamples=-1):
        X = {}
        if nsamples == -1:
            nsamples = self.val_size
        inds = np.random.choice(range(self.val_size), nsamples, replace=False)
        for l, lang in self.v_list.items():
            test_list_tmp = [self.test_list[l][i] for i in inds]
            max_len = max(list(map(len, test_list_tmp)))
            formatted = list(map(lambda x: seq_format(x, max_len), test_list_tmp))
            X[l] = torch.tensor(list(map(lambda x: seq2ind(x, lang), formatted)))
        return X

    def build_initial_embedding(self, lang_info):
        for l, path in lang_info.items():
            embeddings, id2word, word2id = load_vec(path)
            cur_voc = self.v_list[l]

            dum_len = len(cur_voc.get_dummies())
            emb = torch.zeros((cur_voc.n_words, embeddings.shape[1] + dum_len))
            for i in range(dum_len):
                emb[i, -dum_len + i] = 1

            for i in range(dum_len, cur_voc.n_words):
                w = cur_voc.index2word[i]
                if w in word2id:
                    emb[i, :-dum_len] = torch.from_numpy(embeddings[word2id[w]])
                else:
                    x = torch.rand(embeddings.shape[1]) + 1e-8
                    emb[i, :-dum_len] = x / x.norm()

            self.emb[l] = emb

    def ind2sent(self, ind_seq, lang):
        return ind2sent(ind_seq, self.v_list[lang])

    def ind2words(self, ind_seq, lang):
        return ind2words(ind_seq, self.v_list[lang])


class NaiveTranslator:

    def __init__(self, lang_info, max_words=-1):
        self.emb = {}
        self.id2word = {}
        self.word2id = {}
        self.names = []
        for l, path in lang_info.items():
            self.names.append(l)
            self.emb[l], self.id2word[l], self.word2id[l] = load_vec(path,
                                                                     max_words)

        self.cache = {l: {} for l in self.names}
        for l in self.names:
            for w in ["<SOS>", "<EOS>", "<PAD>", "<UNK>"]:
                self.cache[l][w] = w

    def translate(self, word, from_lang, to_lang):
        if word in self.cache[from_lang]:
            return self.cache[from_lang][word]
        else:
            if word in self.word2id[from_lang]:
                id = self.word2id[from_lang][word]
            else:
                self.cache[from_lang][word] = "<UNK>"
                return "<UNK>"

            vec = self.emb[from_lang][id]
            dist = np.dot(self.emb[to_lang], vec)
            ind = np.asscalar(np.argmax(dist, axis=0))
            tr = self.id2word[to_lang][ind]
            self.cache[from_lang][word] = tr
            return tr

    def translate_sent(self, sent, from_lang, to_lang):
        new_sent = ' '.join([self.translate(w, from_lang, to_lang) for w in sent.split()])
        return new_sent

    def translate_seq(self, seq, from_lang, to_lang):
        return [self.translate(w, from_lang, to_lang) for w in seq]