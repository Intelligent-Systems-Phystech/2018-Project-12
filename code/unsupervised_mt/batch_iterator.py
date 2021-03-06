import numpy as np
import torch
from unsupervised_mt.utils import pad_monolingual_batch


class BatchIterator:
    def __init__(self, dataset):
        self.languages = dataset.languages
        self.load_sentence = dataset.load_sentence
        self.train_ids = dataset.train_ids
        self.test_ids = dataset.test_ids
        self.pad_index = {l: dataset.vocabs[l].get_pad(l) for l in self.languages}

    def load_raw_monolingual_batch(self, batch_size, language, test=False, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)

        ids = self.test_ids if test else self.train_ids
        random_ids = np.random.choice(ids[language], size=batch_size)
        return [self.load_sentence(language, idx) for idx in random_ids]

    def load_monolingual_batch(self, batch_size, language, test=False, random_state=None):
        return torch.tensor(pad_monolingual_batch(
            self.load_raw_monolingual_batch(batch_size, language, test, random_state),
            self.pad_index[language]
        ), dtype=torch.long).transpose(0, 1)

    def load_batch(self, batch_size, test=False, random_state=None):
        return {l: self.load_monolingual_batch(batch_size, l, test, random_state) for l in self.languages}

    def batch_generator(self, batch_size):
        while True:
            yield self.load_batch(batch_size)





