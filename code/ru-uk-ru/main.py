# wget -nc https://raw.githubusercontent.com/Intelligent-Systems-Phystech/2018-Project-12/master/data/opus/samples.ru
# wget -nc https://raw.githubusercontent.com/Intelligent-Systems-Phystech/2018-Project-12/master/data/opus/samples.uk
# wget -nc https://s3.amazonaws.com/arrival/embeddings/wiki.multi.ru.vec
# wget -nc https://s3.amazonaws.com/arrival/embeddings/wiki.multi.uk.vec

import pickle
import numpy as np
import unicodedata
import string
import re
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import time
import gc

LOAD_PICKLED = False
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cuda:7' # 1080Ti


from dictionaries import *
from model import *


# ################################################################
if LOAD_PICKLED:
    with open('NaiveTranslator', 'rb') as f:
        tr = pickle.load(f)
else:
    lang_info = {'ru': 'wiki.multi.ru.vec',
                 'uk': 'wiki.multi.uk.vec'}

    tr = NaiveTranslator(lang_info)
    with open('NaiveTranslator', 'wb') as f:
        pickle.dump(tr, f)

max_in_len = 20
min_in_freq = 5

if LOAD_PICKLED:
    with open('Dataset', 'rb') as f:
        D = pickle.load(f)
else:
    lang_info = {'ru': 'samples.ru',
                 'uk': 'samples.uk'}

    print('-> Loading dataset')
    D = Dataset(lang_info, max_in_len, min_in_freq)
    print('-> Translating')
    D.translate(tr)
    print('-> Building embedding')
    lang_info = {'ru': 'wiki.multi.ru.vec',
                 'uk': 'wiki.multi.uk.vec'}
    D.build_initial_embedding(lang_info)

    with open('Dataset', 'wb') as f:
        pickle.dump(D, f)

train = D.get_train(1)

for l in ['ru', 'uk']:
    print('{} train samples: {}'.format(l, len(D.seq_list[l])))
    print('{} test samples: {}'.format(l, len(D.test_list[l])))

print('\nExamples:')
l = 'ru'
other = 'uk'

print('X_auto:', D.ind2sent(train[0][l][0], l))
print('Y_auto:', D.ind2sent(train[1][l][0], l))
print('X_cross:', D.ind2sent(train[2][l][0], other))
print('Y_cross:', D.ind2sent(train[3][l][0], l))

import json

log = None
try:
    log = open('ML.log', 'a')
except:
    log = open('ML.log', 'w')


#####################3####################################

d_loss_for_plot = []
tr_loss_for_plot = []
iteration_for_plot = []

niters = 100000
batch_size = 64

# Number of iterations between validations
val_per = 50
# Batch size for validation (-1 == use full test set)
val_size = -1
# Number of iterations between printing current iteration
it_per = 5
# Number of iterations between model saves
save_per = 1000

tmp_json = {'decoder_size': 800, 'attn_size': 70, 'discr_size': 300, 'border_loss': 11, 'dataset': 'wiki'}

# Decoder hidden state size
decoder_size = tmp_json['decoder_size']
# Attention net size
attn_size = tmp_json['attn_size']
# Discriminator size
discr_size = tmp_json['discr_size']
# Start embeddings optimisation only when loss is sufficiently small
border_loss = tmp_json['border_loss']

dataset = tmp_json['dataset']

# Ability to translate
tr_crit = nn.NLLLoss()
# Ability to fool the discriminator
tr_fake_crit = nn.BCELoss()
# Ability to predict language correctly
discr_crit = nn.BCELoss()

wr = Wrapper(decoder_size, attn_size, discr_size, D).to(device)

d_opt = torch.optim.Adam(wr.discr.parameters(), lr=0.0003)
tr_opt = torch.optim.Adam(list(wr.enc.parameters()) + list(wr.dec.parameters()),
                          lr=0.001, betas=(0.5, 0.999))

class_num = {name: cl for cl, name in enumerate(D.names)}
other = dict(zip(D.names, D.names[::-1]))
pad_ind = Vocabulary.get_dummy_ind('<PAD>')

start = time.time()
avg_d_loss = 0
avg_tr_loss = 0

for name in D.names:
    wr.emb[name].weight.requires_grad = False

for it in range(0, niters):
    tr_opt.zero_grad()
    d_opt.zero_grad()

    X_auto, Y_auto, X_cross, Y_cross = D.get_train(batch_size)

    d_loss = 0

    tr_loss = 0
    for l in D.names:
        X_auto[l] = X_auto[l].to(device)
        Y_auto[l] = Y_auto[l].to(device)
        X_cross[l] = X_cross[l].to(device)
        Y_cross[l] = Y_cross[l].to(device)

        encoder_outputs, decoder_outputs = \
            wr.encode_decode(X_auto[l], l, l, Y_auto[l].shape[1])

        if torch.any(Y_auto[l] == pad_ind):
            decoder_outputs[Y_auto[l] == pad_ind][:, pad_ind] = 0
        tr_loss += tr_crit(decoder_outputs.transpose(1, 2), Y_auto[l])

        predicted = wr.discriminate(encoder_outputs)
        wanted = torch.full_like(predicted, class_num[other[l]], device=device)
        tr_loss += tr_fake_crit(predicted, wanted)

        correct = torch.full_like(predicted, class_num[l], device=device)
        predicted_det = wr.discriminate(encoder_outputs.detach())
        d_loss += discr_crit(predicted_det, correct)

        encoder_outputs, decoder_outputs = \
            wr.encode_decode(X_cross[l], other[l], l, Y_cross[l].shape[1])
        if torch.any(Y_cross[l] == pad_ind):
            decoder_outputs[Y_cross[l] == pad_ind][:, pad_ind] = 0
        tr_loss += tr_crit(decoder_outputs.transpose(1, 2), Y_cross[l])

        predicted = wr.discriminate(encoder_outputs)
        wanted = torch.full_like(predicted, class_num[l], device=device)
        tr_loss += tr_fake_crit(predicted, wanted)

        correct = torch.full_like(predicted, class_num[other[l]], device=device)
        predicted_det = wr.discriminate(encoder_outputs.detach())
        d_loss += discr_crit(predicted_det, correct)

    avg_d_loss += d_loss.item()
    avg_tr_loss += tr_loss.item()

    d_loss_for_plot.append(d_loss.item())
    tr_loss_for_plot.append(tr_loss.item())
    iteration_for_plot.append(it)

    for name in D.names:
        if tr_loss.item() < border_loss:
            wr.emb[name].weight.requires_grad = True
        else:
            wr.emb[name].weight.requires_grad = False

    for p in wr.discr.parameters():
        p.requires_grad = False

    tr_loss.backward()
    tr_opt.step()

    for p in wr.discr.parameters():
        p.requires_grad = True

    d_loss.backward()
    d_opt.step()

    if not it % it_per:
        end = time.time()
        print('Iterations: {} ({} sec/iter)'.format(it, (end - start) / it_per))
        print('Average losses:')
        print('\td_loss:', avg_d_loss / it_per)
        print('\ttr_loss:', avg_tr_loss / it_per)

        start = time.time()

        itt = np.array(iteration_for_plot)
        itt *= 10

        fig = plt.figure(figsize=(12, 6))
        axar = fig.subplots(1, 2)
        ax = axar[0]
        ax.grid()
        ax.plot(itt, tr_loss_for_plot, label='tr_loss')
        ax.set_xlabel('iterations')
        ax.set_ylabel('loss')
        ax.legend()

        ax = axar[1]
        ax.grid()
        ax.plot(itt, d_loss_for_plot, label='d_loss')
        ax.set_xlabel('iterations')
        ax.set_ylabel('loss')
        ax.legend()

        fig.suptitle('WIKI: decoder_size={},attn_size={},discr_size={}.pdf'.format(decoder_size, attn_size, discr_size),
                     fontsize=16)

        fig.savefig(
            '/content/gdrive/My Drive/pictures/WIKI,decoder_size={},attn_size={},discr_size={}.png'.format(decoder_size,
                                                                                                           attn_size,
                                                                                                           discr_size))
        fig = None

        avg_tr_loss = 0
        avg_d_loss = 0

        gc.collect()

    if not it % val_per:
        print('Last loss:')
        print('  d_loss =', d_loss.data)
        print('  tr_loss =', tr_loss.data)

        print(f'\n[{it}] Validation:')

        wr_tr = IndTranslator(wr, max_in_len)
        X = D.get_test(val_size)
        ref_list = []
        hyp_list = []
        print('Sample translations:')
        for l in D.names:
            print('\t', other[l], ' --> ', l)

            ind_list = X[l].cpu().tolist()
            sent_list = [D.ind2sent(inds, l) for inds in ind_list]
            print('\t<< ', D.ind2sent(X[other[l]][0].cpu().tolist(), other[l]))
            print('\t== ', sent_list[0])
            ref_list += sent_list

            X_cur = X[other[l]].to(device)
            X_tr = wr_tr.translate(X_cur, other[l], l)

            ind_list = X_tr.cpu().tolist()
            sent_list = [D.ind2sent(inds, l) for inds in ind_list]
            print('\t>> ', sent_list[0])
            hyp_list += sent_list

        bleu = BLEU(ref_list, hyp_list)

        tmp_json['bleu'] = bleu
        tmp_json['iteration'] = it

        log.write(json.dumps(tmp_json, indent=4, sort_keys=True))
        log.write('\n' + ('-' * 50) + '\n')
        log.flush()

        print('BLEU: {:.2f}\n'.format(bleu))