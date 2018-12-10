from dictionaries import *

import torch
import torch.nn
import torch.optim
import torch.autograd


class Encoder(torch.nn.Module):
    def __init__(self, embeddings, hidden_size):
        super().__init__()
        self.emb = embeddings
        self.hidden_size = hidden_size
        for emb in embeddings.values():
            self.input_size = emb.embedding_dim

        self.gru = torch.nn.GRU(self.input_size, self.hidden_size, batch_first=True,
                          bidirectional=True)

    def step(self, input, hidden, from_lang):
        embedded = self.emb[from_lang](input)
        output, hidden = self.gru(embedded, hidden)

        return output, hidden

    def forward(self, ind_batch, nsteps, from_lang):
        encoder_hidden = torch.zeros((2, ind_batch.shape[0], self.hidden_size),
                                     device=ind_batch.device)

        embedded = self.emb[from_lang](ind_batch)
        encoder_outputs, encoder_hidden = self.gru(embedded, encoder_hidden)
        return encoder_outputs, encoder_hidden

class AttnLinear(torch.nn.Module):
    def __init__(self, input_size, state_size, inner_size=10):
        super().__init__()
        self.W = torch.nn.Linear(input_size + state_size, inner_size)
        self.v = torch.nn.Linear(inner_size, 1)

    def forward(self, input, hidden):
        expanded = hidden.expand(-1, input.shape[1], -1)
        return torch.relu(self.v(self.W(torch.cat((input, expanded), dim=2))))

class AttnNet(torch.nn.Module):
    def __init__(self, input_size, state_size, inner_size=10):
        super().__init__()
        self.v = torch.nn.Linear(inner_size, 1, bias=False)
        self.W = torch.nn.Linear(input_size, inner_size, bias=False)
        self.U = torch.nn.Linear(state_size, inner_size, bias=False)

    def forward(self, input, hidden):
        return torch.relu(self.v(self.W(input) + self.U(hidden)))


class AttnDecoder(torch.nn.Module):
    def __init__(self, embeddings, hidden_size, state_size, attn_size):
        super().__init__()
        self.emb = embeddings
        for emb in embeddings.values():
            self.emb_size = emb.embedding_dim
        self.hidden_size = hidden_size
        self.attn_size = attn_size
        self.state_size = state_size

        self.attn = AttnLinear(hidden_size, state_size, attn_size)
        self.gru = torch.nn.GRU(hidden_size + self.emb_size, state_size, batch_first=True)
        self.out = torch.nn.ModuleDict()
        for l, emb in embeddings.items():
            self.out[l] = torch.nn.Linear(state_size, emb.num_embeddings)

    def step(self, ind, hidden, encoder_outputs, to_lang):
        input = self.emb[to_lang](ind)
        attn_weights = torch.softmax(
            self.attn(encoder_outputs, hidden.transpose(0, 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.transpose(1, 2), encoder_outputs)
        gru_input = torch.cat((input, attn_applied), dim=2)
        output, hidden = self.gru(gru_input, hidden)
        output = self.out[to_lang](output)
        output = torch.log_softmax(output, dim=2)

        return output, hidden, attn_weights

    def forward(self, encoder_outputs, nsteps, to_lang):
        decoder_outputs = torch.zeros((encoder_outputs.shape[0], nsteps,
                                       self.emb[to_lang].num_embeddings),
                                      device=encoder_outputs.device)

        decoder_hidden = torch.zeros((1, encoder_outputs.shape[0], self.state_size),
                                     device=encoder_outputs.device)
        input = torch.full((encoder_outputs.shape[0], 1),
                           Vocabulary.get_dummy_ind('<SOS>'),
                           dtype=torch.long)
        input = input.to(encoder_outputs.device)
        for i in range(nsteps):
            decoder_output, decoder_hidden, attn_weights = self.step(input,
                                                                     decoder_hidden,
                                                                     encoder_outputs,
                                                                     to_lang)
            decoder_outputs[:, [i], :] += decoder_output
            _, input = decoder_output.topk(1, dim=2)
            input = input.view(encoder_outputs.shape[0], 1)

        return decoder_outputs, decoder_hidden


class Discriminator(torch.nn.Module):
    def __init__(self, hidden_size, hidden_layer_size, smooth_coef=1e-2):
        super().__init__()
        self.smooth_coef = smooth_coef
        self.hidden_size = hidden_size
        self.hidden_layer_size = hidden_layer_size

        self.hid = torch.nn.Linear(hidden_size, hidden_layer_size)
        self.hid2 = torch.nn.Linear(hidden_layer_size, hidden_layer_size)
        self.out = torch.nn.Linear(hidden_layer_size, 1)

    def forward(self, input):
        #         print('>>Discriminator')
        input = torch.relu(input)
        out = torch.relu(self.hid(input))
        out = torch.sigmoid(self.out(torch.relu(self.hid2(out)))).squeeze()

        # Add smoothing
        smooth = torch.eye(out.shape[-1], device=input.device) * self.smooth_coef
        smooth[0, 0] = 1
        for i in range(1, out.shape[-1]):
            smooth[:i, i] = (1 - self.smooth_coef) * smooth[:i, i - 1]
            # print('<<Discriminator')
        return out  # torch.mm(out, smooth)


class Wrapper(torch.nn.Module):
    def __init__(self, decoder_size, attn_size, discr_size, dataset):
        super().__init__()
        self.emb = torch.nn.ModuleDict()
        self.names = []
        for name, emb in dataset.emb.items():
            self.emb[name] = torch.nn.Embedding.from_pretrained(emb, freeze=False)
            self.names.append(name)
            self.hidden_size = self.emb[name].embedding_dim
        self.decoder_size = decoder_size
        self.attn_size = attn_size
        self.discr_size = discr_size

        self.enc = Encoder(self.emb, self.hidden_size)
        self.dec = AttnDecoder(self.emb, 2 * self.hidden_size, decoder_size, attn_size)
        self.discr = Discriminator(2 * self.hidden_size, discr_size)

    def encode(self, ind_batch, from_lang):
        return self.enc(ind_batch, ind_batch.shape[1], from_lang)

    def decode(self, encoder_outputs, output_len, to_lang):
        return self.dec(encoder_outputs, output_len, to_lang)

    def encode_decode(self, ind_batch, from_lang, to_lang, out_len=None):
        if out_len == None:
            out_len = ind_batch.shape[1]

        encoder_outputs, encoder_hidden = self.encode(ind_batch, from_lang)
        decoder_outputs, decoder_hidden = self.decode(encoder_outputs, out_len, to_lang)

        return encoder_outputs, decoder_outputs

    def discriminate(self, encoder_outputs):
        return self.discr(encoder_outputs)

class Translator:
    def __init__(self, wrapper, vocabs, max_len):
        self.wrapper = wrapper
        self.vocabs = vocabs
        self.max_len = max_len

    def translate_seq(self, seq, from_lang, to_lang):
        ind_seq = seq2ind(seq, self.vocabs[from_lang]) +\
                              [self.vocabs[from_lang].get_dummy_ind('<EOS>')]
        device = self.wrapper.enc.gru.weight_hh_l0.device
        ind_seq = torch.LongTensor(ind_seq).unsqueeze(0).to(device)
        with torch.no_grad():
            encoder_outputs, decoder_outputs =\
                           self.wrapper.encode_decode(ind_seq, from_lang,
                                                      to_lang, self.max_len)
        _, ind = decoder_outputs.squeeze().topk(1, dim=1)
        return ind2words(ind.squeeze().cpu().tolist(), self.vocabs[to_lang])

class IndTranslator:
    def __init__(self, wrapper, max_len):
        self.wrapper = wrapper
        self.max_len = max_len

    def translate(self, ind_batch, from_lang, to_lang):
        device = self.wrapper.enc.gru.weight_hh_l0.device
        with torch.no_grad():
            encoder_outputs, decoder_outputs =\
                           self.wrapper.encode_decode(ind_batch, from_lang,
                                                      to_lang, self.max_len)
        _, ind = decoder_outputs.squeeze().topk(1, dim=2)
        return ind.squeeze()

