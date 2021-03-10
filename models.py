"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers.encoder as encoder
import layers.embedding as embedding
import layers.bidaf as bidaf
import torch
import torch.nn as nn


class SelfAttModel(nn.module):
    super(BiDAFself, self).__init__()

    self.word_embd = embedding.WordEmbedding(word_vectors=word_vectors,
                                             hidden_size=hidden_size,
                                             drop_prob=drop_prob)
    self.char_embd = embedding.CharEmbedding(char_vectors=char_vectors)
    self.hwy = encoder.HighwayEncoder(2, 2*hidden_size)

    self.enc = encoder.SelfAttEncoder(input_size=2*hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)
    
    self.att = bidaf.BiDAFAttention(hidden_size=2*hidden_size,
                                         drop_prob=drop_prob)

    self.mod = encoder.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

    self.out = bidaf.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

        def build_contextual_encoding(self, x_w, x_c, w_len, c_len):
        char_embd = self.char_embd(x_c)

        word_embd = self.word_embd(x_w)

        #print(f'char_embd shape{char_embd.shape}')
        #print(f'word_embd shape{word_embd.shape}')

        embd = torch.cat((char_embd, word_embd), 2)

        #print(f'cat_embd shape {embd.shape}')

        embd = self.hwy(embd)
        #print(f'hwy_embd shape {embd.shape}')
        lens = w_len #+ c_len
        #print(f'lens: {w_len} c_len: {c_len}')



        encoding = self.enc(embd, lens)
        
        return encoding



    # ctx_w, ctx_c, query_w, query_c
    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        cw_mask = torch.zeros_like(cw_idxs) != cw_idxs
        qw_mask = torch.zeros_like(qw_idxs) != qw_idxs
        cw_len, qw_len = cw_mask.sum(-1), qw_mask.sum(-1)

        cc_mask = torch.zeros_like(cc_idxs) != cc_idxs
        qc_mask = torch.zeros_like(qc_idxs) != qc_idxs
        cc_len, qc_len = cc_mask.sum(-1), qc_mask.sum(-1)
     
        c_enc = self.build_contextual_encoding(cw_idxs, cc_idxs, cw_len, cc_len)
        q_enc = self.build_contextual_encoding(qw_idxs, qc_idxs, qw_len, qc_len)

        # hs = 2x
        #print(f'context_encoder {c_enc.shape}')
        #print(f'query_encoder {q_enc.shape}')
        #print(f'Expecting batch, cw_len, 800')
        att = self.att(c_enc, q_enc,
                       cw_mask, qw_mask)    # (batch_size, c_len, 8 * hidden_size)
        
        #print(f'att shape {att.shape}')
        mod = self.mod(att, cw_len)        # (batch_size, c_len, 2 * hidden_size)
        
        #print(f'mod shape {mod.shape}')
        out = self.out(att, mod, cw_mask)  # 2 tensors, each (batch_size, c_len)

        return out

class BiDAFplus(nn.Module):
    # word_vectors -> word_embeddings
    # char_vectors -> char_embeddings

    # word_embeddings + char_embeddings -> encoder / (phrase embeddings)

   
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.):
        super(BiDAFplus, self).__init__()

        self.word_embd = embedding.WordEmbedding(word_vectors=word_vectors,
                                             hidden_size=hidden_size,
                                             drop_prob=drop_prob)


        self.char_embd = embedding.CharEmbedding(char_vectors=char_vectors)

        # input size = CharEmbedding Size

        self.hwy = encoder.HighwayEncoder(2, 2*hidden_size)

        self.enc = encoder.RNNEncoder(input_size=2*hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = bidaf.BiDAFAttention(hidden_size=2*hidden_size,
                                         drop_prob=drop_prob)

        self.mod = encoder.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = bidaf.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def build_contextual_encoding(self, x_w, x_c, w_len, c_len):
        char_embd = self.char_embd(x_c)

        word_embd = self.word_embd(x_w)

        #print(f'char_embd shape{char_embd.shape}')
        #print(f'word_embd shape{word_embd.shape}')

        embd = torch.cat((char_embd, word_embd), 2)

        #print(f'cat_embd shape {embd.shape}')

        embd = self.hwy(embd)
        #print(f'hwy_embd shape {embd.shape}')
        lens = w_len #+ c_len
        #print(f'lens: {w_len} c_len: {c_len}')



        encoding = self.enc(embd, lens)
        
        return encoding



    # ctx_w, ctx_c, query_w, query_c
    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        cw_mask = torch.zeros_like(cw_idxs) != cw_idxs
        qw_mask = torch.zeros_like(qw_idxs) != qw_idxs
        cw_len, qw_len = cw_mask.sum(-1), qw_mask.sum(-1)

        cc_mask = torch.zeros_like(cc_idxs) != cc_idxs
        qc_mask = torch.zeros_like(qc_idxs) != qc_idxs
        cc_len, qc_len = cc_mask.sum(-1), qc_mask.sum(-1)
     
        c_enc = self.build_contextual_encoding(cw_idxs, cc_idxs, cw_len, cc_len)
        q_enc = self.build_contextual_encoding(qw_idxs, qc_idxs, qw_len, qc_len)

        # hs = 2x
        #print(f'context_encoder {c_enc.shape}')
        #print(f'query_encoder {q_enc.shape}')
        #print(f'Expecting batch, cw_len, 800')
        att = self.att(c_enc, q_enc,
                       cw_mask, qw_mask)    # (batch_size, c_len, 8 * hidden_size)
        
        #print(f'att shape {att.shape}')
        mod = self.mod(att, cw_len)        # (batch_size, c_len, 2 * hidden_size)
        
        #print(f'mod shape {mod.shape}')
        out = self.out(att, mod, cw_mask)  # 2 tensors, each (batch_size, c_len)

        return out


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.emb = embedding.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)


        self.enc = encoder.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = bidaf.BiDAFAttention(hidden_size=2*hidden_size,
                                         drop_prob=drop_prob)

        self.mod = encoder.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = bidaf.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)



    def forward(self, cw_idxs, qw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out
