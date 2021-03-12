"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers.encoder as encoder
import layers.embedding as embedding
import layers.bidaf as bidaf
import torch
import torch.nn as nn


from datetime import datetime


class SelfAttModel(nn.Module):
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.1):
        super(SelfAttModel, self).__init__()

        self.word_embd = embedding.WordEmbedding(word_vectors=word_vectors,
                                                hidden_size=hidden_size,
                                                drop_prob=drop_prob)
        self.char_embd = embedding.CharEmbedding(char_vectors=char_vectors)
        self.pos_embd = embedding.PositionEmbedding(word_vectors=word_vectors,
                                                hidden_size=hidden_size, 
                                                drop_prob=drop_prob)
        self.hwy = encoder.HighwayEncoder(2, 2*hidden_size)

        #self.enc = encoder.SelfAttEncoder(input_size=2*hidden_size,
        #                             hidden_size=hidden_size,
        #                             num_layers=1,
        #                             drop_prob=drop_prob)

        #self.att = bidaf.BiDAFAttention(hidden_size=2*hidden_size,
        #                                 drop_prob=drop_prob)

        #self.mod = encoder.RNNEncoder(input_size=8 * hidden_size,
        #                             hidden_size=hidden_size,
        #                             num_layers=2,
        #                             drop_prob=drop_prob)

        #self.out = bidaf.BiDAFOutput(hidden_size=hidden_size,
        #                              drop_prob=drop_prob)

    def build_contextual_encoding(self, x_w, x_c, w_len, c_len):
        char_embd = self.char_embd(x_c)

        word_embd = self.word_embd(x_w)

        pos_embd = self.pos_embd(x_c)

        #print("Char emb shape:", char_embd.shape, 
        #print("word emb shape:", word_embd.shape)
        #print("Pos embd shape:", pos_embd.shape)

        #print(f'char_embd shape{char_embd.shape}')
        #print(f'word_embd shape{word_embd.shape}')

        #embd = torch.cat((char_embd, word_embd), 2)
        embd = torch.cat((char_embd, word_embd, pos_embd), 2)

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

   
    def __init__(self, word_vectors, char_vectors, hidden_size, params):
        super(BiDAFplus, self).__init__()

        self.model_name = 'BiDAFplus'

        self.word_embd = embedding.WordEmbedding(word_vectors=word_vectors,
                                                 hidden_size=hidden_size,
                                                 drop_prob=params['drop_prob'])


        self.char_embd = embedding.CharEmbedding(char_vectors=char_vectors,
                                                 drop_prob=params['drop_prob'],
                                                 filters=params['filters'],
                                                 out_channels=params['out_channels']
                                                 )

        # input size = CharEmbedding Size

        self.d = 2 * hidden_size # char_embedding_size + word_embedding_size

        self.hwy = encoder.HighwayEncoder(params['hwy_layers'], self.d)

        
        # select the phrase encoder
        self.phrase_encoder = params['phrase_encoder']

        if self.phrase_encoder == 'lstm':
            encoder_fn = encoder.RNNEncoder
        elif self.phrase_encoder == 'gru':
            encoder_fn = encoder.GRUEncoder

        self.enc = encoder_fn(input_size=self.d,
                              hidden_size=self.d,
                              num_layers=params['encoder_layers'],
                              drop_prob=params['drop_prob'])
        



        self.att = bidaf.BiDAFAttention(hidden_size=2*self.d,
                                        drop_prob=params['drop_prob'])

        self.mod = encoder.RNNEncoder(input_size=8 * self.d,
                                      hidden_size=self.d,
                                      num_layers=params['model_layers'],
                                      drop_prob=params['drop_prob'])

        self.out = bidaf.BiDAFOutput(hidden_size=self.d,
                                     drop_prob=params['drop_prob'])

    def build_contextual_encoding(self, x_w, x_c, w_len, c_len):

        # embed_size = 100
        char_embd = self.char_embd(x_c)

        word_embd = self.word_embd(x_w)

        print("{:<6} {:<32} ({:<6} {:<6} {:<6}) ({:<6} {:<6} {:<6})".format('.1', 'Char Embedding Layer',
                                                      'N', 'seq_len', 'embd_size', *char_embd.shape))
        print("{:<6} {:<32} ({:<6} {:<6} {:<6}) ({:<6} {:<6} {:<6})".format('.2', 'Word Embedding Layer',
                                                      'N', 'seq_len', 'embd_size', *word_embd.shape))
        #print(f'char_embd shape{char_embd.shape}')
        #print(f'word_embd shape{word_embd.shape}')

        embd = torch.cat((char_embd, word_embd), 2)

        #print(f'cat_embd shape {embd.shape}')

        embd = self.hwy(embd)
        

        #print(f'hwy_embd shape {embd.shape}')
        lens = w_len #+ c_len
        #print(f'lens: {w_len} c_len: {c_len}')
        print("{:<6} {:<32} ({:<6} {:<6} {:<6}) ({:<6} {:<6} {:<6})".format('.2b', 'Highway Out',
                                                      'N', 'seq_len', 'd', *embd.shape))

        if self.phrase_encoder == 'lstm':
            encoding = self.enc(embd, lens)

        elif self.phrase_encoder == 'gru':
            encoding = self.enc(embd, lens)
        else:
            raise Exception('Invalid phrase_encoder')
        

        


        return encoding



    # ctx_w, ctx_c, query_w, query_c
    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        print(f'Embed_size: {cw_idxs.shape(2)}. d={self.d}')
        print(f'Context sequence Length: T={len(cw_idxs)}')
        print(f'Query sequence Length: J={len(qw_idxs)}')
        cw_mask = torch.zeros_like(cw_idxs) != cw_idxs
        qw_mask = torch.zeros_like(qw_idxs) != qw_idxs
        cw_len, qw_len = cw_mask.sum(-1), qw_mask.sum(-1)

        #print(f'CW mask shape {cw_mask.shape}')
        #print(f'QW mask shape {qw_mask.shape}')
        cc_mask = torch.zeros_like(cc_idxs) != cc_idxs
        qc_mask = torch.zeros_like(qc_idxs) != qc_idxs
        cc_len, qc_len = cc_mask.sum(-1), qc_mask.sum(-1)

        #print(f'CC mask shape {cc_mask.shape}')
        #print(f'QC mask shape {qc_mask.shape}')
     
        c_enc = self.build_contextual_encoding(cw_idxs, cc_idxs, cw_len, cc_len)
        q_enc = self.build_contextual_encoding(qw_idxs, qc_idxs, qw_len, qc_len)
        print("{:<6} {:<48} ({:<6} {:<6} {:<6}) ({:<6} {:<6} {:<6})".format('.3a', 'Context Contextual Embedding Layer',
                                                      'N', 'T', '2d', *c_enc.shape))

        print("{:<6} {:<48} ({:<6} {:<6} {:<6}) ({:<6} {:<6} {:<6})".format('.3b', 'Context Contextual Embedding Layer',
                                                      'N', 'J', '2d', *q_enc.shape))


        # hs = 2x
        #print(f'context_encoder {c_enc.shape}')
        #print(f'query_encoder {q_enc.shape}')
        #print(f'Expecting batch, cw_len, 800')
        # need to figure this out
        att = self.att(c_enc, q_enc,
                       cw_mask, qw_mask)    # (batch_size, c_len, 8 * d)
        
        #print(f'att shape {att.shape}')
        mod = self.mod(att, cw_len)        # (batch_size, c_len, 2 * d)
        
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
    def __init__(self, word_vectors, hidden_size, drop_prob=0.2):
        super(BiDAF, self).__init__()

        self.model_name = 'BiDAF'


        self.emb = embedding.Embedding(word_vectors=word_vectors,
                                       hidden_size=hidden_size,
                                       drop_prob=drop_prob)


        self.enc = encoder.GRUEncoder(input_size=hidden_size,
                                      hidden_size=hidden_size,
                                      num_layers=1,
                                      drop_prob=drop_prob)
        '''
        self.gru = encoder.GRUEncoder(input_size=hidden_size,
                                       hidden_size=hidden_size,
                                       num_layers=1,
                                       drop_prob=drop_prob)
        '''
        self.att = bidaf.BiDAFAttention(hidden_size=2*hidden_size,
                                        drop_prob=drop_prob)

        self.mod = encoder.GRUEncoder(input_size=8 * hidden_size,
                                      hidden_size=hidden_size,
                                      num_layers=2,
                                      drop_prob=drop_prob)
        '''
        self.gru_mod = encoder.GRUEncoder(input_size=8 * hidden_size,
                                      hidden_size=hidden_size,
                                      num_layers=2,
                                      drop_prob=drop_prob)
        '''
        self.out = bidaf.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)



    def forward(self, cw_idxs, qw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)


        #lstm_start_time = datetime.now()

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)
        '''
        gru_start_time = datetime.now()

        gru_c_enc = self.enc(c_emb, c_len)
        gru_q_enc = self.enc(q_emb, q_len)

        gru_stop_time = datetime.now()


        lstm_elapsed_time = "{}".format((gru_start_time-lstm_start_time).microseconds * .001)
        gru_elapsed_time = "{}".format((gru_stop_time-gru_start_time).microseconds * .001)
        print('='*20)
        print('{:<8} {:<16} {:<16} {:<8}'.format('Encoder', 'C_shape', 'Q_shape', 'Time'))
        print('{:<8} {:<16} {:<16} {:<8}'.format('lstm', str(c_enc.shape)[10:], str(q_enc.shape)[10:], lstm_elapsed_time))
        print('{:<8} {:<16} {:<16} {:<8}'.format('gru', str(gru_c_enc.shape)[10:], str(gru_q_enc.shape)[10:], gru_elapsed_time))
        print('-'*20)
        '''
        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        #lstm_start_time = datetime.now()
        
        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)
        '''
        gru_start_time = datetime.now()


        gru_mod = self.mod(att, c_len)

        gru_stop_time = datetime.now()

        lstm_elapsed_time = "{}".format((gru_start_time-lstm_start_time).microseconds * .001)
        gru_elapsed_time = "{}".format((gru_stop_time-gru_start_time).microseconds * .001)

        print('{:<8} {:<16} {:<8}'.format('Model', 'Shape', 'Time'))
        print('{:<8} {:<16} {:<8}'.format('lstm', str(mod.shape)[10:], lstm_elapsed_time))
        print('{:<8} {:<16} {:<8}'.format('gru', str(gru_mod.shape)[10:], gru_elapsed_time))
        print('='*20)
        '''
        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out
