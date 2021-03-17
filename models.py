"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers.encoder as encoder
import layers.embedding as embedding
import layers.transformer as transformer
import layers.bidaf as bidaf
import torch
import torch.nn as nn

#
# Note couldn't get this to work using custom EncoderLayer and TransformerEncoderClass
#from torch.nn import TransformerEncoder, TransformerEncoderLayer
#  
from math import sqrt
from datetime import datetime


class TransformerModel(nn.Module):
    def __init__(self, word_vectors, char_vectors, params):
        super(TransformerModel, self).__init__()

        self.model_type = "Transformer"
        self.params = params
        self.hidden_size = params.hidden_size
        self.drop_prob = params.drop_prob

        # 1. Embedding layer
        embd_params = self.params.embedding_layer
        self.embd = embedding.Embedding(word_vectors=word_vectors,
                                        char_vectors=char_vectors,
                                        hidden_size=self.hidden_size,
                                        drop_prob=self.drop_prob,
                                        params=embd_params)

        embedding_size = 2 * hidden_size
        self.pos_enc = transformer.PositionalEncoder(embedding_size=embedding_size, 
                                                  drop_prob=self.drop_prob)

        # 2. Encoding layer
        enc_params = self.params.encoder_layer

        self.enc = transformer.TransformerBlock(d_model=self.d_model,
                                            num_layers=enc_params["num_layers"],
                                            num_heads=enc_params["num_heads"],
                                            n_conv_blocks=1,
                                            drop_prob=self.drop_prob)


        # 3. Attention layer
        self.att = bidaf.BiDAFAttention(hidden_size=2*self.hidden_size,
                                        drop_prob=self.drop_prob)

        att_out_size = 4 * self.hidden_size
        # 3. Decoder layer
        dec_params = self.params.decoder_layer
        self.mod = encoder.TransformerDecoder(input_dim=att_out_size,
                                              params=dec_params)

        mod_out_size = self.hidden_size
        # output layer
        self.out = Output(input_dim=self.hidden_size)

    # https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    #def generate_square_subsequent_mask(self, sz):
    #    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    #    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    #    return mask

    def forward(self, ctx_char_idxs, query_char_idxs, ctx_word_idxs, query_word_idxs):
        # masks
        ctx_mask = torch.zeros_like(ctx_word_idxs) != ctx_word_idxs
        query_mask = torch.zeros_like(query_word_idxs) != query_word_idxs
        ctx_len, query_len = ctx_mask.sum(-1), query_mask.sum(-1)

        # 1. Embedding layer
        # context and query embeddings
        ctx_emb = self.embd(ctx_char_idxs, ctx_word_idxs)
        query_emb = self.embd(query_char_idxs, query_word_idxs)
        
        ctx_emb = self.pos_enc(ctx_emb)
        query_emb = self.pos_enc(query_emb)

        # 2. encoding layer
        ctx_enc = self.enc(ctx_emb)
        query_enc = self.enc(query_emb)

        print("Successful encoding")

        # 3. Attention layer
        att = self.att(ctx_enc, query_enc, ctx_mask, query_mask)

        print("Successful attention")

        # 4. decoder layer
        mod = self.mod(att)

        out = self.out(mod)
        print("Successful decodings")
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
    def __init__(self, word_vectors, char_vectors, hidden_size, params):
        super(BiDAF, self).__init__()

        self.hidden_size = hidden_size
        self.drop_prob = params.drop_prob
        self.params = params

        self.d = hidden_size * 2
        

        # 1. Embedding layer
        embd_params = self.params.embedding_layer
        self.embd = embedding.Embedding(word_vectors=word_vectors,
                                        char_vectors=char_vectors,
                                        params=embd_params) # embedding_size and
                                                            # drop_prob in params
        self.embd_out_size = 2 * embd_params['embedding_size']
        # shpape: (batch_size, seq_len, embd_out_size)
        # 2. Encoding layer
        enc_params = self.params.encoder_layer

        if enc_params['rnn'] == 'lstm':
            enc = encoder.LSTMEncoder
        elif enc_params['rnn'] == 'GRU':
            enc = encoder.GRUEncoder
        

        self.enc = enc(input_size=self.embd_out_size,
                       hidden_size=enc_params['hidden_size'], # self.d
                       num_layers=enc_params['layers'], # 1
                       drop_prob=self.drop_prob) # 
        
        self.enc_out_size = 2 * enc_params['hidden_size']


        # 3. Attention layer
        # maybe params?
        att_params = self.params.attention_layer
        self.att = bidaf.BiDAFAttention(hidden_size=self.enc_out_size,
                                        drop_prob=att_params['drop_prob'])

        self.att_out_size = 4 * self.enc_out_size
        # 4. Modeling layer
        mod_params = self.params.modeling_layer

        if mod_params['rnn'] == 'lstm':
            mod = encoder.LSTMEncoder
        elif mod_params['rnn'] == 'GRU':
            mod = encoder.GRUEncoder


        self.mod = mod(input_size=self.att_out_size,
                       hidden_size=mod_params['hidden_size'], # hidden_size
                       num_layers=mod_params['layers'], # 2
                       drop_prob=mod_params['drop_prob'])

        self.mod_out_size = 2 * mod_params['hidden_size']

        # 5. Output layer
        out_params = self.params.output_layer
        self.out = bidaf.BiDAFOutput(hidden_size=self.embd_out_size,
                                     drop_prob=self.drop_prob)



    def forward(self, ctx_char_idxs, query_char_idxs, ctx_word_idxs, query_word_idxs):
        # masks
        ctx_mask = torch.zeros_like(ctx_word_idxs) != ctx_word_idxs
        query_mask = torch.zeros_like(query_word_idxs) != query_word_idxs
        ctx_len, query_len = ctx_mask.sum(-1), query_mask.sum(-1)

        # 1. Embedding layer
        # context and query embeddings
        ctx_emb = self.embd(ctx_char_idxs, ctx_word_idxs)
        query_emb = self.embd(query_char_idxs, query_word_idxs)

        # 2. encoding layer
        ctx_enc = self.enc(ctx_emb, ctx_len)
        query_enc = self.enc(query_emb, query_len)


        # 3. Attention layer
        att = self.att(ctx_enc, query_enc, ctx_mask, query_mask)

        # 4. Modeling layer
        mod = self.mod(att, ctx_len)

        # 5. output layer
        out = self.out(att, mod, ctx_mask)

        return out


class BiDAFbase(nn.Module):
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
        super(BiDAFbase, self).__init__()



        self.emb = embedding.defEmbedding(word_vectors=word_vectors,
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



    def forward(self, ctx_w_idx, query_w_idxs):
        c_mask = torch.zeros_like(ctx_w_idx) != ctx_w_idx
        q_mask = torch.zeros_like(query_w_idxs) != query_w_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(ctx_w_idx)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(query_w_idxs)         # (batch_size, q_len, hidden_size)


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


class BiDAFplus(nn.Module):
    # word_vectors -> word_embeddings
    # char_vectors -> char_embeddings

    # word_embeddings + char_embeddings -> encoder / (phrase embeddings)

   
    def __init__(self, word_vectors, char_vectors, hidden_size, params):
        super(BiDAFplus, self).__init__()

        self.model_name = 'BiDAFplus'

        self.word_embd = embedding.WordEmbedding(word_vectors=word_vectors,
                                                 hidden_size=hidden_size,
                                                 drop_prob=params.drop_prob)#['drop_prob'])


        self.char_embd = embedding.CharEmbedding(char_vectors=char_vectors,
                                                 drop_prob=params.drop_prob)#,#['drop_prob'],
                                                 #filters=params.filters,#['filters'],
                                                 #out_channels=params.out_channels#['out_channels']
                                                 #)

        # input size = CharEmbedding Size

        self.d = 2 * hidden_size # char_embedding_size + word_embedding_size

        self.hwy = encoder.HighwayEncoder(2, self.d)#params.hwy_layers, self.d)

        
        # select the phrase encoder
        self.phrase_encoder = "lstm"#params.phrase_encoder#['phrase_encoder']

        if self.phrase_encoder == 'lstm':
            encoder_fn = encoder.LSTMEncoder
        elif self.phrase_encoder == 'gru':
            encoder_fn = encoder.GRUEncoder

        self.enc = encoder_fn(input_size=self.d,
                              hidden_size=self.d, #self.d,
                              num_layers=params.encoder_layer["layers"],#['encoder_layers'],
                              drop_prob=params.drop_prob)#['drop_prob'])
        



        self.att = bidaf.BiDAFAttention(hidden_size=2*self.d,
                                        drop_prob=params.drop_prob)#['drop_prob'])

        self.mod = encoder.LSTMEncoder(input_size=8 * self.d,
                                      hidden_size=self.d,
                                      num_layers=params.modeling_layer["layers"],#['model_layers'],
                                      drop_prob=params.drop_prob)#['drop_prob'])

        self.out = bidaf.BiDAFOutput(hidden_size=self.d,
                                     drop_prob=params.drop_prob)# ['drop_prob'])

    def build_contextual_encoding(self, x_w, x_c, w_len, c_len):

        # embed_size = 100
        char_embd = self.char_embd(x_c)

        word_embd = self.word_embd(x_w)

        '''

        print("{:<6} {:<32} ({:<6} {:<6} {:<6}) ({:<6} {:<6} {:<6})".format('.1', 'Char Embedding Layer',
                                                      'N', 'seq_len', 'embd_size', *char_embd.shape))
        print("{:<6} {:<32} ({:<6} {:<6} {:<6}) ({:<6} {:<6} {:<6})".format('.2', 'Word Embedding Layer',
                                                      'N', 'seq_len', 'embd_size', *word_embd.shape))
        '''
        #print(f'char_embd shape{char_embd.shape}')
        #print(f'word_embd shape{word_embd.shape}')

        embd = torch.cat((char_embd, word_embd), 2)

        #print(f'cat_embd shape {embd.shape}')

        embd = self.hwy(embd)
        

        #print(f'hwy_embd shape {embd.shape}')
        lens = w_len #+ c_len
        #print(f'lens: {w_len} c_len: {c_len}')
        '''
        print("{:<6} {:<32} ({:<6} {:<6} {:<6}) ({:<6} {:<6} {:<6})".format('.2b', 'Highway Out',
                                                      'N', 'seq_len', 'd', *embd.shape))
        '''
        if self.phrase_encoder == 'lstm':
            encoding = self.enc(embd, lens)

        elif self.phrase_encoder == 'gru':
            encoding = self.enc(embd, lens)
        else:
            raise Exception('Invalid phrase_encoder')
        

        


        return encoding



    # ctx_w, ctx_c, query_w, query_c
    def forward(self, ctx_w_idx, query_w_idxs, ctx_char_idxs, query_char_idxs):
        print(f'd={self.d}')
        print(f'Context sequence Length: T={ctx_w_idx.size(1)}')
        print(f'Query sequence Length: J={query_w_idxs.size(1)}')
        cw_mask = torch.zeros_like(ctx_w_idx) != ctx_w_idx
        qw_mask = torch.zeros_like(query_w_idxs) != query_w_idxs
        cw_len, qw_len = cw_mask.sum(-1), qw_mask.sum(-1)

        #print(f'CW mask shape {cw_mask.shape}')
        #print(f'QW mask shape {qw_mask.shape}')
        cc_mask = torch.zeros_like(ctx_char_idxs) != ctx_char_idxs
        qc_mask = torch.zeros_like(query_char_idxs) != query_char_idxs
        cc_len, qc_len = cc_mask.sum(-1), qc_mask.sum(-1)

        print(f'cw_len={cw_len}')
        print(f'qw_len={qw_len}')
        print(f'cc_len={cc_len}')
        print(f'qc_len={qc_len}')
        #print(f'CC mask shape {cc_mask.shape}')
        #print(f'QC mask shape {qc_mask.shape}')
     
        c_enc = self.build_contextual_encoding(ctx_w_idx, ctx_char_idxs, cw_len, cc_len)
        q_enc = self.build_contextual_encoding(query_w_idxs, query_char_idxs, qw_len, qc_len)
        print("{:<6} {:<48} ({:<6} {:<6} {:<6}) ({:<6} {:<6} {:<6})".format('.3a', 'Context Contextual Embedding Layer',
                                                      'N', 'T', '2d', *c_enc.shape))

        print("{:<6} {:<48} ({:<6} {:<6} {:<6}) ({:<6} {:<6} {:<6})".format('.3b', 'Query Contextual Embedding Layer',
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
