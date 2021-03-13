import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import HighwayEncoder
from math import log


class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x):
        emb = self.embed(x)   # (batch_size, seq_len, embed_size)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)
        # hidden size = cout_word_ebedding size

        return emb


# https://github.com/jojonki/BiDAF/blob/master/layers/word_embedding.py
class WordEmbedding(nn.Module):
    '''
    In : (N, sentence_len)
    Out: (N, sentence_len, embd_size)
    '''
    def __init__(self, word_vectors, hidden_size, drop_prob):
        super(WordEmbedding, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(word_vectors)
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.drop_prob = drop_prob
        '''
        self.embedding = nn.Embedding(args.vocab_size_w, args.w_embd_size)
        if args.pre_embd_w is not None:
            self.embedding.weight = nn.Parameter(args.pre_embd_w, requires_grad=is_train_embd)
        '''
    def forward(self, x):
        emb = self.embedding(x)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)
        
        return emb

# https://github.com/jojonki/BiDAF/blob/master/layers/char_embedding.py
class CharEmbedding(nn.Module):
    '''
     In : (N, sentence_len, word_len, vocab_size_c)
     Out: (N, sentence_len, c_embd_size)
     '''
    def __init__(self, char_vectors, out_channels=100, filters=[[1,5]], drop_prob=0.2):
        super(CharEmbedding, self).__init__()
        
        vocab_size_c = char_vectors.size(0)
        self.drop_prob = drop_prob
        self.embd_size = char_vectors.size(1) # c -> idx
        self.embedding = nn.Embedding(vocab_size_c, self.embd_size)
        # nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, ...
        self.conv = nn.ModuleList([nn.Conv2d(1, out_channels, (f[0], f[1])) for f in filters])
        self.dropout = nn.Dropout(self.drop_prob)

    def forward(self, x):
        # x: (N, seq_len, word_len)
        input_shape = x.size()
        # bs = x.size(0)
        # seq_len = x.size(1)
        word_len = x.size(2)
        x = x.view(-1, word_len) # (N*seq_len, word_len)
        x = self.embedding(x) # (N*seq_len, word_len, c_embd_size)
        x = x.view(*input_shape, -1) # (N, seq_len, word_len, c_embd_size)
        x = x.sum(2) # (N, seq_len, c_embd_size)

        # CNN
        x = x.unsqueeze(1) # (N, Cin, seq_len, c_embd_size), insert Channnel-In dim
        # Conv2d
        #    Input : (N,Cin, Hin, Win )
        #    Output: (N,Cout,Hout,Wout)
        x = [F.relu(conv(x)) for conv in self.conv] # (N, Cout, seq_len, c_embd_size-filter_w+1). stride == 1
        # [(N,Cout,Hout,Wout) -> [(N,Cout,Hout*Wout)] * len(filter_heights)
        # [(N, seq_len, c_embd_size-filter_w+1, Cout)] * len(filter_heights)
        x = [xx.view((xx.size(0), xx.size(2), xx.size(3), xx.size(1))) for xx in x]
        # maxpool like
        # [(N, seq_len, Cout)] * len(filter_heights)
        x = [torch.sum(xx, 2) for xx in x]
        # (N, seq_len, Cout==word_embd_size)
        x = torch.cat(x, 1)
        x = self.dropout(x)
        # N, Cout, len(filter_heights)
        return x

https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionEmbedding(nn.Module):
    def __init__(self, d_model, drop_prob=0.1, max_len=5000):
        super(PositionEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

''' Old Position Embedding draft
class PositionEmbedding(nn.Module):
    
    Section 3.5
    PE(pos,2i) = sin(pos/10000^(2i/hidden_size))
    PE(pos,2i+1) = cos(pos/10000^(2i/hidden_size))
    pos = position and i = dimension
     In : (hidden_size, drop_probability)
     Out: (N, sentence_len, c_embd_size)
    
    def __init__(self, word_vectors, hidden_size, drop_prob=0.2):
        super(PositionEmbedding, self).__init__()
        
        #self.embd_size = char_vectors.size(1) # c -> idx
        #self.embedding = nn.Embedding(vocab_size_c, self.embd_size)
        # nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, ...
        #self.conv = nn.ModuleList([nn.Conv2d(1, out_channels, (f[0], f[1])) for f in filters])
        
        self.word_vectors = word_vectors
        self.hidden_size = hidden_size
        self.drop_prob = drop_prob
        self.dropout = nn.Dropout(self.drop_prob)
    
    def get_wavelength(self, dim):
        return 1/pow(10000, 2*dim/self.hidden_size)

    def forward(self, x):
        # x: (N, seq_len, word_len)
        #input_shape = x.size()

        # TODO SIN & COSINE Embeddings
        # for each dimension in text
            # for each position in dimension
                # if position % 2 = 0:
                    # sin(pos * get_wavelength(dimension))
                # else:
                    # cos(pos * get_wavelength(dimension))
        
        #What is N in this case?
        #Can I loop trhough words at this point or wouold I need to modify setup.py
        
        return x
'''