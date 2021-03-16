import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from math import sqrt, sin, cos
from util import masked_softmax


#Modified version of multi-head self-attention from assign 5
class MultiHeadAttention(nn.Module):
    """ 
    A vanilla multi-head masked self-attention layer with a projection at the end.
    I believe I could have just used torch.nn.MultiheadAttention but their documentation
    is all but absent and code ugly so I don't trust it, rolling my own here.
    """

    def __init__(self, d_model, num_heads, attn_drop_prob=0.1, resid_drop_prop=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.d_k = d_model // num_heads #key dimensions = model_dims/num_heads
        self.num_heads = num_heads

        # key, query, value projections for all heads
        self.key = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(attn_drop_prob)
        self.resid_dropout = nn.Dropout(resid_drop_prop)

        # output projection
        self.proj = nn.Linear(d_model, d_model)

        # Set Num Heads
        self.num_heads = num_heads

    def scaledDotProductAttention(self, q, k, v, T, mask=None, drop_prob=0.1):
        att = (q @ k.transpose(-2, -1)) * (1.0 / sqrt(self.d_k))

        if mask is not None:
            pass
            #TODO apply mask
            #att.masked_fill(mask[:,:,:T,:T] == 0, -1e10) 

            #mask = mask.unsqueeze(1)
            #att = att.masked_fill(mask == 0, -1e10)

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        return att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)


    def forward(self, x, mask=None, layer_past=None):
        B, T, _ = x.size() # C = self.d_model

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2) # (B, nh, T, hs)

        y = self.scaledDotProductAttention(q, k, v, T, mask)
        y = y.transpose(1, 2).contiguous().view(B, T, self.d_model) # re-assemble all head outputs side by side

        # output projection
        return self.resid_dropout(self.proj(y))

#https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec#3fa3
class FeedForward(nn.Module):
    def __init__(self, embd_size, d_ff=2048, drop_prob=0.1):
        super().__init__() 
        self.linear_1 = nn.Linear(embd_size, d_ff)
        self.dropout = nn.Dropout(drop_prob)
        self.linear_2 = nn.Linear(d_ff, embd_size)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class TransformerEncoder(nn.Module):
    ''' 
        Per Vaswani et al. Attention is all you need.
        Transformer encoder constructs num_layers identical TranformerEncoderLayers
        The forward function applies each layer to an input
    '''
    def __init__(self, d_model, num_layers, num_heads, norm=None, drop_prob=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.layers = self.get_clones(TransformerEncoderLayer(d_model=d_model, num_heads=num_heads), num_layers)
        self.norm = norm

    #Convenience function to allow us to generate num_layers identical copies of our encoder layer
    def get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    
    # Apply each attention layer
    # TODO still confused on mask
    def forward(self, x, mask):
        for i in range(self.num_layers):
            x = self.layers[i](x, mask)
        return self.norm(x)

#https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec#3fa3
class TransformerEncoderLayer(nn.Module):
    ''' 
    Per Vaswani et al. Attention is all you need.
    Each encoder layer consists of two sublayers
        - MultiHeadAttention Sublayer
        - FeedForward Sublayer
    The forward function applies each sublayer with a dropout for each layer
    '''
    def __init__(self, d_model, num_heads, dropout = 0.1, norm=None):
        super().__init__()
        self.norm = norm
        #self.norm_1 = norm
        #self.norm_2 = norm
        self.attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.ff = FeedForward(d_model)
        
        self.dropout = nn.Dropout(dropout)
        #self.dropout_1 = nn.Dropout(dropout)
        #self.dropout_2 = nn.Dropout(dropout)

    def normalize(self, x):
        return self.norm(x) if self.norm is not None else x

    def forward(self, x, mask):
        x = self.normalize(x)
        x = x + self.dropout(self.attn(x, mask))
        x = self.normalize(x)
        x = x + self.dropout(self.attn(x))
        return x

class TransformerDecoder(nn.Module):
    ''' 
        Per Vaswani et al. Attention is all you need.
        Transformer encoder constructs num_layers identical TranformerDecoderLayers
        The forward function applies each layer to an input

        *Note
        Similair too Transformer Encoder but has following changes
            - Uses DecoderLayer rather than Encoder layer
            - Must add another level of masking
            - Includes encoder_outputs as well as the emebeddings in the forward function
    '''
    def __init__(self, d_model, num_layers, num_heads):
        super().__init__()
        self.num_layers = num_layers
        self.layers = self.get_clones(TransformerDecoderLayer(d_model=d_model, num_heads=num_heads), num_layers)
        self.norm = Norm(d_model)

    #Convenience function to allow us to generate num_layers identical copies of our encoder layer
    def get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    

    def forward(self, x, encoder_outputs, src_mask, trg_mask):
        for i in range(self.num_layers):
            x = self.layers[i](x, encoder_outputs, src_mask, trg_mask)
        return self.norm(x)

class TransformerDecoderLayer(nn.Module):
    ''' 
    Per Vaswani et al. Attention is all you need.
    Each decoder layer consists of three sublayers
        - MultiHeadAttention Sublayer
        - FeedForward Sublayer
        - Additional MultiHeadAttention Sublayer that attends to encoder stack

    The forward function applies each sublayer with a dropout for each layer
    '''
    def __init__(self, d_model, num_heads, dropout = 0.1):
        super().__init__()
        self.norm_1 = Norm(embd_size)
        self.norm_2 = Norm(embd_size)
        self.attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.ff = FeedForward(embd_size)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

    def forward(self, x, encoder_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, encoder_outputs, encoder_outputs,
        src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x

# Positional encodings from Attention is All You Need paper - Applied in each sub-layer of the encoder
class PositionalEncoding(nn.Module):
    '''
    Input: Embd Size, Drop Probability, Max Sequence Length
    Apply formula from Attention is All You Need:
        PE(pos, 2i) = sin(pos/10000^(2i/embd_size))
        PE(pos, 2i + 1) = cos(pos/10000^(2i/embd_size))
        pos = position, i = dimension
    Output: Sinusoidal Positional Encodings
    '''

    def __init__(self, d_model, drop_prob=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(drop_prob)
        self.pe = self.createPositionalEncodings(d_model, max_len)
        #self.register_buffer('pe', self.pe)

    def createPositionalEncodings(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)

        for pos in range(max_len):
            for i in range(d_model):
                power = 2 * i / d_model
                pe[pos, i] = sin(pos / (10000 ** (power))) if i % 2 == 0 else cos(pos / (10000 ** (power)))
        
        return pe.unsqueeze(0).transpose(0, 1)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

#https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec#3fa3
class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm