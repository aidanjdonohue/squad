import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from math import sqrt, sin, cos
from util import masked_softmax

BLOCK_SIZE = 5000


class TransformerOut(nn.Module):

    def __init__(self, input_size, output_size, drop_prob=0.1):

        self.start_dense = Linear(input_size, output_size)
        self.end_dense = Linear(input_size, output_size)

    def forward(self, mod):

        m1, m2, m3 = mod

        # start predictor
        # m1, m2




        # end predictor
        # m1, m3



class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    I believe I could have just used torch.nn.MultiheadAttention but their documentation
    is all but absent and code ugly so I don't trust it, rolling my own here.
    """

    def __init__(self, input_size, output_size, drop_prob=0.1, max_len=BLOCK_SIZE):
        super(SelfAttention, self).__init__()
        # key, query, value projections for all heads
        self.key = nn.Linear(input_size, output_size)
        self.query = nn.Linear(input_size, output_size)
        self.value = nn.Linear(input_size, output_size)
        # regularization
        self.attn_drop = nn.Dropout(drop_prob)
        self.resid_drop = nn.Dropout(drop_prob)
        # output projection
        self.proj = nn.Linear(input_size, output_size)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(max_len, max_len))
                                     .view(1, 1, max_len, max_len))
        self.n_head = 8

    def forward(self, x, layer_past=None):
        B, T, C = x.size()
        # batch_size, seq_len, embed_size

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, -1e10) # todo: just use float('-inf') instead?
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


############################################
#####

class TransformerDecoder(nn.Module):
    def __init__(self, input_dim, params):
        super().__init__()


        self.block1 = TransformerBlock(input_dim=input_dim,
                                       num_heads=params['num_heads'],
                                       drop_prob=params['drop_prob'],
                                       num_conv_blocks=params['num_conv_blocks'])

        self.block2 = TransformerBlock(input_dim=input_dim,
                                       num_heads=params['num_heads'],
                                       drop_prob=params['drop_prob'],
                                       num_conv_blocks=params['num_conv_blocks'])
        self.block2 = TransformerBlock(input_dim=input_dim,
                                       num_heads=params['num_heads'],
                                       drop_prob=params['drop_prob'],
                                       num_conv_blocks=params['num_conv_blocks'])

    def forward(self, x):

        m1 = self.block1(x)
        m2 = self.block2(M1)
        m3 = self.block3(M2)

        return m1, m2, m3


class TransformerBlock(nn.Module):
    def __init__(self, input_dim, num_heads, drop_prob, num_conv_blocks=1):
        super().__init__()

        self.drop_prob = drop_prob

        self.conv_blocks = nn.ModuleList(ConvBlock() for _ in range(num_conv_blocks))

        self.att = SelfAttention(input_dim, input_dim, drop_prob)
        # todo what shape
        self.att_norm = nn.LayerNorm(input_dim)

        self.ff = FeedForward(d_model=input_dim)
        # todo what shape
        self.ff_norm = nn.LayerNorm(input_dim)

    def forward(self, x):

        # potentialy reshape to go into the conv blocks
        out = x.permute(0, 2, 1)
        for conv_block in self.conv_blocks:
            out = conv_block(out)

        out = out.permute(0, 2, 1)
        # potential reshape back

        # self attention
        out = self.att(out)
        out = self.att_norm(out)

        # feed forward
        out = self.ff(out)
        out = self.ff_norm(out)

        out = F.dropout(out, self.drop_prob, self.training)

        return out




class PositionalEncoder(nn.Module):

    def __init__(self, embedding_size, drop_prob=0.1, max_len=BLOCK_SIZE, scale_factor=0.5):
        super(PositionalEncoder, self).__init__()

        # ammount by which we scale the embeddings before adding the positional encodings
        self.scale_factor = embedding_size ** scale_factor
        self.embedding_size = embedding_size
        self.drop_prob = drop_prob

        self.pe = createEncodingMatrix(embedding_size, max_len)


    def forward(self, x):
        # create mask
        # masks
        x_mask = torch.zeros_like(x) != x
        
        # broadcasting 
        # (batch_size, seq_len, embed_size) * (seq_len, embed_size) -> (batch_size, seq_len, embed_size)            
        pe_masked = x_mask * self.pe[:x.size(1),:] # 

        # scale embeddings so positional embeddings 
        # don't affect their meaning
        x_scaled = x * self.scale_factor
        # add encoding
        out = x_scaled + pe_masked
        # drop_prob
        out = F.dropout(out, self.drop_prob, self.training)

        return out


    
    def createEncodingMatrix(self, embedding_size, max_len):
        """Positional encoding matrix code from:
           https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
        """
        if embedding_size % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(embedding_size))
        
        pe = torch.zeros(max_len, embedding_size)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, embedding_size, 2, dtype=torch.float) *
                             -(math.log(10000.0) / embedding_size)))

        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe



class ConvBlock(nn.Module):

    def __init__(self, input_dim, kernel_size):

        self.depthwise = nn.Conv1d(in_channels=input_dim,
                                   out_channels=input_dim,
                                   kernel_size=kernel_size,
                                   padding=kernel_size // 2,
                                   groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels=input_dim,
                                   out_channels=input_dim,
                                   kernel_size=1)

        self.norm = nn.LayerNorm(input_dim)


    def forward(self, x):
        # shape batch, dim, seq_len
       
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.norm(out)

        return out






