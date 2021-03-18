import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import math
from util import masked_softmax

BLOCK_SIZE = 500


class TransformerOut(nn.Module):

    def __init__(self, input_size, drop_prob=0.1):
        super().__init__()
        self.proj1 = nn.Linear(input_size, 1)
        self.proj2 = nn.Linear(input_size, 1)
        self.proj3 = nn.Linear(input_size, 1)

    def forward(self, mod, mask):

        m1, m2, m3 = mod

        # start predictor
        # m1, m2

        #start_in = torch.cat((m1, m2), 2)
        m1_proj = self.proj1(m1)
        m2_proj = self.proj1(m2)
        m3_proj = self.proj1(m3)


        start = m1_proj + m2_proj
        log_p1 = masked_softmax(start.squeeze(), mask, log_softmax=True)

        #end_in = torch.cat((m1,m3), 2)
        end = m1_proj + m3_proj
        

        log_p2 = masked_softmax(end.squeeze(), mask, log_softmax=True)
        # end predictor
        # m1, m3
        return log_p1, log_p2


class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    I believe I could have just used torch.nn.MultiheadAttention but their documentation
    is all but absent and code ugly so I don't trust it, rolling my own here.
    """

    def __init__(self, input_size, output_size, num_heads, drop_prob=0.1, max_len=BLOCK_SIZE):
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
        self.n_head = num_heads

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


        self.block1 = TransformerBlock(input_dim=input_dim, params=params)

        self.block2 = TransformerBlock(input_dim=input_dim, params=params)

        self.block3 = TransformerBlock(input_dim=input_dim, params=params)

    def forward(self, x):

        m1 = self.block1(x)
        m2 = self.block2(m1)
        m3 = self.block3(m2)

        return m1, m2, m3


class TransformerBlock(nn.Module):
    def __init__(self, input_dim, params):
        super().__init__()

        self.drop_prob = params['drop_prob']

        num_conv_blocks = params['num_conv_blocks']
        self.conv_blocks = nn.ModuleList(ConvBlock(input_dim, params['kernel_size']) for _ in range(num_conv_blocks))

        self.att = SelfAttention(input_size=input_dim, 
                                 output_size=input_dim, 
                                 num_heads=params['num_heads'],
                                 drop_prob=params['drop_prob'])
        # todo what shape
        self.att_norm = nn.LayerNorm(input_dim)

        self.ff = FeedForward(input_dim=input_dim, hidden_dim=params['hidden_dim'], drop_prob=params['drop_prob'])
        # todo what shape
        self.ff_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        # input batch_size, seq_len, embedding_size

        # potentialy reshape to go into the conv blocks
        out = x.permute(0, 2, 1)
        for conv_block in self.conv_blocks:
            out = conv_block(out)

        out = out.permute(0, 2, 1)
        # potential reshape back

        # self attention

        out = self.att(out) # batch_size, seq_len, output_dim
        out = self.att_norm(out)

        # feed forward
        out = self.ff(out) # same in same in same out
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

        self.createEncodingMatrix(embedding_size, max_len)
        

    def forward(self, x):
        # create mask
        # masks
        x_mask = torch.zeros_like(x) != x
        
        # broadcasting 
        # (batch_size, seq_len, embed_size) * (seq_len, embed_size) -> (batch_size, seq_len, embed_size)            
        #pe_masked = x_mask.float() * self.pe[:x.size(1),:] # 

        # scale embeddings so positional embeddings 
        # don't affect their meaning
        x_scaled = x * self.scale_factor
        # add encoding
        out = x_scaled + self.pe[:x.size(1),:]#pe_masked
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

        self.pe = nn.Parameter(pe, requires_grad=False)



class ConvBlock(nn.Module):

    def __init__(self, input_dim, kernel_size):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels=input_dim,
                                   out_channels=input_dim,
                                   kernel_size=kernel_size,
                                   padding=kernel_size // 2,
                                   groups=input_dim)

        self.pointwise = nn.Conv1d(in_channels=input_dim,
                                   out_channels=input_dim,
                                   kernel_size=1)

        self.norm = nn.LayerNorm(input_dim)


    def forward(self, x):
        # shape batch, dim, seq_len
       
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = out.permute(0,2,1)
        out = self.norm(out)
        out = out.permute(0,2,1)

        return out



class FeedForward(nn.Module):

    def __init__(self, input_dim, hidden_dim, drop_prob):
        super().__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)

        self.dropout = nn.Dropout(drop_prob)

        self.linear2 = nn.Linear(hidden_dim, input_dim)


    def forward(self, x):

        out = F.relu(self.linear1(x))
        out = self.dropout(out)
        out = self.linear2(out)

        return out


