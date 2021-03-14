import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

from util import masked_softmax

#from encoder import RNNEncoder


#Modified version of multi-head self-attention from assign 5
class MultiHeadAttention(nn.Module):
    """ 
    A vanilla multi-head masked self-attention layer with a projection at the end.
    I believe I could have just used torch.nn.MultiheadAttention but their documentation
    is all but absent and code ugly so I don't trust it, rolling my own here.
    """

    def __init__(self, embd_size, num_heads, attn_drop_prob=0.1, resid_drop_prop=0.1):
        super().__init__()
        assert embd_size % num_heads == 0

        # key, query, value projections for all heads
        self.key = nn.Linear(embd_size, embd_size)
        self.query = nn.Linear(embd_size, embd_size)
        self.value = nn.Linear(embd_size, embd_size)

        #TODO - Version from class uses two dropout layers, not sure if this should be changed or is necessary
        # regularization 
        self.attn_drop = nn.Dropout(attn_drop_prob)
        self.resid_drop = nn.Dropout(resid_drop_prop)

        # output projection
        self.proj = nn.Linear(embd_size, embd_size)

        # Set Num Heads
        self.num_heads = num_heads
        
        #self.resid_drop = nn.Dropout(resid_pdrop)

        #TODO - Rmemoved for now, not using block size anywhere else might need to replace
        # causal mask to ensure that attention is only applied to the left in the input sequence
        #self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
        #                             .view(1, 1, block_size, block_size))

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, -1e10) # todo: just use float('-inf') instead?
        att = F.softmax(att, dim=-1)

        att = self.attn_drop_prob(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop_prop(self.proj(y))
        return y

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
    def __init__(self, embd_size, num_layers, num_heads):
        super().__init__()
        self.num_layers = num_layers
        self.layers = self.get_clones(TransformerEncoderLayer(embd_size=embd_size, num_heads=num_heads), num_layers)
        self.norm = Norm(embd_size)

    #Convenience function to allow us to generate num_layers identical copies of our encoder layer
    def get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    
    # Apply each attention layer
    # TODO not sure how mask comes into play, we're already using masking for embeddings so maybe that will do
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
    def __init__(self, embd_size, num_heads, dropout = 0.1):
        super().__init__()
        self.norm_1 = Norm(embd_size)
        self.norm_2 = Norm(embd_size)
        self.attn = MultiHeadAttention(embd_size=embd_size, num_heads=num_heads)
        self.ff = FeedForward(embd_size)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
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
    def __init__(self, embd_size, num_layers, num_heads):
        super().__init__()
        self.num_layers = num_layers
        self.layers = self.get_clones(TransformerDecoderLayer(embd_size=embd_size, num_heads=num_heads), num_layers)
        self.norm = Norm(embd_size)

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
    def __init__(self, embd_size, num_heads, dropout = 0.1):
        super().__init__()
        self.norm_1 = Norm(embd_size)
        self.norm_2 = Norm(embd_size)
        self.attn = MultiHeadAttention(embd_size=embd_size, num_heads=num_heads)
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



'''
class selfAttOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2
'''

'''
class PositionWiseFeedForwardLayer(nn.Module):
    """ 
    Feed forward layer based on Vaswani et al 2017
    """

    def __init__(self, n_embd, attn_pdrop, resid_pdrop, block_size, n_head=8):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))
        self.n_head = n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()


        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # todo equation (2) section 3.3 
        # FFN(x) = max(0, xW1 + b1) W2 + b2
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, -1e10) # todo: just use float('-inf') instead?
        att = F.softmax(att, dim=-1)

        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

    #copied from class
class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    I believe I could have just used torch.nn.MultiheadAttention but their documentation
    is all but absent and code ugly so I don't trust it, rolling my own here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

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

'''