import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .transformer import SelfAttention, FeedForward
import math

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

        M1 = self.block1(x)
        M2 = self.block2(M1)
        M3 = self.block3(M2)

        return M1, M2, M3


class TransformerBlock(nn.Module):
    def __init__(self, input_dim, num_heads, drop_prob, num_conv_blocks=1):
        super().__init__()

        self.drop_prob

        self.conv_blocks = nn.ModuleList(ConvBlock() for _ in range(n_conv_blocks))

        self.att = SelfAttention(input_dim, input_dim, drop_prob)
        # todo what shape
        self.att_norm = nn.LayerNorm(input_dim)

        self.ff = FeedForward(d_model=input_dim)
        # todo what shape
        self.ff_norm = nn.LayerNorm(input_dim)

    def forward(self, x):

        # potentialy reshape to go into the conv blocks
        out = x
        for conv_block in self.conv_blocks:
            out = conv_block(out)

        # potential reshape back

        # self attention
        out = self.att(out)
        out = self.att_norm(out)

        # feed forward
        out = self.ff(out)
        out = self.ff_norm(out)

        out = F.dropout(out, self.drop_prob, self.training)

        return out


class ConvBlock(nn.Module):

    def __init__(self, input_dim, kernel_size):


    def forward(self, x):


        out = self.conv(x)
        out = self.norm(out)

        return out





class PositionalEncoder(nn.Module):

    def __init__(self, embedding_size, drop_prob=0.1, max_len=5000, scale_factor=0.5):
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







class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class GRUEncoder(nn.Module):
    """
    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.2):


        super(GRUEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.GRU(input_size=input_size, 
                          hidden_size=hidden_size, 
                          num_layers=num_layers,
                          batch_first=True,
                          bidirectional=True,
                          dropout=drop_prob if num_layers > 1 else 0.)


        # self.enc(input, h_0) where input -> (batch, seq, feature) 
        #                              h_0 -> (batch, num_layers * num_directions, hidden_size)
        # and outputs are           output -> (batch, seq, num_directions * hidden_size)
        #                              h_n -> (batch, num_layers * num_directions, hidden_size)
        


    def forward(self, x, lengths):

        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths, batch_first=True)

        # check shapes here
        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x 


class LSTMEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.2):

        super(LSTMEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        #print(f'X shape: {x.shape}')
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths, batch_first=True)

        #print(f'Packed X shape: {x.data.shape}')

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x