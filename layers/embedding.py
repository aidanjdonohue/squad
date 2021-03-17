import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import HighwayEncoder



class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        char_vectors (torch.Tensor): Char vectors
        embedding (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
        char_embedder (str): specify a default char embedder (cnn, dense)
    
    """

    def __init__(self, word_vectors, char_vectors, params):
        super(Embedding, self).__init__()

        self.drop_prob = params['drop_prob']

        self.embedding_size = params['embedding_size']
        self.hidden_size = params['embedding_size'] * 2
        
        self.word_embd = WordEmbedding(word_vectors=word_vectors, 
                                       embedding_size=self.embedding_size, 
                                       drop_prob=self.drop_prob)


        self.hwy = HighwayEncoder(params['hwy_layers'], 
                                  self.hidden_size)

        if params['char_embedder'] == 'cnn':
            self.char_embd = CNNCharEmbedding(char_vectors=char_vectors, 
                                              embedding_size=self.embedding_size, 
                                              drop_prob=self.drop_prob, 
                                              kernel_size=params['kernel_size'])
        else:
            raise Exception(f"char_embedder: {params['char_embedder']} not implemented")


    def forward(self, c, w):
        """
        Args:
            c (torch.Tensor): char_idxs (batch_size, seq_len, word_len)
            w (torch.Tensor): word_idxs (batch_size, seq_len, word_len)
        Output:
            out (torch.Tensor): batch_size, seq_len, hidden_size
        """
        # 1. char embed
        char_embd = self.char_embd(c)

        # 2. word embed
        word_embd = self.word_embd(w)

        # 3. concat and pass to highway
        comb_embd = torch.cat((char_embd, word_embd), 2)
        out = self.hwy(comb_embd) # (batch_size, seq, hidden_size)

        return out


        
class DepthwiseSeperableConv(nn.Module):

    def __init__(self, in_size, kernel_size):
        super(DepthwiseSeperableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels=in_size, 
                                   out_channels=in_size, 
                                   kernel_size=kernel_size,
                                   padding=1,
                                   groups=in_size)

        self.pointwise = nn.Conv2d(in_channels=in_size, 
                                   out_channels=in_size, 
                                   kernel_size=kernel_size,
                                   padding=1,
                                   groups=in_size)


    def forward(self, x):
        # input_shape = 
        out = self.depthwise(x)
        out = self.pointwise(x)
        return out




class CNNCharEmbedding(nn.Module):

    def __init__(self, char_vectors, embedding_size, drop_prob, kernel_size):
        super(CNNCharEmbedding, self).__init__()

        #print(f'char_vectors: {str(char_vectors.size())[10:]}')
        vocab_size = char_vectors.size(0)
        embed_size = char_vectors.size(1)
        self.embd = nn.Embedding(vocab_size, embed_size)

        self.embedding_size = embedding_size
        self.drop_prob = drop_prob
        
        self.cnn = nn.Conv2d(1, self.embedding_size, kernel_size)


    def forward(self, x):


        batch_size, seq_len, word_len = x.size()
        #print(f'x_in: {str(x.size())[10:]}')
        out = x.contiguous().view(-1, word_len) # (N*seq_len, word_len)
        #print(f'2: {str(out.size())[10:]}')
        out = self.embd(out)
        
        #print(f'3: {str(out.size())[10:]}')
        out = out.view(*x.size(), -1).sum(2)# Batch, seq, c_embd_size
        #print(f'x_mod: {str(out.size())[10:]}')
        
        out = out.unsqueeze(1)
        #print(f'cnn_inn: {str(out.size())[10:]}')
        

        out = self.cnn(out)
        out = F.relu(out)

        #print(f'cnn_out: {str(out.size())[10:]}')
        

        out = torch.max(out, dim=-1)[0] 

        #print(f'max_out: {str(out.size())[10:]}')
        out = out.permute(0, 2, 1)

        #print(f'permute_out: {str(out.size())[10:]}')

        assert out.shape[0] == batch_size
        assert out.shape[1] == seq_len
        assert out.shape[2] == self.embedding_size

        out = F.dropout(out, self.drop_prob, self.training)


        return out


# https://github.com/jojonki/BiDAF/blob/master/layers/word_embedding.py
class WordEmbedding(nn.Module):
    '''
    In : (N, sentence_len)
    Out: (N, sentence_len, embd_size)
    '''
    def __init__(self, word_vectors, embedding_size, drop_prob):
        super(WordEmbedding, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(word_vectors)
        self.proj = nn.Linear(word_vectors.size(1), embedding_size, bias=False)
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
    def __init__(self, char_vectors, out_channels=100, filters=[[1,5]], drop_prob=0.2, pre_trained=False):
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


class defEmbedding(nn.Module):
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
