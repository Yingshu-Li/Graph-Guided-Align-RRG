import torch
import copy
import math
from pickle import STACK_GLOBAL
import torch
from torch import nn
import torch.nn.functional as F

fw_adj = torch.tensor([
    [1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0],  # 1 Atelectasis
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 2 Cardiomegaly
    [1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0],  # 3 Consolidation
    [1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0],  # 4 Edema
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 5 Enlarged Cardiomediastinum
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],  # 6 Fracture
    [1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0],  # 7 Lung Lesion
    [1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0],  # 8 Lung Opacity
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 9 normal (no findings)
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0],  # 10 Pleural Effusion
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0],  # 11 Pleural Other
    [1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0],  # 12 Pneumonia
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0],  # 13 Pneumothorax
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]   # 14 Support Devices
], dtype=torch.float)

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model,self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x,x,x,mask))
        return self.sublayer[1](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(90, d_model)
        position = torch.arange(0, 90).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, x_node_inds):
        self.inds = x_node_inds.tolist()
        tmp_pe = self.pe[:, :x.size(1)]
        final_pe = torch.zeros(x.size())
        for num in range(len(self.inds)):
            final_pe[:,num,:] = tmp_pe[:,self.inds[num]]
        x = x + final_pe.to(x.device)
        return self.dropout(x)


class TagEncoder(nn.Module):

    def __init__(self,d_model, dropout):
        super(TagEncoder,self).__init__()
        c = copy.deepcopy
        self.dropout = nn.Dropout(p=dropout)
        self.encoder = Encoder(EncoderLayer(d_model, c(MultiHeadedAttention(8, d_model)), 
                                            c(PositionwiseFeedForward(d_model, 2048, 0.1)), 
                                dropout), 
                                2)
        self.pe = PositionalEncoding(d_model, dropout)
        self.layer_norm = LayerNorm(d_model)
        self.node_inds = torch.tensor([2, 1, 2, 2, 1, 4, 2, 2, 5, 3, 3, 2, 3, 6])
        self.mask = fw_adj
        
    def forward(self, node_embedding):
        x = node_embedding
        x = self.pe(x, self.node_inds)
        x = self.layer_norm(x)
        mask = self.mask.to(x.device)
        x = self.encoder(x, mask)
        return self.dropout(x)
    

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, hidden_states):
        for layer in self.layers:
            x, y = layer(x, hidden_states)
        return self.norm(x), y


class DecoderLayer(nn.Module):
    def __init__(self, d_model, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        # self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)

    def forward(self, x, hidden_states):
        m = hidden_states
        x = self.sublayer[0](x, lambda x: self.src_attn(x, m, m))
        return self.sublayer[1](x, self.feed_forward), self.src_attn.attn