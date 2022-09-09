''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
import math
#from transformer.Layers import EncoderLayer, DecoderLayer


__author__ = "Peng-Yu Chen"


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)

def get_subsequent_mask(seq, sz_limit=0):
    ''' For masking out the subsequent info. 
        * args:
            * limit: Leave visible for those locating at (sz-sz_limit <= (token position) < sz)
    '''
    len_s, sz, _ = seq.size()
    mask = torch.zeros((sz, sz))
    _sz = sz - sz_limit
    print(_sz)
    mask[:_sz, :_sz] = torch.triu(torch.full((_sz, _sz), float('-inf')), diagonal=1)

    return mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, d_word_vec, n_layers, n_head, d_model, d_inner,
            use_proj=True, d_src_vocab=None,
            dropout=0.1, scale_emb=False):

        super().__init__()
        
        if use_proj:
            assert type(d_src_vocab) == int, 'ValueError: \`d_src_vocab\` should be specified when \`use_proj\` is \`True\`'

        self.proj = nn.Linear(d_src_vocab, d_word_vec, bias=True) if use_proj else nn.Identity()
        self.position_enc = PositionalEncoding(d_word_vec, dropout=dropout)
        self.layer_stack = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward=d_inner, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask):

        enc_output = self.proj(src_seq).transpose(0, 1) # In NLP, data should be (seq_len, batch_size, ...)
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5
        enc_output = self.position_enc(enc_output)
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, src_mask=src_mask)

        return enc_output.transpose(0, 1)


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, d_word_vec, n_layers, n_head, d_model, d_inner,
            use_proj=True, d_trg_vocab=None, 
            dropout=0.1, scale_emb=False):

        super().__init__()

        if use_proj:
            assert type(d_trg_vocab) == int, 'ValueError: \`d_trg_vocab\` should be specified when \`use_proj\` is \`True\`'

        self.proj = nn.Linear(d_trg_vocab, d_word_vec, bias=True) if use_proj else nn.Identity()
        self.position_enc = PositionalEncoding(d_word_vec, dropout=dropout)
        self.layer_stack = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model, n_head, dim_feedforward=d_inner, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask):

        enc_output = enc_output.transpose(0, 1) # In NLP, input data is used to be (seq_len, batch_size, ...)
        dec_output = self.proj(trg_seq).transpose(0, 1)
        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        dec_output = self.position_enc(dec_output)
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output = dec_layer(dec_output, enc_output, tgt_mask=trg_mask, memory_mask=src_mask)

        return dec_output.transpose(0, 1)
