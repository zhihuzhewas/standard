import torch
import torch.nn as nn
from layers.transformer_encdec import Decoder, DecoderLayer, Encoder, EncoderLayer, Decoder_performer, DecoderLayer_performer, Encoder_performer, EncoderLayer_performer
from layers.embed import DataEmbedding
from layers.attention import MultiHeadAttention
from performer_pytorch import SelfAttention, CrossAttention

class Model(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    MultiHeadAttention(configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    MultiHeadAttention(configs.d_model, configs.n_heads),
                    MultiHeadAttention(configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, original=True):
        # init
        if original:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev  # B L C


        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        # back
        if original:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, dec_out.shape[1], 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, dec_out.shape[1], 1))

        return dec_out[:, -self.pred_len:, :], attns


class Model_performer(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs):
        super(Model_performer, self).__init__()
        self.pred_len = configs.pred_len

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.encoder = Encoder_performer(
            [
                EncoderLayer_performer(
                    # MultiHeadAttention(configs.d_model, configs.n_heads),
                    SelfAttention(dim=configs.d_model, heads=configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder_performer(
            [
                DecoderLayer_performer(
                    # MultiHeadAttention(configs.d_model, configs.n_heads),
                    # MultiHeadAttention(configs.d_model, configs.n_heads),
                    SelfAttention(dim=configs.d_model, heads=configs.n_heads, causal=True),
                    CrossAttention(dim=configs.d_model, heads=configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, original=True):
        # init
        if original:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev  # B L C


        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        # back
        if original:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, dec_out.shape[1], 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, dec_out.shape[1], 1))
        return dec_out[:, -self.pred_len:, :], 1