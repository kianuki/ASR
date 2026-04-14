"""
Conformer implementation based on the following papers:

[1] Conformer: Convolution-augmented Transformer for Speech Recognition (https://doi.org/10.48550/arXiv.2005.08100)
[2] Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context (https://doi.org/10.48550/arXiv.1901.02860)
"""

import math

import torch
from torch import nn
from torch.nn.functional import pad


class FeedForwardModule(nn.Module):
    """
    Feed Forward Module for Conformer [1]
    """

    def __init__(self, in_features, dropout_p=0.1):
        """
        Args:
            in_features (int): number of input features.
            dropout_p (float): probability of zeroing an element in nn.Dropout
        """
        super().__init__()

        self.module = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features=in_features, out_features=4 * in_features),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(in_features=4 * in_features, out_features=in_features),
            nn.Dropout(dropout_p),
        )

    def forward(self, x):
        return self.module(x)


class ConvolutionModule(nn.Module):
    """
    Convolution Module for Conformer [1]
    """

    def __init__(self, in_features, dropout_p=0.1):
        """
        Args:
            in_features (int): number of input features.
            dropout_p (float): probability of zeroing an element in nn.Dropout
        """
        super().__init__()

        self.layernorm = nn.LayerNorm(in_features)
        self.pointwise_conv_expand = nn.Conv1d(
            in_channels=in_features,
            out_channels=2 * in_features,
            kernel_size=1,
        )
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            in_channels=in_features,
            out_channels=in_features,
            kernel_size=31,
            padding=15,
            groups=in_features,
        )
        self.batchnorm = nn.BatchNorm1d(num_features=in_features)
        self.swish = nn.SiLU()
        self.pointwise_conv = nn.Conv1d(
            in_channels=in_features, out_channels=in_features, kernel_size=1
        )
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.layernorm(x).transpose(1, 2)
        x = self.pointwise_conv_expand(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batchnorm(x)
        x = self.swish(x)
        x = self.pointwise_conv(x)
        x = self.dropout(x)

        return x.transpose(1, 2)


class RelativePositionalEncoding(nn.Module):
    """
    Relative Positional Encoding using sinusoid encoding described in Tranformer-XL [2]
    Used in Conformer's [1] Attention Module
    """

    def __init__(self, in_features, max_len):
        """
        Args:
            in_features (int): number of input features.
            max_len (int): maximum context length.
        """
        super().__init__()
        self.center = max_len - 1
        self.register_buffer("pe", self._generate_sinusoids(in_features, max_len))

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, T, n_mels)
        Returns sinusoid matrix of shape (max_len, in_features) for all i-j distances
        """
        T = x.size(1)

        return self.pe[:, self.center - (T - 1) : self.center + T, :]

    def _generate_sinusoids(self, in_features, max_len):
        pos = torch.arange(-(max_len - 1), max_len, dtype=torch.float)
        div_term = torch.arange(in_features // 2)
        div_term = torch.exp(2 * div_term * (-math.log(10000.0) / in_features))
        terms = pos.unsqueeze(1) * div_term

        pe = torch.zeros(2 * max_len - 1, in_features)
        pe[:, 0::2] = torch.sin(terms)
        pe[:, 1::2] = torch.cos(terms)

        return pe.unsqueeze(0)


class MultiheadAttention(nn.Module):
    """
    Multihead Attention with Relative Positional Encoding, described in Tranformer-XL [2]
    Used in Conformer's [1] Attention Module
    """

    def __init__(self, in_features, max_len, num_heads):
        """
        Args:
            in_features (int): number of input features.
            max_len (int): maximum context length.
            num_heads (int): number of heads for Attention
        """
        super().__init__()

        self.num_heads = num_heads
        assert in_features % num_heads == 0, "in_features % num_heads should be 0"
        self.head_dim = in_features // num_heads
        self.sqrt_dim = math.sqrt(self.head_dim)

        self.query_proj = nn.Linear(in_features=in_features, out_features=in_features)
        self.value_proj = nn.Linear(in_features=in_features, out_features=in_features)
        self.k_content_proj = nn.Linear(
            in_features=in_features, out_features=in_features
        )
        self.k_location_proj = nn.Linear(
            in_features=in_features, out_features=in_features
        )

        self.position_encoder = RelativePositionalEncoding(in_features, max_len)

        self.u = nn.Parameter(torch.Tensor(num_heads, self.head_dim))
        self.v = nn.Parameter(torch.Tensor(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.u)
        nn.init.xavier_uniform_(self.v)

    def forward(self, x, mask):
        B, T, out_f = x.shape
        pe = self.position_encoder(x)

        # (B, T, out_f) > (B, T, num_heads, head_dim) > (B, num_heads, T, head_dim)
        query = self.query_proj(x).view(
            B, T, self.num_heads, self.head_dim
        )  # .transpose(1, 2)
        value = (
            self.value_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        )
        k_content = (
            self.k_content_proj(x)
            .view(B, T, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k_location = (
            self.k_location_proj(pe)
            .view(1, 2 * T - 1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Calculating terms, as described in Transformer-XL Section 3.3
        ac = torch.matmul((query + self.u).transpose(1, 2), k_content.transpose(-2, -1))
        bd = torch.matmul(
            (query + self.v).transpose(1, 2), k_location.transpose(-2, -1)
        )
        bd = self._relative_shift(bd)

        attention = (ac + bd) / self.sqrt_dim
        attention = attention.masked_fill_(mask == 0, -float("inf"))
        log_probs = nn.functional.softmax(attention, dim=-1)
        outputs = torch.matmul(log_probs, value)

        return outputs.view(B, T, out_f)

    def _relative_shift(self, pos_score):
        B, H, T1, T2 = pos_score.shape

        padded_pos_score = pad(pos_score, (1, 0))
        padded_pos_score = padded_pos_score.view(B, H, T2 + 1, T1)
        padded_pos_score = padded_pos_score[:, :, 1:, :]

        return padded_pos_score.view(B, H, T1, T2)[:, :, :, : T2 // +1]


class MultiheadSelfAttentionModule(nn.Module):
    def __init__(self, in_features, dropout_p=0.1, max_len=5000, num_heads=4):
        super().__init__()

        self.layernorm = nn.LayerNorm(in_features)
        self.attention = MultiheadAttention(in_features, max_len, num_heads)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        pass


class Conformer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        pass

    def forward(self):
        pass
