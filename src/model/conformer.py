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

        self.out_proj = nn.Linear(in_features=in_features, out_features=in_features)

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
        mask_value = -1e30 if attention.dtype == torch.float32 else -1e4
        attention.masked_fill_(~mask.unsqueeze(1), mask_value)
        attn_weights = nn.functional.softmax(attention, dim=-1)
        outputs = torch.matmul(attn_weights, value).transpose(1, 2)

        return self.out_proj(outputs.contiguous().view(B, T, out_f))

    def _relative_shift(self, pos_score):
        B, H, T1, T2 = pos_score.shape

        padded_pos_score = pad(pos_score, (1, 0))
        padded_pos_score = padded_pos_score.view(B, H, T2 + 1, T1)
        padded_pos_score = padded_pos_score[:, :, 1:, :]

        return padded_pos_score.view(B, H, T1, T2)[:, :, :, : T2 // 2 + 1]


class MultiheadSelfAttentionModule(nn.Module):
    def __init__(self, in_features, dropout_p, max_len, num_heads):
        super().__init__()

        self.layernorm = nn.LayerNorm(in_features)
        self.attention = MultiheadAttention(in_features, max_len, num_heads)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, mask):
        x = self.layernorm(x)
        x = self.attention(x, mask)
        x = self.dropout(x)

        return x


class ConformerBlock(nn.Module):
    def __init__(self, in_features, dropout_p, max_len, num_heads):
        super().__init__()

        self.ffn1 = FeedForwardModule(in_features, dropout_p)
        self.ffn2 = FeedForwardModule(in_features, dropout_p)
        self.mhsa = MultiheadSelfAttentionModule(
            in_features, dropout_p, max_len, num_heads
        )
        self.conv = ConvolutionModule(in_features, dropout_p)
        self.layernorm = nn.LayerNorm(in_features)

    def forward(self, x, mask):
        x = x + 0.5 * self.ffn1(x)
        x = x + self.mhsa(x, mask)
        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)

        return self.layernorm(x)


class SubsamplingModule(nn.Module):
    def __init__(self, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=out_channels, stride=2, kernel_size=3
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, stride=2, kernel_size=3
        )
        self.swish = nn.SiLU()
        self.pish = nn.SiLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.swish(x)
        x = self.conv2(x)
        x = self.pish(x)

        return x


class Conformer(nn.Module):
    def __init__(
        self,
        in_features,
        n_tokens,
        n_mels=80,
        dropout_p=0.1,
        max_len=5000,
        num_heads=4,
        num_blocks=4,
        out_channels=4,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.conformer_blocks = nn.ModuleList(
            [
                ConformerBlock(in_features, dropout_p, max_len, num_heads)
                for _ in range(num_blocks)
            ]
        )
        self.proj = nn.Linear(
            in_features=out_channels * (((n_mels - 1) // 2 - 1) // 2),
            out_features=in_features,
        )
        self.dropout = nn.Dropout(dropout_p)
        self.subsampling = SubsamplingModule(out_channels)
        self.classifier_proj = nn.Linear(in_features=in_features, out_features=n_tokens)

    def forward(
        self, spectrogram, spectrogram_length, text_encoded_length=None, **batch
    ):
        # B, n_mels, T -> B, 1, n_mels, T -> B, out_channels, n_mels/4, T/4 -> B, out_channels * n_mels/4, T/4 -> B, T/4, d_model
        B = spectrogram.size(0)
        max_len = spectrogram.size(-1)
        device = spectrogram.device
        mask = torch.arange(max_len, device=device).expand(
            B, max_len
        ) < spectrogram_length.unsqueeze(1).to(device)

        x = self.subsampling(spectrogram.unsqueeze(1))  # B, out_channels, nn_mels, TT

        mask = mask[:, :-2:2]
        mask = mask[:, :-2:2]
        mask = torch.min(mask[:, None, :], mask[:, :, None])

        x = x.permute(0, 3, 1, 2)  # B, TT, out_channels, nn_mels
        TT = x.size(1)
        x = x.view(B, TT, -1)
        x = self.proj(x)  # B, TT, d_model
        x = self.dropout(x)

        for conformer_block in self.conformer_blocks:
            x = conformer_block(x, mask)

        x = self.classifier_proj(x)

        log_probs = nn.functional.log_softmax(x, dim=-1)
        log_probs_length = self.transform_input_lengths(spectrogram_length)

        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    def transform_input_lengths(self, input_lengths):
        """
        Calculate output lengths after subsampling.

        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            output_lengths (Tensor): new temporal lengths
        """
        return ((input_lengths // 2) - 1) // 2 - 1

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
