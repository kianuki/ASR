from torch import nn


class FeedForwardModule(nn.Module):
    def __init__(self, in_features, dropout_p=0.1):
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
    def __init__(self, in_features, dropout_p=0.1):
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


class Conformer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        pass

    def forward(self):
        pass
