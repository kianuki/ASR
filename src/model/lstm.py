from torch import nn
from torch.nn import Sequential


class LSTM(nn.Module):
    def __init__(self, n_feats, n_tokens, fc_hidden, n_layers, **kwargs):
        super().__init__()

        self.n_feats = n_feats
        self.n_tokens = n_tokens
        self.fc_hidden = fc_hidden
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            input_size=self.n_feats,
            hidden_size=self.fc_hidden,
            num_layers=self.n_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(
            in_features=2 * self.fc_hidden, out_features=self.n_tokens
        )

    def forward(self, spectrogram, spectrogram_length, **batch):
        output, _ = self.lstm(spectrogram.transpose(2, 1))
        output = self.proj(output)
        log_probs = nn.functional.log_softmax(output, dim=-1)
        log_probs_length = self.transform_input_lengths(spectrogram_length)

        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    def transform_input_lengths(self, input_length):
        # doesn't compress the time dimension

        return input_length

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
