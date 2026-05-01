import torch
from torch import nn
from torchaudio import transforms as T


class FrequencyMasking(nn.Module):
    """
    Augmentation that masks frequency on the spectrogram
    """

    def __init__(self, freq_mask_param=15, n_freq_masks=2):
        """
        Args:
            freq_mask_param (int): maximum possible length of the mask.
            n_freq_masks (int): number of frequency masks to apply.
        """
        super().__init__()

        self.freq_masking = nn.ModuleList(
            [
                T.FrequencyMasking(freq_mask_param=freq_mask_param)
                for _ in range(n_freq_masks)
            ]
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): input tensor.
        Returns:
            x (Tensor): augmented tensor.
        """
        for aug in self.freq_masking:
            x = aug(x)

        return x
