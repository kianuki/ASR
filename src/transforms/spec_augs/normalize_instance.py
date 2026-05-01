import torch
from torch import nn
from torchaudio import transforms as T


class NormalizeInstance(nn.Module):
    """
    Augmentation that masks time on the spectrogram
    """

    def __init__(self):
        """
        Args:
            time_mask_param (int): maximum possible length of the mask.
            n_time_masks (int): number of time masks to apply.
        """
        super().__init__()

    def forward(self, x):
        """
        Args:
            x (Tensor): input tensor.
        Returns:
            x (Tensor): augmented tensor.
        """
        # n_mels, T
        x_mean = x.mean()
        x_std = x.std()
        x = (x - x_mean) / (x_std + 1e-5)

        return x
