import torch
from torch import nn
from torchaudio import transforms as T


class TimeMasking(nn.Module):
    """
    Augmentation that masks time on the spectrogram
    """

    def __init__(self, time_mask_param=35, n_time_masks=2):
        """
        Args:
            time_mask_param (int): maximum possible length of the mask.
            n_time_masks (int): number of time masks to apply.
        """
        super().__init__()

        self.time_masking = nn.ModuleList(
            [
                T.TimeMasking(time_mask_param=time_mask_param)
                for _ in range(n_time_masks)
            ]
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): input tensor.
        Returns:
            x (Tensor): augmented tensor.
        """
        for aug in self.time_masking:
            x = aug(x)

        return x
