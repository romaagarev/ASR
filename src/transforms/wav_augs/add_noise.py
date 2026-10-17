import torch
import torchaudio
from torch import Tensor, nn


class AddNoise(nn.Module):
    def __init__(self, min_snr=20, max_snr=30):
        super().__init__()
        self.min_snr = min_snr
        self.max_snr = max_snr

    def __call__(self, data: Tensor):
        noise = torch.randn_like(data)
        snr = data.new_empty(data.shape[:-1]).uniform_(self.min_snr, self.max_snr)
        return torchaudio.functional.add_noise(data, noise, snr)
