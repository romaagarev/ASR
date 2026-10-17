import torchaudio
from torch import Tensor, nn


class SpecAugment(nn.Module):
    def __init__(self, freq_mask_param=12, time_mask_param=30, time_mask_ratio=0.1):
        super().__init__()
        self._freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        self._time_mask = torchaudio.transforms.TimeMasking(
            time_mask_param, p=time_mask_ratio
        )

    def __call__(self, data: Tensor):
        mask_value = data.mean().item()
        x = self._freq_mask(data, mask_value=mask_value)
        return self._time_mask(x, mask_value=mask_value)
