import torchaudio
from torch import Tensor, nn


class SpeedPerturbation(nn.Module):
    def __init__(self, sample_rate=16000, factors=(0.9, 1.0, 1.1)):
        super().__init__()
        self._aug = torchaudio.transforms.SpeedPerturbation(sample_rate, factors)

    def __call__(self, data: Tensor):
        x, _ = self._aug(data)
        return x
