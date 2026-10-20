from torch import Tensor, nn


class StandardNormalizer(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def __call__(self, data: Tensor):
        mean = data.mean(dim=-1, keepdim=True)
        std = data.std(dim=-1, keepdim=True, unbiased=False)
        return (data - mean) / (std + self.eps)
