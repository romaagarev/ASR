from typing import List

from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_wer

class ArgmaxWERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, logits: Tensor, logits_length: Tensor, text: List[str], **kwargs
    ):
        wers = []
        predictions = logits.detach().cpu()
        lengths = logits_length.detach().cpu().tolist()
        for logits_vec, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_decode(logits_vec[:length])
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)
