from typing import List

from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_cer


class CERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self,
        logits: Tensor,
        logits_length: Tensor,
        text: List[str],
        pred_text=None,
        **kwargs,
    ):
        if pred_text is None:
            predictions = logits.detach().cpu()
            lengths = logits_length.detach().cpu().tolist()
            pred_text = [
                self.text_encoder.ctc_decode(logits_vec[:length])
                for logits_vec, length in zip(predictions, lengths)
            ]

        cers = []
        for target_text, prediction in zip(text, pred_text):
            target_text = self.text_encoder.normalize_text(target_text)
            cers.append(calc_cer(target_text, prediction))
        return sum(cers) / len(cers)
