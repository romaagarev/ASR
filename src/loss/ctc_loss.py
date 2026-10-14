import torch
from torch import Tensor
from torch.nn import CTCLoss


class CTCLossWrapper(CTCLoss):
    def forward(
        self, logits, logits_length, text_encoded, text_encoded_length, **batch
    ) -> Tensor:
        log_probs_t = torch.log_softmax(logits, dim=-1).transpose(0, 1)

        loss = super().forward(
            log_probs=log_probs_t,
            targets=text_encoded,
            input_lengths=logits_length,
            target_lengths=text_encoded_length,
        )

        return {"loss": loss}
