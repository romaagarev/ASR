import numpy as np
import torch

from src.text_encoder.ctc_text_encoder import CTCTextEncoder


class CTCBMTextEncoder(CTCTextEncoder):
    def __init__(self, alphabet=None, beam_width=10, **kwargs):
        super().__init__(alphabet=alphabet, **kwargs)
        self.beam_width = beam_width

    def ctc_decode(self, logits: torch.Tensor) -> str:
        log_probs = logits.detach().log_softmax(dim=-1).cpu().tolist()
        blank = self.char2ind[self.EMPTY_TOK]
        beams = {(): [0.0, -np.inf]}

        for frame in log_probs:
            next_beams = {}
            for prefix, (blank_score, nonblank_score) in beams.items():
                total_score = np.logaddexp(blank_score, nonblank_score)
                if prefix not in next_beams:
                    next_beams[prefix] = [-np.inf, -np.inf]

                prefix_scores = next_beams[prefix]
                prefix_scores[0] = np.logaddexp(
                    prefix_scores[0], total_score + frame[blank]
                )

                for token, token_score in enumerate(frame):
                    if token == blank:
                        continue

                    if prefix and token == prefix[-1]:
                        prefix_scores[1] = np.logaddexp(
                            prefix_scores[1], nonblank_score + token_score
                        )
                        extension_score = blank_score + token_score
                    else:
                        extension_score = total_score + token_score

                    extension = prefix + (token,)
                    if extension not in next_beams:
                        next_beams[extension] = [-np.inf, -np.inf]

                    extension_scores = next_beams[extension]
                    extension_scores[1] = np.logaddexp(
                        extension_scores[1], extension_score
                    )

            beams = dict(
                sorted(
                    next_beams.items(),
                    key=lambda item: np.logaddexp(item[1][0], item[1][1]),
                    reverse=True,
                )[: self.beam_width]
            )

        best_prefix = max(
            beams,
            key=lambda prefix: np.logaddexp(beams[prefix][0], beams[prefix][1]),
        )
        return "".join(self.ind2char[token] for token in best_prefix)
