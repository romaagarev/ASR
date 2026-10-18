from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from src.text_encoder.ctc_bm_text_encoder import CTCBMTextEncoder


@dataclass
class LMState:
    score: float
    kv_cache: torch.Tensor
    log_probs: np.ndarray


class CTCBMLMTextEncoder(CTCBMTextEncoder):
    def __init__(
        self,
        alphabet=None,
        beam_width=10,
        lm_name="nllg/bygpt5-small-en",
        lm_weight=0.3,
        word_bonus=1.0,
        device=None,
        **kwargs,
    ):
        super().__init__(alphabet=alphabet, beam_width=beam_width, **kwargs)
        self.lm_name = lm_name
        self.lm_weight = lm_weight
        self.word_bonus = word_bonus
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.lm = None

    def _load_lm(self):
        if self.lm is not None:
            return

        from uniformers.models.bygpt5 import ByGPT5LMHeadModel, ByGPT5Tokenizer

        tokenizer = ByGPT5Tokenizer.from_pretrained(
            self.lm_name,
            add_prefix_space=False,
            add_bos_token=False,
            add_eos_token=False,
        )
        self.lm_token_ids = {}
        for char in self.alphabet:
            token_ids = tokenizer.encode(char, add_special_tokens=False)

            self.lm_token_ids[self.char2ind[char]] = token_ids[0]
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.lm = ByGPT5LMHeadModel.from_pretrained(self.lm_name, torch_dtype=dtype).to(
            self.device
        )
        self.lm.eval()
        self.lm.requires_grad_(False)
        self.start_token_id = self.lm.config.decoder_start_token_id

    def _start_lm(self):
        self._load_lm()
        tokens = torch.tensor([[self.start_token_id]], device=self.device)
        output = self.lm(tokens, use_cache=True, return_dict=True)
        kv_cache = torch.stack(
            [value for layer in output.past_key_values for value in layer]
        )
        log_probs = output.logits[0, -1].float().log_softmax(-1).cpu().numpy()
        return {(): LMState(0.0, kv_cache, log_probs)}

    def _lm_score(self, prefix, states):
        if not states:
            return 0.0
        if prefix in states:
            return states[prefix].score

        parent = states[prefix[:-1]]
        token = self.lm_token_ids[prefix[-1]]
        return parent.score + float(parent.log_probs[token])

    def _update_lm(self, prefixes, states):
        next_states = {
            prefix: states[prefix] for prefix in prefixes if prefix in states
        }
        missing = [prefix for prefix in prefixes if prefix not in states]
        if not missing:
            return next_states

        parents = [states[prefix[:-1]] for prefix in missing]
        lengths = [parent.kv_cache.shape[-2] for parent in parents]
        max_length = max(lengths)
        kv_cache = torch.cat(
            [
                F.pad(parent.kv_cache, (0, 0, max_length - length, 0))
                for parent, length in zip(parents, lengths)
            ],
            dim=1,
        )
        past_key_values = tuple(
            (kv_cache[i], kv_cache[i + 1]) for i in range(0, kv_cache.shape[0], 2)
        )
        mask = torch.zeros(
            (len(missing), max_length + 1), device=self.device, dtype=torch.long
        )
        for i, length in enumerate(lengths):
            mask[i, max_length - length :] = 1

        tokens = torch.tensor(
            [[self.lm_token_ids[prefix[-1]]] for prefix in missing], device=self.device
        )
        output = self.lm(
            tokens,
            past_key_values=past_key_values,
            attention_mask=mask,
            use_cache=True,
            return_dict=True,
        )
        kv_cache = torch.stack(
            [value for layer in output.past_key_values for value in layer]
        )
        log_probs = output.logits[:, -1].float().log_softmax(-1).cpu().numpy()
        for i, (prefix, length) in enumerate(zip(missing, lengths)):
            prefix_kv_cache = kv_cache[
                :, i : i + 1, :, max_length - length :, :
            ].clone()
            next_states[prefix] = LMState(
                self._lm_score(prefix, states), prefix_kv_cache, log_probs[i].copy()
            )
        return next_states

    @torch.inference_mode()
    def ctc_decode(self, logits: torch.Tensor) -> str:
        if logits.shape[0] == 0:
            return ""
        if self.lm_weight == 0 and self.word_bonus == 0:
            return super().ctc_decode(logits)

        log_probs = logits.detach().float().log_softmax(dim=-1).cpu().tolist()
        blank = self.char2ind[self.EMPTY_TOK]
        space = self.char2ind.get(" ")
        beams = [((), [0.0, -np.inf])]
        states = self._start_lm() if self.lm_weight else {}

        def beam_score(item):
            prefix, (blank_score, nonblank_score) = item
            return (
                np.logaddexp(blank_score, nonblank_score)
                + self.lm_weight * self._lm_score(prefix, states)
                + self.word_bonus * prefix.count(space)
            )

        for frame in log_probs:
            next_beams = {}
            for prefix, (blank_score, nonblank_score) in beams:
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
                    if extension_score == -np.inf:
                        continue

                    extension = prefix + (token,)
                    if extension not in next_beams:
                        next_beams[extension] = [-np.inf, -np.inf]

                    extension_scores = next_beams[extension]
                    extension_scores[1] = np.logaddexp(
                        extension_scores[1], extension_score
                    )

            beams = sorted(next_beams.items(), key=beam_score, reverse=True)[
                : self.beam_width
            ]
            if states:
                states = self._update_lm([prefix for prefix, _ in beams], states)

        best_prefix = beams[0][0]
        return "".join(self.ind2char[token] for token in best_prefix).strip()
