import re
from string import ascii_lowercase

import torch

# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, logits: torch.Tensor) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            logits (Tensor): frame scores with shape (time, vocab).
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        inds = logits.argmax(dim=-1).tolist()
        return "".join([self.ind2char[ind] for ind in inds]).strip()

    def ctc_decode(self, logits: torch.Tensor) -> str:
        raise NotImplementedError()

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text


class ArgmaxCTCTextEncoder(CTCTextEncoder):
    def ctc_decode(self, logits: torch.Tensor) -> str:
        inds = logits.argmax(dim=-1).tolist()
        output = ""
        last_char = None
        for ind in inds:
            char = self.ind2char[ind]
            if char != self.EMPTY_TOK and (last_char is None or char != last_char):
                output += char
            last_char = char
        return output
