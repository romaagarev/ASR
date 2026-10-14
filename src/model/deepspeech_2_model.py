from torch import nn
from torch.nn import Sequential

from src.model.rnn import RNN


class DeepSpeech2Model(nn.Module):

    def __init__(self, n_feats, n_tokens, fc_hidden=512):
        """
        Args:
            n_feats (int): number of input features.
            n_tokens (int): number of tokens in the vocabulary.
            fc_hidden (int): number of hidden features.
        """
        super().__init__()

        self.net = nn.ModuleDict(
            {
                "conv": Sequential(
                    nn.Conv2d(
                        in_channels=1,
                        out_channels=32,
                        kernel_size=(41, 11),
                        stride=(2, 2),
                        padding=(20, 5),
                    ),
                    nn.BatchNorm2d(32),
                    nn.Hardtanh(min_val=0, max_val=20),
                    nn.Conv2d(
                        in_channels=32,
                        out_channels=32,
                        kernel_size=(21, 11),
                        stride=(2, 1),
                        padding=(10, 5),
                    ),
                    nn.BatchNorm2d(32),
                    nn.Hardtanh(min_val=0, max_val=20),
                    nn.Conv2d(
                        in_channels=32,
                        out_channels=96,
                        kernel_size=(21, 11),
                        stride=(2, 1),
                        padding=(10, 5),
                    ),
                    nn.BatchNorm2d(96),
                    nn.Hardtanh(min_val=0, max_val=20),
                ),
                "rnn": RNN(
                    input_size=96 * ((n_feats + 7) // 8),
                    hidden_size=fc_hidden,
                    num_layers=7,
                    bidirectional=True,
                ),
                "fc": nn.Linear(in_features=fc_hidden, out_features=n_tokens),
            }
        )

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): input spectrogram.
            spectrogram_length (Tensor): spectrogram original lengths.
        Returns:
            output (dict): output dict containing logits and
                transformed lengths.
        """
        output = self.net["conv"](spectrogram.unsqueeze(1))
        output = output.flatten(1, 2).transpose(1, 2)
        logits_length = self.transform_input_lengths(spectrogram_length)
        output = self.net["rnn"](output, logits_length)
        logits = self.net["fc"](output)
        return {"logits": logits, "logits_length": logits_length}

    def transform_input_lengths(self, input_lengths):
        """
        As the network may compress the Time dimension, we need to know
        what are the new temporal lengths after compression.

        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            output_lengths (Tensor): new temporal lengths
        """
        return (input_lengths + 1) // 2
    
    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
