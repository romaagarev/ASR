import torch
from torch import nn
from torch.nn import Sequential
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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
                    nn.BatchNorm1d(32),
                    nn.Hardtanh(min_val=0, max_val=20),
                    nn.Conv2d(
                        in_channels=32,
                        out_channels=32,
                        kernel_size=(21, 11),
                        stride=(2, 1),
                        padding=(10, 5),
                    ),
                    nn.BatchNorm1d(32),
                    nn.Hardtanh(min_val=0, max_val=20),
                    nn.Conv2d(
                        in_channels=32,
                        out_channels=96,
                        kernel_size=(21, 11),
                        stride=(2, 1),
                        padding=(10, 5),
                    ),
                    nn.BatchNorm1d(96),
                    nn.Hardtanh(min_val=0, max_val=20),
                ),
                "rnn": nn.RNN(
                    input_size=96 * ((n_feats + 7) // 8),
                    hidden_size=fc_hidden,
                    num_layers=7,
                    nonlinearity="relu",
                    batch_first=True,
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
        output = spectrogram.unsqueeze(1)
        lengths = spectrogram_length.to(output.device)
        mask = torch.arange(output.shape[-1], device=output.device) < lengths[:, None]
        output = output.masked_fill(~mask[:, None, None, :], 0)

        for layer in self.net["conv"]:
            if isinstance(layer, nn.Conv2d):
                output = layer(output)
                stride = layer.stride[1]
                lengths = (lengths + stride - 1) // stride
                mask = (
                    torch.arange(output.shape[-1], device=output.device)
                    < lengths[:, None]
                )
                output = output.masked_fill(~mask[:, None, None, :], 0)
            elif isinstance(layer, nn.BatchNorm1d):
                output = output.permute(0, 2, 3, 1)
                valid = mask[:, None, :].expand(-1, output.shape[1], -1)
                normalized = torch.zeros_like(output)
                normalized[valid] = layer(output[valid])
                output = normalized.permute(0, 3, 1, 2)
            else:
                output = layer(output)

        output = output.flatten(1, 2).transpose(1, 2)
        batch_size, time_steps, _ = output.shape
        logits_length = self.transform_input_lengths(spectrogram_length)
        output = pack_padded_sequence(
            output, logits_length.cpu(), batch_first=True, enforce_sorted=False
        )
        output, _ = self.net["rnn"](output)
        output, _ = pad_packed_sequence(
            output, batch_first=True, total_length=time_steps
        )
        output = output.reshape(batch_size, time_steps, 2, -1).sum(dim=2)
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
