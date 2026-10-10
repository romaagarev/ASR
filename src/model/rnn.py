import torch
from torch import nn


class RNN(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bidirectional=False,
        min_val=0,
        max_val=20,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.activation = nn.Hardtanh(min_val=min_val, max_val=max_val)

        self.input_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.recurrent_layers = nn.ModuleList()
        self.backward_recurrent_layers = nn.ModuleList()

        for layer_index in range(num_layers):
            layer_input_size = input_size if layer_index == 0 else hidden_size
            self.input_layers.append(
                nn.Linear(layer_input_size, hidden_size, bias=False)
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_size))
            self.recurrent_layers.append(
                nn.Linear(hidden_size, hidden_size, bias=False)
            )
            if bidirectional:
                self.backward_recurrent_layers.append(
                    nn.Linear(hidden_size, hidden_size, bias=False)
                )

    def _run_direction(self, normalized_input, mask, recurrent_layer, reverse=False):
        batch_size, time_steps, _ = normalized_input.shape
        hidden = normalized_input.new_zeros(batch_size, self.hidden_size)
        outputs = normalized_input.new_zeros(batch_size, time_steps, self.hidden_size)
        steps = range(time_steps) if not reverse else range(time_steps - 1, -1, -1)

        for step in steps:
            active = mask[:, step].unsqueeze(1)
            next_hidden = self.activation(
                normalized_input[:, step] + recurrent_layer(hidden)
            )
            hidden = torch.where(active, next_hidden, hidden)
            outputs[:, step] = torch.where(active, hidden, torch.zeros_like(hidden))

        return outputs

    def forward(self, x, lengths=None):
        batch_size, time_steps, _ = x.shape
        if lengths is None:
            lengths = torch.full(
                (batch_size,), time_steps, dtype=torch.long, device=x.device
            )
        else:
            lengths = torch.as_tensor(lengths, device=x.device)

        time_indices = torch.arange(time_steps, device=x.device)
        mask = time_indices < lengths.unsqueeze(1)

        for layer_index in range(self.num_layers):
            normalized = self.input_layers[layer_index](x)
            normalized[mask] = self.batch_norms[layer_index](normalized[mask])
            
            x = self._run_direction(
                normalized, mask, self.recurrent_layers[layer_index]
            )

            if self.bidirectional:
                backward_output = self._run_direction(
                    normalized,
                    mask,
                    self.backward_recurrent_layers[layer_index],
                    reverse=True,
                )
                x = x + backward_output

        return x
