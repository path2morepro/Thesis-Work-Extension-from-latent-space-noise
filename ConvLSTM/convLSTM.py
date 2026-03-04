import torch
import torch.nn as nn

# the purpose of this model is to examine whether the latent space can be forcasted
class ConvLSTMCell(nn.Module):
    """A small ConvLSTM cell for 2D spatiotemporal forecasting."""

    def __init__(self, input_channels, hidden_channels, kernel_size=3, bias=True):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(self, x, state):
        h_prev, c_prev = state
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.gates(combined)
        i, f, o, g = torch.chunk(gates, chunks=4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_state(self, batch_size, spatial_size, device=None, dtype=None):
        height, width = spatial_size
        shape = (batch_size, self.hidden_channels, height, width)
        h = torch.zeros(shape, device=device, dtype=dtype)
        c = torch.zeros(shape, device=device, dtype=dtype)
        return h, c


class ConvLSTMForecaster(nn.Module):
    """
    Small ConvLSTM forecaster for sequences of 64x64 single-channel frames.

    Expected inputs:
      - [B, T, 1, 64, 64]
      - [B, T, 64, 64] for single-channel inputs

    Expected output:
      - [B, 1, 64, 64]
    """

    def __init__(
        self,
        input_channels=1,
        hidden_channels=(16, 16),
        kernel_size=3,
        input_frames=10,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.input_frames = input_frames
        self.hidden_channels = tuple(hidden_channels)

        cells = []
        in_ch = input_channels
        for hidden_ch in self.hidden_channels:
            cells.append(
                ConvLSTMCell(
                    input_channels=in_ch,
                    hidden_channels=hidden_ch,
                    kernel_size=kernel_size,
                )
            )
            in_ch = hidden_ch
        self.cells = nn.ModuleList(cells)

        self.head = nn.Sequential(
            nn.Conv2d(self.hidden_channels[-1], self.hidden_channels[-1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_channels[-1], input_channels, kernel_size=1),
        )

    def forward(self, x):
        if x.ndim == 4:
            x = x.unsqueeze(2)
        if x.ndim != 5:
            raise ValueError(
                f"Expected input shape [B, T, C, H, W] or [B, T, H, W], got {tuple(x.shape)}"
            )

        batch_size, seq_len, channels, height, width = x.shape
        if channels != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channel(s), got {channels}"
            )
        if seq_len != self.input_frames:
            raise ValueError(
                f"Expected {self.input_frames} input frames, got {seq_len}"
            )

        states = [
            cell.init_state(
                batch_size=batch_size,
                spatial_size=(height, width),
                device=x.device,
                dtype=x.dtype,
            )
            for cell in self.cells
        ]

        for t in range(seq_len):
            layer_input = x[:, t]
            for idx, cell in enumerate(self.cells):
                states[idx] = cell(layer_input, states[idx])
                layer_input = states[idx][0]

        h_last = states[-1][0]
        return self.head(h_last)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# sanity check code
# if __name__ == "__main__":
#     model = ConvLSTMForecaster(input_channels=1, hidden_channels=(16, 16), input_frames=10)
#     x = torch.randn(4, 10, 1, 64, 64)
#     y = model(x)
#     print("input shape:", tuple(x.shape))
#     print("output shape:", tuple(y.shape))
#     print("trainable params:", count_parameters(model))
