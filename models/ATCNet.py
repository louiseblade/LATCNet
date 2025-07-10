import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import numpy as np


class ATCNet(nn.Module):
    def __init__(self, n_classes=4, in_chans=22, n_windows=5,
                 eegn_F1=16, eegn_D=2, eegn_kernelSize=64, eegn_poolSize=7, eegn_dropout=0.3,
                 tcn_depth=2, tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3,
                 fuse='average', from_logit=False):
        super(ATCNet, self).__init__()
        self.n_classes = n_classes
        self.n_windows = n_windows
        self.fuse = fuse
        self.from_logits = from_logit
        self.attention_list = nn.ModuleList()
        self.fc_list = nn.ModuleList()

        self.F2 = eegn_F1 * eegn_D # F1 = 16, D = 2, F2 = 32

        self.conv_block = ConvBlock(F1=eegn_F1, D=eegn_D, kernLength=eegn_kernelSize, poolSize=eegn_poolSize,
                                    in_chans=in_chans, dropout=eegn_dropout)

        self.attention_list = nn.ModuleList(
            [AttentionBlock(in_shape=self.F2, head_dim=8, num_heads=2, dropout=0.5) for _ in range(n_windows)])

        self.fc_list = nn.ModuleList([nn.Linear(tcn_filters, n_classes) for _ in range(n_windows)])
        self.tcn_list = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        TCNResidualBlock(
                            in_channels=self.F2,
                            kernel_size=tcn_kernelSize,
                            n_filters=tcn_filters,
                            dropout=tcn_dropout,
                            dilation=2 ** i,
                        )
                        for i in range(tcn_depth)
                    ]
                )
                for _ in range(self.n_windows)
            ]
        )



    def forward(self, x):
        # Conv Block
        x = self.conv_block(x)  # Output shape: (batch, F2, 1, samples)


        # Sliding window mechanism
        sw_outputs = []
        for i in range(self.n_windows):

            start = i
            end = x.shape[2] - self.n_windows + i + 1

            window = x[:, :, start:end]

            window = self.attention_list[i](window)

            # TCN block
            tcn_out = self.tcn_list[i](window)

            #  take the last time step in TCN output
            tcn_out = tcn_out[:, :, -1]  # Shape: (batch, filters)

            # Fully connected layer for each window
            sw_outputs.append(self.fc_list[i](tcn_out))

        # Fuse sliding window outputs
        if self.fuse == 'average':
            out = torch.stack(sw_outputs, dim=0).mean(dim=0)

        if self.from_logits:
            return out  # Logits output
        else:
            return F.softmax(out, dim=-1)  # Softmax output

class ConvBlock(nn.Module):
    def __init__(self, F1=16, kernLength=64, poolSize=7, D=2, in_chans=22,
                 dropout=0.3):
        super(ConvBlock, self).__init__()

        F2 = F1 * D

        # First convolutional layer
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        # Depthwise convolutional layer
        self.depthwise_conv = nn.Conv2d(F1, F2, (in_chans, 1), groups=F1, padding="valid", bias=False)
        self.bn2 = nn.BatchNorm2d(F2)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(F2, F2, (1, 16), padding='same', bias=False)
        self.bn3 = nn.BatchNorm2d(F2)

        # Pooling and dropout layers
        self.avgpool1 = nn.AvgPool2d((1, 8))
        self.avgpool2 = nn.AvgPool2d((1, poolSize))
        self.dropout = nn.Dropout(dropout)

        # activation
        self.activation = nn.ELU()

    def forward(self, x):

        x = x.unsqueeze(1)

        # First block
        x = self.conv1(x)

        x = self.bn1(x)

        # Second block
        x = self.depthwise_conv(x)

        x = self.bn2(x)
        x = self.activation(x)
        x = self.avgpool1(x)
        x = self.dropout(x)

        # Third block
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.activation(x)


        x = self.avgpool2(x)
        x = self.dropout(x)

        return x[:, :, -1, :]

class MHA(nn.Module):
    def __init__(
            self,
            input_dim: int,
            head_dim: int,
            output_dim: int,
            num_heads: int,
            dropout: float = 0.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        # The total embedding dimension is the head dimension times the number of heads
        self.embed_dim = head_dim * num_heads

        # Embeddings for multi-head projections
        self.fc_q = nn.Linear(input_dim, self.embed_dim)
        self.fc_k = nn.Linear(input_dim, self.embed_dim)
        self.fc_v = nn.Linear(input_dim, self.embed_dim)

        # Output mapping
        self.fc_o = nn.Linear(self.embed_dim, output_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize weights using Glorot Uniform (Xavier Uniform) initializer
        self._reset_parameters()

    def _reset_parameters(self):

        # Glorot Uniform initialization for weight matrices
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)

        # Initialize biases to zero if they exist
        if self.fc_q.bias is not None:
            nn.init.zeros_(self.fc_q.bias)
        if self.fc_k.bias is not None:
            nn.init.zeros_(self.fc_k.bias)
        if self.fc_v.bias is not None:
            nn.init.zeros_(self.fc_v.bias)
        if self.fc_o.bias is not None:
            nn.init.zeros_(self.fc_o.bias)

    def forward(
            self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:

        assert Q.shape[-1] == K.shape[-1] == V.shape[-1] == self.input_dim

        batch_size, seq_len, _ = Q.shape

        # Embedding for multi-head projections
        Q = self.fc_q(Q)  # (B, S, D)
        K = self.fc_k(K)  # (B, S, D)
        V = self.fc_v(V)  # (B, S, D)

        # Split into num_heads
        Q_ = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K_ = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V_ = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q_, K_.transpose(-2, -1)) / np.sqrt(self.head_dim)
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        context = torch.matmul(weights, V_)  # (B, num_heads, S, head_dim)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # Final linear layer
        out = self.fc_o(context)

        return self.dropout(out)

class AttentionBlock(nn.Module):
    def __init__(
            self,
            in_shape=32,
            head_dim=8,
            num_heads=2,
            dropout=0.5,
    ):
        super().__init__()
        self.in_shape = in_shape
        self.head_dim = head_dim

        # Puts time dimension at -2 and feature dim at -1
        self.dimshuffle = Rearrange("batch C T -> batch T C")

        # Layer normalization
        self.ln = nn.LayerNorm(normalized_shape=in_shape, eps=1e-6)

        self.mha = MHA(
            input_dim=in_shape,
            head_dim=head_dim,
            output_dim=in_shape,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.drop = nn.Dropout(0.3)

    def forward(self, X):
        X = self.dimshuffle(X)
        out = self.ln(X)
        out = self.mha(out, out, out)

        out = X + self.drop(out)

        return self.dimshuffle(out)

class TCNResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        kernel_size=4,
        n_filters=32,
        dropout=0.3,
        activation: nn.Module = nn.ELU,
        dilation=1,
    ):
        super().__init__()
        self.activation = activation()
        self.dilation = dilation
        self.dropout = dropout
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.in_channels = in_channels

        self.conv1 = _CausalConv1d(
            in_channels=in_channels,
            out_channels=n_filters,
            kernel_size=kernel_size,
            dilation=dilation,
        )

        self.bn1 = nn.BatchNorm1d(n_filters)

        self.drop1 = nn.Dropout(dropout)

        self.conv2 = _CausalConv1d(
            in_channels=n_filters,
            out_channels=n_filters,
            kernel_size=kernel_size,
            dilation=dilation,
        )

        self.bn2 = nn.BatchNorm1d(n_filters)

        self.drop2 = nn.Dropout(dropout)

        # Reshape the input for the residual connection when necessary
        if in_channels != n_filters:
            self.reshaping_conv = nn.Conv1d(
                n_filters, in_channels,
                kernel_size=1,
                padding="same",
            )
        else:
            self.reshaping_conv = nn.Identity()

        # self.gate_conv = nn.Conv1d(n_filters, n_filters, kernel_size=1)
        # self.gate_conv = SoftThresholdGate(n_filters)

    def forward(self, x):
        # Dimension: (batch_size, F2, Tw)
        # ----- Double dilated convolutions -----


        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.drop2(out)

        out = self.reshaping_conv(out)

        # ----- Residual connection -----
        out = x + out

        # ----- Conv1D Gating mechanism -----

        return self.activation(out)

class _CausalConv1d(nn.Conv1d):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        **kwargs,
    ):
        assert "padding" not in kwargs, (
            "The padding parameter is controlled internally by "
            f"{type(self).__name__} class. You should not try to override this"
            " parameter."
        )

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation,
            **kwargs,
        )

    def forward(self, X):
        out = super().forward(X)
        return out[..., : -self.padding[0]]


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    # Example usage
    import random
    random.seed(42)
    torch.manual_seed(42)

    x = torch.randn(32, 22, 1125)  # Input tensor with shape (batch_size, in_chans, in_samples)
    model = ATCNet(n_classes=2, in_chans=22, from_logit=False)
    out = model(x)
