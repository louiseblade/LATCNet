import torch
import torch.nn as nn
import numpy as np
from einops.layers.torch import Rearrange

class GATCNet(nn.Module):
    def __init__(self, num_classes=4, n_windows=5, in_chans=22, n_filter=32, gate="conv1d"):
        super(GATCNet, self).__init__()

        self.conv_block = ConvBlock(in_chans=in_chans, dropout=0.3)
        self.n_windows = n_windows

        self.attention_list = nn.ModuleList(
            AttentionBlock(in_shape=n_filter, head_dim=8, num_heads=2, dropout=0.5)
            for _ in range(n_windows))

        self.tcn_list = nn.ModuleList([nn.Sequential(*[TCNResidualBlock(n_filters=n_filter, dilation=2 ** i,
                                                                        gate_type=gate)
                                                       for i in range(2)]) for _ in range(self.n_windows)])


        self.slide_out_list = nn.ModuleList([nn.Linear(n_filter, num_classes) for _ in range(n_windows)])
        self.final_activation = nn.Softmax(dim=-1)



    def forward(self, x):

        conv_block = self.conv_block(x)
        windows_features = []

        for i in range(self.n_windows):
            st = i
            end = conv_block.shape[-1] - self.n_windows + i + 1

            split_window = conv_block[:, :, st:end]
            attention_block = self.attention_list[i](split_window)

            tcn_block = self.tcn_list[i](attention_block)
            output_feats = tcn_block[:, :, -1]

            # feed forward
            fc = self.slide_out_list[i](output_feats) # [B,4]
            windows_features.append(fc)

        out = torch.stack(windows_features, dim=0).mean(dim=0)  # [B,4]

        out = self.final_activation(out)

        return out
class GraphSE(nn.Module):
    def __init__(self, in_features, out_features, adjacency_matrix):
        super().__init__()

        # adjacency_matrix expected shape [C, C] C = 22 EEG or 3 EEG electrodes (dataset IV  2a and  2b)
        self.A = nn.Parameter(adjacency_matrix, requires_grad=True)
        self.fc = nn.Linear(in_features, out_features)

        # early Squeeze&Excitation
        self.SE = SqueezeExcitation(channels=out_features, reduction=(out_features//2))

    def forward(self, x):
        """
        x: [B, C, T]
        returns: [B, out_features, T]
        """
        # Multiply over channels => [B, C, T]

        x_gn = torch.einsum("ij,bjt->bit", self.A, x)
        x_out = self.fc(x_gn.permute(0,2,1)).permute(0,2,1)
        x_out = self.SE(x_out).transpose(1, 2)

        return x_out


class ConvBlock(nn.Module):
    def __init__(self, F1=16, K1=64, K2=16, poolSize_1=8, poolSize_2=7, D=2, in_chans=22,
                 dropout=0.3):
        super().__init__()
        F2 = F1 * D
        self.register_buffer('adjacency', torch.eye(in_chans))
        self.gcn = GraphSE(in_features=in_chans, out_features=in_chans, adjacency_matrix=self.adjacency)


        # First convolutional layer
        self.conv1 = nn.Conv2d(1, F1, (1, K1), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        # Depthwise convolutional layer
        self.depthwise_conv = nn.Conv2d(F1, F2, (in_chans, 1), groups=F1, padding="valid", bias=False)
        self.bn2 = nn.BatchNorm2d(F2)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(F2, F2, (1, K2), padding='same', bias=False)
        self.bn3 = nn.BatchNorm2d(F2)

        # Late Squeeze-and-Excitation layer
        self.se = SqueezeExcitation(channels=F2, reduction=max(1, F2 // 2))

        # Pooling and dropout layers
        self.avgpool1 = nn.AvgPool2d((1, poolSize_1))
        self.avgpool2 = nn.AvgPool2d((1, poolSize_2))
        self.dropout = nn.Dropout(dropout)

        # activation
        self.activation = nn.ELU()

    def forward(self, x):

        x = self.gcn(x)

        # First blocka
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

        # squeeze and excitation
        x = self.se(x)

        x = self.avgpool2(x)
        x = self.dropout(x)

        return x[:, :, -1, :]


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=16):
        """
        Squeeze-and-Excitation Module.

        Args:
            channels (int): Number of input channels.
            reduction (int): Reduction ratio for the intermediate FC layer.
        """
        super(SqueezeExcitation, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Squeeze
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)  # Excitation
        self.relu = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels = x.size(0), x.size(1)
        # Squeeze: Global Average Pooling
        if len(x.shape) == 3:
            x = x.unsqueeze(2)

        y = self.global_avg_pool(x).view(batch_size, channels)

        # Excitation: FC -> ReLU -> FC -> Sigmoid
        y = self.fc1(y)

        y = self.relu(y)
        y = self.fc2(y)

        y = self.sigmoid(y).view(batch_size, channels, 1, 1)

        # Scale: Channel-wise multiplication
        return x * y.expand_as(x)
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

# -------------------------------------------------------------------
#  Gate options
# -------------------------------------------------------------------
class ScalarGate(nn.Module):
    def __init__(self, init=0.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(init))

    def forward(self, y):
        return torch.sigmoid(self.alpha) * y


class GRUGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gru  = nn.GRU(channels, channels, batch_first=True)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x):                      # x: (B,C,T)
        seq, _ = self.gru(x.permute(0, 2, 1))  # (B,T,C)
        g      = torch.sigmoid(self.proj(seq)) # (B,T,C)
        return g.permute(0, 2, 1)              # (B,C,T)


# -------------------------------------------------------------------
#  Residual block with selectable gate
# -------------------------------------------------------------------
class TCNResidualBlock(nn.Module):
    """
    gate_type ∈ {'conv1d', 'swish', 'scalar', 'gru'}
        1. conv1d : 1×1 Conv + Sigmoid gate
        2. swish  : σ(out) * out       (no extra params)
        3. scalar : learnable scalar   (1 param)
        4. gru    : GRU-based gate     (time-aware)
    """
    def __init__(self,
                 n_filters: int = 32,
                 kernel_size: int = 4,
                 dropout: float = 0.3,
                 activation: nn.Module = nn.ELU,
                 dilation: int = 1,
                 gate_type: str = "conv1d"):
        super().__init__()
        self.act = activation()
        self.gate_type = gate_type.lower()

        # ---- double causal conv ----
        self.conv1 = CausalConv1d(n_filters, n_filters,
                                  kernel_size, dilation)
        self.bn1   = nn.BatchNorm1d(n_filters)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(n_filters, n_filters,
                                  kernel_size, dilation)
        self.bn2   = nn.BatchNorm1d(n_filters)
        self.drop2 = nn.Dropout(dropout)

        # ---- gate modules  ----
        if self.gate_type == "conv1d":
            self.gate = nn.Conv1d(n_filters, n_filters, kernel_size=1)
        elif self.gate_type == "scalar":
            self.gate = ScalarGate()
        elif self.gate_type == "gru":
            self.gate = GRUGate(n_filters)
        elif self.gate_type == "swish":
            self.gate = None                     # handled inline
        else:
            pass                                 # no gate

    # ----------------------------------------------------------------
    def forward(self, x):                        # x: (B,C,T)
        res = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)
        out = self.drop2(out)

        # -------- gating --------
        if self.gate_type == "conv1d":
            gate = torch.sigmoid(self.gate(out))
            out  = res + gate * out
        elif self.gate_type == "swish":          # σ(out) * out
            out  = res + torch.sigmoid(out) * out
        elif self.gate_type == "scalar":
            out  = res + self.gate(out)          # ScalarGate returns σ(α)·out
        elif self.gate_type == "gru":
            gate = self.gate(out)                # (B,C,T) ∈ (0,1)
            out  = res + gate * out
        else:
            out = res + out

        return self.act(out)


class CausalConv1d(nn.Conv1d):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            dilation=1,
            **kwargs,
    ):
        assert "padding" not in kwargs, (
            "padding is "
            f"{type(self).__name__} class. You should not try to override this"
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

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # Example usage
    import random
    from weight_init import *
    random.seed(42)
    torch.manual_seed(42)

    x = torch.randn(32, 22, 1125)

    model = GATCNet(num_classes=4, n_windows=5, in_chans=x.size(1))
    model.apply(initialize_weights_keras_style)

    output = model(x)
    # check_requires_grad(model)
    print("Output shape:", output.shape)
