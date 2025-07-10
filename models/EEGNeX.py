# EEGNeX.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNeX_8_32(nn.Module):
    """
    PyTorch version of EEGNeX_8_32 (Chen et al. 2022).

    Args:
        n_timesteps  (int): number of time points in each trial/window
        n_features   (int): number of EEG channels (e.g. 22 or 3)
        n_outputs    (int): number of classes to predict (e.g. 4 or 2)
        dropout_p    (float, default=0.5): dropout probability after
                                             depthwise and separable blocks
        max_norm_val (float or None): max‐norm for conv weights (depthwise uses 1.0; others use 2.0).
                                      Set to None to disable weight clipping.
    Input shape:
        (batch, 1, n_features, n_timesteps)
    Output shape:
        (batch, n_outputs) softmax probabilities
    """

    def __init__(
        self,
        n_timesteps: int,
        n_features: int,
        n_outputs: int,
        dropout_p: float = 0.5,
    ):
        super().__init__()

        # ─── Block 1: Conv → BN → ELU ─────────────────────────────────────────
        # Conv2d(in=1,  out=8,  kernel=(1,32), padding=(0,16))
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=(1, 32),
            padding=(0, 16),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(8)

        # ─── Block 2: Conv → BN → ELU ─────────────────────────────────────────
        # Conv2d(in=8, out=32, kernel=(1,32), padding=(0,16))
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=32,
            kernel_size=(1, 32),
            padding=(0, 16),
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(32)

        # ─── Block 3: DepthwiseConv → BN → ELU → AvgPool → Dropout ────────────
        # DepthwiseConv2d with kernel=(n_features,1), depth_multiplier=2
        #   in_channels=32, out_channels=32*2=64, groups=32
        self.conv_dw = nn.Conv2d(
            in_channels=32,
            out_channels=32 * 2,
            kernel_size=(n_features, 1),
            groups=32,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(32 * 2)
        self.pool3 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=(0, 2))
        self.drop3 = nn.Dropout(dropout_p)

        # ─── Block 4: Dilated Conv → BN → ELU ──────────────────────────────────
        # Conv2d(in=64, out=32, kernel=(1,16), padding=(0,8), dilation=(1,2))
        self.conv3 = nn.Conv2d(
            in_channels=32 * 2,
            out_channels=32,
            kernel_size=(1, 16),
            padding=(0, 8),
            dilation=(1, 2),
            bias=False,
        )
        self.bn4 = nn.BatchNorm2d(32)

        # ─── Block 5: Dilated Conv → BN → ELU → Dropout ────────────────────────
        # Conv2d(in=32, out=8, kernel=(1,16), padding=(0,8), dilation=(1,4))
        self.conv4 = nn.Conv2d(
            in_channels=32,
            out_channels=8,
            kernel_size=(1, 16),
            padding=(0, 8),
            dilation=(1, 4),
            bias=False,
        )
        self.bn5 = nn.BatchNorm2d(8)
        self.drop5 = nn.Dropout(dropout_p)

        # ─── Classifier head ───────────────────────────────────────────────────
        # We need to know “flattened” size after Block 5. Use a dummy forward pass:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_features, n_timesteps)
            feat = self._forward_features(dummy)
            n_feats = feat.shape[1]

        self.classifier = nn.Linear(n_feats, n_outputs)

    # -------------------------------------------------------------------------
    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the stacked convolutional blocks (Blocks 1–5), then flatten.
        """
        # Block 1
        x = self.conv1(x)        # → (B,  8,  n_features,  n_timesteps)
        x = self.bn1(x)
        x = F.elu(x)

        # Block 2
        x = self.conv2(x)        # → (B, 32,  n_features,  n_timesteps)
        x = self.bn2(x)
        x = F.elu(x)

        # Block 3: Depthwise → BN → ELU → AvgPool → Dropout
        x = self.conv_dw(x)      # → (B, 64,  1,  n_timesteps)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool3(x)        # width: roughly → ceil(n_timesteps/4)
        x = self.drop3(x)

        # Block 4: Dilated Conv → BN → ELU
        x = self.conv3(x)        # → (B, 32,  1,  approx_time)
        x = self.bn4(x)
        x = F.elu(x)

        # Block 5: Dilated Conv → BN → ELU → Dropout
        x = self.conv4(x)        # → (B,  8,  1,  approx_time)
        x = self.bn5(x)
        x = F.elu(x)
        x = self.drop5(x)

        # Flatten everything except batch dimension
        return x.view(x.size(0), -1)  # → (B,  8 × 1 × approx_time)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 1, n_features, n_timesteps)
        returns: (batch, n_outputs) probabilities (softmax)
        """
        if len(x.size()) < 4:
            x = x.unsqueeze(1)
        feats = self._forward_features(x)
        logits = self.classifier(feats)
        return F.softmax(logits, dim=1)


if __name__ == "__main__":
    # Suppose you have 22 EEG channels, 1125 time‐points per trial, and 4 classes:
    n_features = 3
    n_timesteps = 1125
    n_outputs = 2
    model = EEGNeX_8_32(
        n_timesteps=n_timesteps,
        n_features=n_features,
        n_outputs=n_outputs,
        dropout_p=0.5,
    )

    # Dummy batch of 8 trials:
    x = torch.randn(8, 1, n_features, n_timesteps)
    probs = model(x)  # shape (8, 4), already softmaxed
    print("Output shape:", probs.shape)
