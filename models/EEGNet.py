# eegnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNetClassifier(nn.Module):
    """
    PyTorch version of EEGNet (v2.0) from Lawhern et al., 2018.

    Parameters
    ----------
    n_classes    : int   – number of output classes (4 or 2)
    chans        : int   – EEG channels (22 or 3)
    samples      : int   – time samples per trial (default 1125)
    F1           : int   – temporal filter count (default 8)
    D            : int   – depth-wise multiplier (default 2)
    kern_length  : int   – temporal kernel length (default 64)
    dropout_p    : float – dropout probability (default 0.25)
    max_norm_val : float | None – L2 max-norm for conv weights (2.0) and
                                  classifier weights (0.5).  Set None to disable.
    """

    def __init__(
        self,
        n_classes: int,
        chans: int = 22,
        samples: int = 1125,
        F1: int = 8,
        D: int = 2,
        kern_length: int = 64,
        dropout_p: float = 0.25,
    ):
        super().__init__()

        F2 = F1 * D                          # number of point-wise filters

        # ─── Block 1: temporal conv ────────────────────────────────────────
        self.conv_temporal = nn.Conv2d(
            1,
            F1,
            kernel_size=(1, kern_length),
            padding=(0, kern_length // 2),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(F1, eps=1e-5, momentum=0.9)

        # ─── Block 2: depth-wise spatial filtering ─────────────────────────
        self.conv_spatial = nn.Conv2d(
            F1,
            F1 * D,
            kernel_size=(chans, 1),
            groups=F1,            # depth-wise
            bias=False,
        )
        self.bn2   = nn.BatchNorm2d(F1 * D, eps=1e-5, momentum=0.9)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop2 = nn.Dropout(dropout_p)

        # ─── Block 3: separable conv (depth-wise + point-wise) ────────────
        # depth-wise part
        self.conv_dw3 = nn.Conv2d(
            F1 * D,
            F1 * D,
            kernel_size=(1, 16),
            padding=(0, 8),
            groups=F1 * D,
            bias=False,
        )
        # point-wise part
        self.conv_pw3 = nn.Conv2d(
            F1 * D,
            F2,
            kernel_size=1,
            bias=False,
        )
        self.bn3   = nn.BatchNorm2d(F2, eps=1e-5, momentum=0.9)
        self.pool3 = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop3 = nn.Dropout(dropout_p)

        # ─── Classifier head ───────────────────────────────────────────────
        with torch.no_grad():
            dummy = torch.zeros(1, 1, chans, samples)
            n_feats = self._extract_features(dummy).shape[1]
        self.classifier = nn.Linear(n_feats, n_classes)

    # ---------------------------------------------------------------------
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.bn1(self.conv_temporal(x))
        # Block 2
        x = F.elu(self.bn2(self.conv_spatial(x)))
        x = self.pool2(x)
        x = self.drop2(x)
        # Block 3
        x = self.conv_dw3(x)
        x = self.conv_pw3(x)
        x = F.elu(self.bn3(x))
        x = self.pool3(x)
        x = self.drop3(x)
        return x.flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.size()) < 4:
            x = x.unsqueeze(1)
        logits = self.classifier(self._extract_features(x))
        return F.softmax(logits, dim=1)

# ────────────────────────────── usage demo ───────────────────────────────────
if __name__ == "__main__":
    # 22-channel, 4-class model
    net = EEGNetClassifier(n_classes=4, chans=22)
    x = torch.randn(10, 1, 22, 1125)          # (batch, 1, C, T)
    print("probabilities:", net(x).shape)     # (10, 4)

    # 3-channel, 2-class model
    net_small = EEGNetClassifier(n_classes=2, chans=3)
    y = torch.randn(16, 1, 3, 1125)
    print("small net output:", net_small(y).shape)  # (16, 2)
