# tcnet_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable


# ---------------------------------------------------------------------
# EEGNet feature extractor (compact; outputs (B, F2, T_eeg))
# ---------------------------------------------------------------------
class EEGNetBackbone(nn.Module):
    def __init__(
        self,
        chans: int,
        samples: int,
        F1: int = 24,
        D: int = 2,
        kern_length: int = 32,
        dropout_p: float = 0.3,
    ):
        super().__init__()
        F2 = F1 * D
        self.conv_t = nn.Conv2d(1, F1, (1, kern_length),
                                padding=(0, kern_length // 2), bias=False)
        self.bn_t   = nn.BatchNorm2d(F1)

        self.conv_dw = nn.Conv2d(F1, F2, (chans, 1),
                                 groups=F1, bias=False)
        self.bn_dw   = nn.BatchNorm2d(F2)
        self.pool_dw = nn.AvgPool2d((1, 4), stride=(1, 4), padding=(0, 2))
        self.drop_dw = nn.Dropout(dropout_p)

        self.conv_sep_dw = nn.Conv2d(F2, F2, (1, 16),
                                     padding=(0, 8), dilation=(1, 2),
                                     groups=F2, bias=False)
        self.conv_sep_pw = nn.Conv2d(F2, F2, 1, bias=False)
        self.bn_sep  = nn.BatchNorm2d(F2)
        self.pool_sp = nn.AvgPool2d((1, 4), stride=(1, 4), padding=(0, 2))
        self.drop_sp = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,1,C,T)
        x = F.elu(self.bn_t(self.conv_t(x)))
        x = F.elu(self.bn_dw(self.conv_dw(x)))
        x = self.pool_dw(x)
        x = self.drop_dw(x)
        x = self.conv_sep_dw(x)
        x = self.conv_sep_pw(x)
        x = F.elu(self.bn_sep(x))
        x = self.pool_sp(x)
        x = self.drop_sp(x)
        return x.squeeze(2)                              # (B, F2, T_eeg)



# ---------------------------------------------------------------------
# Single residual TCN block
# ---------------------------------------------------------------------
class _TemporalBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int,
        dilation: int,
        p_drop: float,
        act: Callable[[torch.Tensor], torch.Tensor],
    ):
        super().__init__()
        pad = (k - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, k,
                               padding=pad, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k,
                               padding=pad, dilation=dilation)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(p_drop)
        self.down  = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )
        self.act = act

    def _crop(self, x, L):
        return x[..., :L]

    def forward(self, x):
        L = x.size(-1)
        y = self.drop(self.act(self.bn1(self._crop(self.conv1(x), L))))
        y = self.drop(self.act(self.bn2(self._crop(self.conv2(y), L))))
        return self.act(y + self.down(self._crop(x, L)))


class TCNBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        depth: int,
        k: int,
        filt: int,
        dropout: float,
        activation: str = "elu",
    ):
        super().__init__()
        act = F.elu if activation.lower() == "elu" else F.relu
        layers = []
        for i in range(depth):
            dil = 2 ** i
            layers.append(
                _TemporalBlock(
                    in_dim if i == 0 else filt, filt, k, dil, dropout, act
                )
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # (B, in_dim, T) ➜ (B, filt, T)
        return self.net(x)


# ---------------------------------------------------------------------
# TCNet-Fusion – main model
# ---------------------------------------------------------------------
class TCNetFusion(nn.Module):
    def __init__(
        self,
        n_classes: int,
        Chans: int = 22,
        Samples: int = 1125,
        layers: int = 2,
        kernel_s: int = 4,
        filt: int = 12,
        dropout: float = 0.3,
        activation: str = "elu",
        F1: int = 24,
        D: int = 2,
        kernLength: int = 32,
        dropout_eeg: float = 0.3,
    ):
        super().__init__()

        # 1) EEGNet backbone
        self.backbone = EEGNetBackbone(
            chans=Chans,
            samples=Samples,
            F1=F1,
            D=D,
            kern_length=kernLength,
            dropout_p=dropout_eeg,
        )
        F2 = F1 * D

        # 2) TCN block
        self.tcn = TCNBlock(
            in_dim=F2,
            depth=layers,
            k=kernel_s,
            filt=filt,
            dropout=dropout,
            activation=activation,
        )

        # 3) Build classifier dynamically ----------------------------------
        with torch.no_grad():
            dummy = torch.zeros(1, 1, Chans, Samples)
            fusion_vec = self._fusion_vector(dummy)
            self.classifier = nn.Linear(fusion_vec.size(1), n_classes)




    # ------------------------------------------------------------------
    def _fusion_vector(self, x: torch.Tensor) -> torch.Tensor:
        """
        Constructs the exact fusion feature vector used for classification,
        WITHOUT allocating new tensors in forward().
        """
        feats = self.backbone(x)                 # (B, F2, T)
        last  = feats[:, :, -1]                  # (B, F2)
        tcn   = self.tcn(feats)                  # (B, filt, T)
        concat1 = torch.cat([feats, tcn], dim=1) # (B, F2+filt, T)
        flat1   = concat1.flatten(start_dim=1)   # (B, (F2+filt)*T)
        return torch.cat([flat1, last], dim=1)   # (B, (F2+filt)*T + F2)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.size()) < 4:
            x = x.unsqueeze(1)
        fusion = self._fusion_vector(x)
        logits = self.classifier(fusion)
        return F.softmax(logits, dim=1)          # → (B, n_classes)

# ---------------------------------------------------------------------
# quick test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    model = TCNetFusion(n_classes=4)               # 22-ch, 4-class
    dummy = torch.randn(3, 1, 22, 1125)
    print("Output:", model(dummy).shape)           # torch.Size([3, 4])

    small = TCNetFusion(n_classes=2, Chans=3)      # 3-ch, 2-class
    print("Small:", small(torch.randn(5, 1, 3, 1125)).shape)
