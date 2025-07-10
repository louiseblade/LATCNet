# eeg_tcnet_axis_fixed.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Helper: causal padding + “chomp” to keep sequence length unchanged
# ---------------------------------------------------------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp):
        super().__init__()
        self.chomp = chomp

    def forward(self, x):
        return x[..., :-self.chomp] if self.chomp > 0 else x


# ---------------------------------------------------------------------
# 1) EEGNet (channels-last inside, channels-first outside)
# ---------------------------------------------------------------------
class EEGNetBackbone(nn.Module):
    """
    Returns a tensor of shape (B, F2, T) where F2 = F1*D
    """
    def __init__(self, chans, F1=8, D=2, kern_len=64, drop_p=0.2, act='elu'):
        super().__init__()
        F2 = F1 * D
        self.act = nn.ELU() if act.lower() == 'elu' else nn.ReLU()

        # temporal conv (along Samples dimension after permute)
        self.conv_t = nn.Conv2d(1, F1, (kern_len, 1),
                                padding=(kern_len // 2, 0), bias=False)
        self.bn_t   = nn.BatchNorm2d(F1)

        # depth-wise spatial conv (collapse Chans)
        self.conv_dw = nn.Conv2d(F1, F2, (1, chans),
                                 groups=F1, bias=False)
        self.bn_dw   = nn.BatchNorm2d(F2)
        self.pool_dw = nn.AvgPool2d((8, 1))
        self.drop_dw = nn.Dropout(drop_p)

        # separable conv: depth-wise (time) + point-wise
        self.conv_sep_dw = nn.Conv2d(F2, F2, (16, 1),
                                     padding=(8, 0), groups=F2, bias=False)
        self.conv_sep_pw = nn.Conv2d(F2, F2, 1, bias=False)
        self.bn_sep = nn.BatchNorm2d(F2)
        self.pool_sp = nn.AvgPool2d((8, 1))
        self.drop_sp = nn.Dropout(drop_p)

    def forward(self, x):                       # x: (B,1,Chans,Samples)
        if len(x.size()) < 4:
            x= x.unsqueeze(1)
        x = x.permute(0, 1, 3, 2)               # → (B,1,Samples,Chans)

        x = self.act(self.bn_t(self.conv_t(x)))
        x = self.act(self.bn_dw(self.conv_dw(x)))
        x = self.pool_dw(x)
        x = self.drop_dw(x)

        x = self.conv_sep_dw(x)
        x = self.conv_sep_pw(x)
        x = self.act(self.bn_sep(x))
        x = self.pool_sp(x)
        x = self.drop_sp(x)                     # (B, F2, T, 1)

        x = x.squeeze(3)                        # (B, F2, T)
        return x


# ---------------------------------------------------------------------
# 2) Temporal Convolutional Network (TCN) residual stack
# ---------------------------------------------------------------------
class _TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, dil, p_drop, act):
        super().__init__()
        pad = (k - 1) * dil
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=dil, bias=False),
            Chomp1d(pad),
            act,
            nn.Dropout(p_drop),
            nn.Conv1d(out_ch, out_ch, k, padding=pad, dilation=dil, bias=False),
            Chomp1d(pad),
            act,
            nn.Dropout(p_drop),
        )
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = act

    def forward(self, x):
        return self.act(self.net(x) + self.down(x))


class TCNStack(nn.Module):
    def __init__(self, in_dim, depth, k, filt, p_drop, activation='elu'):
        super().__init__()
        act = nn.ELU() if activation.lower() == 'elu' else nn.ReLU()
        layers = []
        for i in range(depth):
            dil = 2 ** i
            layers.append(
                _TCNBlock(in_dim if i == 0 else filt, filt, k, dil, p_drop, act)
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x):          # (B, F2, T) → (B, filt, T)
        return self.net(x)


# ---------------------------------------------------------------------
# 3) EEG-TCNet (axis-correct)
# ---------------------------------------------------------------------
class EEGTCNet(nn.Module):
    """
    Final output: (B, n_classes) soft-max probabilities
    """
    def __init__(self,
                 n_classes,
                 Chans=22,
                 layers=2,
                 kernel_s=4,
                 filt=12,
                 dropout=0.3,
                 activation='elu',
                 F1=8,
                 D=2,
                 kernLength=32,
                 dropout_eeg=0.2,
                 ):
        super().__init__()

        # EEGNet backbone
        self.backbone = EEGNetBackbone(
            chans=Chans,
            F1=F1,
            D=D,
            kern_len=kernLength,
            drop_p=dropout_eeg,
            act=activation,
        )
        F2 = F1 * D

        # TCN stack
        self.tcn = TCNStack(
            in_dim=F2,
            depth=layers,
            k=kernel_s,
            filt=filt,
            p_drop=dropout,
            activation=activation,
        )

        # Dense
        self.classifier = nn.Linear(filt, n_classes)

    # -------------------------------------------------
    def forward(self, x):                 # x: (B,1,Chans,Samples)
        feats = self.backbone(x)          # (B, F2, T)
        outs  = self.tcn(feats)           # (B, filt, T)
        last  = outs[:, :, -1]            # <-- last time step (axis-correct!)
        logits = self.classifier(last)
        return F.softmax(logits, dim=1)


# ---------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------
if __name__ == "__main__":
    model = EEGTCNet(n_classes=4, Chans=22, Samples=1125)
    dummy = torch.randn(8, 1, 22, 1125)
    print("Output shape:", model(dummy).shape)   # torch.Size([8, 4])

    small = EEGTCNet(n_classes=2, Chans=3, Samples=1125)
    print("Small-net  :", small(torch.randn(5, 1, 3, 1125)).shape)  # [5,2]
