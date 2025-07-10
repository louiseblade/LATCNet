import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------
# custom nonlinearities from the original paper
# -------------------------------------------------
class Square(nn.Module):          # x → x²
    def forward(self, x):
        return x ** 2


class SafeLog(nn.Module):         # log(max(x, eps))
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return torch.log(torch.clamp(x, min=self.eps))


# -------------------------------------------------
# the network
# -------------------------------------------------
class ShallowConvNet(nn.Module):
    """
    ShallowConvNet re-implementation (PyTorch).

    Parameters
    ----------
    chans       : int   – number of EEG channels (22 **or** 3)
    n_classes   : int   – output classes  (4 **or** 2)
    samples     : int   – number of time points (fixed to 1 125 by default)
    dropout_p   : float – dropout probability (default .5)

    """

    def __init__(
        self,
        chans: int,
        n_classes: int,
        samples: int = 1_125,
        dropout_p: float = 0.5,
    ):
        super().__init__()

        # ─── feature extractor ────────────────────────────────────────────
        self.conv_time = nn.Conv2d(
            in_channels=1,
            out_channels=40,
            kernel_size=(1, 25),
            bias=True,
        )
        # depth-wise spatial conv (groups = out_channels)
        self.conv_spat = nn.Conv2d(
            in_channels=40,
            out_channels=40,
            kernel_size=(chans, 1),
            bias=False,
            groups=40,
        )

        self.batch_norm = nn.BatchNorm2d(40, eps=1e-5, momentum=0.9)
        self.square     = Square()
        self.avg_pool   = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        self.safe_log   = SafeLog()
        self.dropout    = nn.Dropout(dropout_p)

        # ─── classifier ───────────────────────────────────────────────────
        with torch.no_grad():                   # infer flattened size
            dummy = torch.zeros(1, 1, chans, samples)
            n_feats = self._features(dummy).shape[1]

        self.classifier = nn.Linear(n_feats, n_classes)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = self.batch_norm(x)
        x = self.square(x)
        x = self.avg_pool(x)
        x = self.safe_log(x)
        x = self.dropout(x)
        return x.flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.size()) < 4:
            x = x.unsqueeze(1)
        return F.softmax(self.classifier(self._features(x)), dim=1)



# ───────────────────────────────────────────────────────────────────────────────
# example usage
# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # --- 22-channel, 4-class version
    model_22 = ShallowConvNet(chans=22, n_classes=4)
    x_22 = torch.randn(8, 1, 22, 1125)          # (batch, 1, channels, samples)
    print("22-ch output →", model_22(x_22).shape)  # (8, 4)

    # --- 3-channel, 2-class version
    model_3 = ShallowConvNet(chans=3, n_classes=2)
    x_3 = torch.randn(16, 1, 3, 1125)
    print("3-ch output  →", model_3(x_3).shape)    # (16, 2)
