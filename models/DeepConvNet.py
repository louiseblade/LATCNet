# deep_convnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────── custom activations ──────────────────────────────
class Square(nn.Module):        # (not used here, but handy for other EEG nets)
    def forward(self, x):
        return x ** 2


class SafeLog(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return torch.log(torch.clamp(x, min=self.eps))


# ───────────────────────────── DeepConvNet ───────────────────────────────────
class DeepConvNet(nn.Module):
    """
    PyTorch implementation of DeepConvNet (Schirrmeister et al., 2017).

    Parameters
    ----------
    chans         : int   – number of EEG channels (22 *or* 3)
    n_classes     : int   – target classes (4 *or* 2)
    samples       : int   – # time-points per trial (default = 1 125)
    dropout_p     : float – dropout probability

    """

    def __init__(
        self,
        chans: int,
        n_classes: int,
        samples: int = 1125,
        dropout_p: float = 0.5,
    ):
        super().__init__()

        # ─── block 1 ────────────────────────────────────────────────────────
        # temporal conv
        self.conv1_time = nn.Conv2d(1, 25, kernel_size=(1, 10), bias=True)
        # spatial conv (across channels) – depth-wise
        self.conv1_spat = nn.Conv2d(
            25, 25, kernel_size=(chans, 1), bias=False, groups=25
        )
        self.bn1   = nn.BatchNorm2d(25, eps=1e-5, momentum=0.9)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.drop1 = nn.Dropout(dropout_p)

        # ─── block 2 ────────────────────────────────────────────────────────
        self.conv2 = nn.Conv2d(25, 50, kernel_size=(1, 10), bias=True)
        self.bn2   = nn.BatchNorm2d(50, eps=1e-5, momentum=0.9)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.drop2 = nn.Dropout(dropout_p)

        # ─── block 3 ────────────────────────────────────────────────────────
        self.conv3 = nn.Conv2d(50, 100, kernel_size=(1, 10), bias=True)
        self.bn3   = nn.BatchNorm2d(100, eps=1e-5, momentum=0.9)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.drop3 = nn.Dropout(dropout_p)

        # ─── block 4 ────────────────────────────────────────────────────────
        self.conv4 = nn.Conv2d(100, 200, kernel_size=(1, 10), bias=True)
        self.bn4   = nn.BatchNorm2d(200, eps=1e-5, momentum=0.9)
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.drop4 = nn.Dropout(dropout_p)

        # ─── classifier head ───────────────────────────────────────────────
        with torch.no_grad():
            dummy = torch.zeros(1, 1, chans, samples)
            n_feats = self._extract_features(dummy).shape[1]
        self.classifier = nn.Linear(n_feats, n_classes)

    # ─────────────────────────── internal helpers ────────────────────────────
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:

        # block 1
        x = F.elu(self.bn1(self.conv1_spat(self.conv1_time(x))))
        x = self.pool1(x)
        x = self.drop1(x)
        # block 2
        x = F.elu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.drop2(x)
        # block 3
        x = F.elu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.drop3(x)
        # block 4
        x = F.elu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = self.drop4(x)
        return x.flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.size()) < 4:
            x = x.unsqueeze(1)
        logits = self.classifier(self._extract_features(x))
        # Keep soft-max inside for probability output; remove if you train with CrossEntropyLoss
        return F.softmax(logits, dim=1)


# ───────────────────────── example usage ─────────────────────────────────────
if __name__ == "__main__":
    # 22-channel, 4-class model
    model = DeepConvNet(chans=22, n_classes=4)
    x = torch.randn(5, 1, 22, 1125)          # batch of 5 trials
    probs = model(x)                           # (5, 4)
    print("Output shape:", probs.shape)

    # 3-channel, 2-class model
    model_small = DeepConvNet(chans=3, n_classes=2)
    y = torch.randn(8, 1, 3, 1125)
    print("Small-net output:", model_small(y).shape)
