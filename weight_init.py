import torch.nn as nn

def initialize_weights_keras_style(m, verbose=False):
    # TCN blocks: "He Uniform"
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        if verbose:
            print("TCN Conv1d (He Uniform):", m)

    # Initial conv2d or other conv layers (default Keras => Glorot Uniform)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)  # replicate default Glorot
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        if verbose:
            print("Conv2d (Glorot Uniform):", m)

    elif isinstance(m, nn.Linear):
        # replicate default Glorot uniform for Dense
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        if verbose:
            print("Linear (Glorot Uniform):", m)

    elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
        if verbose:
            print("BatchNorm:", m)