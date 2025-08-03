# LATCNet – EEG Motor‑Imagery Benchmarks

This repository contains **LATCNet**—a novel network—together with a training pipeline and several baseline CNN/TCN architectures for the BCI Competition IV 2a & 2b motor‑imagery datasets.

---
## Directory layout

```
GITHUB_LATCNET/
├── BCICIV2a/            # original .gdf exported to .mat (if you have them)
│   ├── T/               # training sessions  (A01T.mat … A09T.mat)
│   └── E/               # evaluation sessions (A01E.mat … A09E.mat)
├── BCICIV2a_npz/        # *fallback* pre‑extracted NumPy archives
│   ├── data_npz/        #   └── A01T.npz …
│   └── true_labels/     # labels in the same order
├── BCICIV2b/            # dataset 2b (.mat files organised the same way)
│   ├── T/
│   └── E/
├── models/              # model definitions (EEGNet, DeepConvNet, …, LATCNet)
├── data_utils.py        # data‑loading helpers
├── train_val.py         # **main training / evaluation script**
└── …
```

If you cannot download the original gdf files from BNCI‑Horizon, simply keep the provided `BCICIV2a_npz` folder and run with `--npz-data True` .
Or download the npz from this repo : https://github.com/bregydoc/bcidatasetIV2a
---
## Quick start

1. **Create a Conda/virtualenv and install requirements**

   ```bash
   conda create -n latcnet python=3.10
   conda activate latcnet
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # or cpu version
   pip install numpy scikit-learn matplotlib  # plus any extras you need
   ```

2. **Train & test LATCNET on dataset 2a in leave‑one‑subject‑out mode**

   ```bash
   python train_val.py \
          --dataset a \
          --model GATCNet \
          --training-mode loso \
          --epochs 600 \
          --batch-size 64 \
          --seeds 42 43 44
   ```


3. **Monitor progress** – live metrics are printed to console and duplicated to `training.log`.

---
## Command‑line interface (excerpt)

| Flag | Purpose |
|------|---------|
| `--dataset {a,b}` | Choose Dataset 2a (4‑class) or 2b (2‑class) |
| `--model` | One of the architectures registered in **MODEL_ZOO** fileciteturn6file13L21-L33 |
| `--training-mode {within,loso}` | Train per‑subject or leave‑one‑subject‑out |
| `--validation-mode {train_val_test,train_test_only}` | Split or reuse test‑set for validation fileciteturn6file6L24-L31 |
| `--model-selection {best_val_loss,early_stopping,final_epoch}` | Choose checkpointing strategy |
| `--early-stopping-patience` | Patience epochs when early stopping is active fileciteturn6file3L10-L18 |

Run `python train_val.py -h` for the full list.

---

## Adding a new model

1. Drop your implementation in `models/your_model.py`.
2. Add a factory entry in **MODEL_ZOO** inside `train_val.py`:

```python
from models import your_model
MODEL_ZOO["MyNet"] = lambda chans, cls: your_model.MyNet(n_classes=cls, in_chans=chans)
```

3. Train with `--model MyNet`.

---
## Dataset preparation
download from https://bnci-horizon-2020.eu/database/data-sets

* **BCI Competition IV 2a** – 9 subjects, 22 EEG + 3 EOG, 4 classes.
  * Option A: place `A01T.mat … A09E.mat` under `BCICIV2a/{T,E}/`.
  * Option B: use the provided pre‑extracted NPZ archives (`--npz-data True`).
* **BCI Competition IV 2b** – 9 subjects, 3 EEG + 3 EOG, 2 classes, files go to `BCICIV2b/`.
  * place `B01T.mat … B09E.mat` under `BCICIV2a/{T,E}/`.


Ensure folder names match those expected by `data_utils.py`.

---
## License

This work is released for academic research purposes. Please cite the corresponding paper if you find the code useful.

---
*Happy hacking & good luck with your BCI experiments!*

