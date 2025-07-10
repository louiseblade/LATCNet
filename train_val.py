import argparse, random, copy, logging,  sys
import numpy as np
import torch
import torch.nn as nn

from typing import List
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import GradScaler, autocast
from logging.handlers import RotatingFileHandler
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

# Local module
from data_utils       import load_data_2a_npz, load_data_2b, load_data_2a
from maxnorm_constraint import apply_max_norm_refined
from weight_init      import initialize_weights_keras_style
from weight_decay     import get_optimizer
from models           import (DeepConvNet, EEGTCNet, EEGNet, EEGNeX,
                              ShallowConvNet, TCN_Fusion, ATCNet, proposed_model)

MODEL_ZOO = {
    "GATCNet"       : lambda chans, cls: proposed_model.GATCNet(num_classes=cls,
                                                                n_windows=5,
                                                                in_chans=chans,
                                                                gate="conv1d"),
    "ATCNet"        : lambda chans, cls: ATCNet.ATCNet(n_classes=cls, in_chans=chans),
    "DeepConvNet"   : lambda chans, cls: DeepConvNet.DeepConvNet(n_classes=cls, chans=chans),
    "EEGNet"        : lambda chans, cls: EEGNet.EEGNetClassifier(n_classes=cls, chans=chans),
    "EEGNeX"        : lambda chans, cls: EEGNeX.EEGNeX_8_32(n_outputs=cls, n_timesteps=1125, n_features=chans),
    "EEGTCNet"      : lambda chans, cls: EEGTCNet.EEGTCNet(n_classes=cls, Chans=chans),
    "ShallowConvNet": lambda chans, cls: ShallowConvNet.ShallowConvNet(n_classes=cls, chans=chans),
    "TCN_Fusion"    : lambda chans, cls: TCN_Fusion.TCNetFusion(n_classes=cls, Chans=chans),
}

# ----------------------------------------------------------------------
# CLI parsing
# ----------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BCI training (within / LOSO)")
    p.add_argument("--seeds",        type=int, nargs="+", default=[0, 1, 2],
                   help="Random seeds (list)")
    p.add_argument("--subjects",     type=int, nargs="+", choices=range(1,10),
                   default=list(range(1,10)), metavar="[1‑9]",
                   help="Subject IDs to include")
    p.add_argument("--epochs",       type=int, default=1200)
    p.add_argument("--batch-size",   type=int, default=64)
    p.add_argument("--model",        type=str, default="GATCNet",
                   choices=list(MODEL_ZOO.keys()))
    p.add_argument("--dataset",      type=str, default="a", choices=["a","b"])
    p.add_argument("--training-mode",type=str, default="within",
                   choices=["within","loso"],
                   help="'within' = subject‑specific, 'loso' = leave‑one‑subject‑out")
    p.add_argument(
        '--validation-mode',
        type=str,
        default='train_val_test',
        choices=['train_val_test', 'train_test_only'],
        help="Set the validation strategy. 'train_val_test' splits the training set to create a "
             "validation set (default). 'train_test_only' uses the actual test set for validation "
             "during training."
    )
    p.add_argument(
        '--model-selection',
        type=str,
        default='best_val_loss',
        choices=['best_val_loss', 'early_stopping', 'final_epoch'],
        help="Strategy for selecting the final model. 'best_val_loss' uses the model with the best validation loss (default). "
             "'early_stopping' stops training when validation loss stops improving. 'final_epoch' uses the model from the last epoch."
    )
    p.add_argument(
        '--early-stopping-patience',
        type=int,
        default=400,
        help="Number of epochs to wait for validation loss improvement before stopping. Only used if --model-selection is 'early_stopping'."
    )
    p.add_argument('--npz-data',
                   type=bool,
                   default=False,
                   help="For BCI IV dataset 2a, you may download the data in https://bnci-horizon-2020.eu/database/data-sets"
                   "However, for some reasons, I usually bump into connection issue with this website, ended up not being able to download the data 2a."
                   "So I prepare a npz version extracted from the original gdf files. If you can download bnci-horizon data, please set this param to False"
    )
    return p

# ----------------------------------------------------------------------
# Convenience I/O
# ----------------------------------------------------------------------
def setup_logging():
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    rot = RotatingFileHandler("training.log", maxBytes=5*1024*1024, backupCount=3)
    rot.setFormatter(fmt)
    con = logging.StreamHandler(sys.stdout)
    con.setFormatter(fmt)
    logging.basicConfig(level=logging.INFO, handlers=[rot, con])
    return logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Dataset loaders
# ----------------------------------------------------------------------
def load_dataset(dataset: str,
                 subjects: List[int],
                 split: str,
                 transpose: bool = True,
                 npz: bool = True):
    """
    Wrapper to hide file‑system details.
    dataset: 'a' or 'b'
    split  : 'T' (train) or 'E' (eval)
    """
    if dataset == "a":
        if npz:
            npz_dir = "BCICIV2a_npz/"
            return load_data_2a_npz(npz_dir + "data_npz",
                                    npz_dir + "true_labels",
                                    subjects,
                                    split,
                                    transpose=transpose)
        else:
            print("load here")
            return load_data_2a(subjects, split)
    else:  # dataset 'b'
        return load_data_2b(subjects, split)

# ----------------------------------------------------------------------
# Training / evaluation loop
# ----------------------------------------------------------------------
def run_experiment_og(args, logger):

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    num_class = 4 if args.dataset == "a" else 2
    max_norm_value = 0.6

    # results[seed]['accuracy'|'kappa'][subject] = value
    results = {seed: {'accuracy': {}, 'kappa': {}} for seed in args.seeds}

    for seed in args.seeds:
        logger.info(f"\n===== Seed {seed} =====")
        torch.manual_seed(seed); torch.cuda.manual_seed(seed)
        np.random.seed(seed);     random.seed(seed)

        for test_sub in args.subjects:
            logger.info(f"\n--- Subject {test_sub} ---")

            # --------------------------------------------------------------
            # Build train/test subject lists according to protocol
            # --------------------------------------------------------------
            if args.training_mode == "within":
                train_subjects = [test_sub]
                test_subjects  = [test_sub]
            else:  # LOSO
                train_subjects = args.subjects.copy()
                train_subjects.remove(test_sub)
                test_subjects  = [test_sub]
                logger.info(f"Train list: {train_subjects}")
                logger.info(f"Test  list: {test_subjects}")

            # --------------------------------------------------------------
            # Load data
            # --------------------------------------------------------------
            X_full, y_full = load_dataset(args.dataset, train_subjects, "T")
            X_test, y_test = load_dataset(args.dataset, test_subjects, "E")
            logger.info("Data loaded")

            # --------------------------------------------------------------
            # Train/val split (stratified)
            # --------------------------------------------------------------
            X_train, X_val, y_train_np, y_val_np = train_test_split(
                X_full, y_full, test_size=0.2, random_state=seed, stratify=y_full)

            # Tensors on device
            X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
            X_val   = torch.tensor(X_val,   dtype=torch.float32, device=device)
            X_test  = torch.tensor(X_test,  dtype=torch.float32, device=device)
            y_train = torch.tensor(y_train_np, dtype=torch.long, device=device)
            y_val   = torch.tensor(y_val_np,   dtype=torch.long, device=device)
            y_test  = torch.tensor(y_test,     dtype=torch.long, device=device)

            train_loader = DataLoader(
                TensorDataset(X_train, y_train),
                batch_size=args.batch_size, shuffle=True, drop_last=True)
            val_loader   = DataLoader(
                TensorDataset(X_val, y_val),
                batch_size=args.batch_size, shuffle=False)

            # --------------------------------------------------------------
            # Model
            # --------------------------------------------------------------
            model_ctor = MODEL_ZOO[args.model]
            model = model_ctor(X_train.size(1), num_class).to(device)
            model.apply(initialize_weights_keras_style)

            # Loss / optimiser / scheduler
            criterion  = nn.CrossEntropyLoss()
            optimizer  = get_optimizer(model, lr=1e-3, weight_decay=True, verbose=False)
            scaler     = GradScaler(device.type)
            scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=20, factor=0.9, min_lr=1e-4)

            best_state, best_val_loss, best_epoch = copy.deepcopy(model.state_dict()), float("inf"), 0

            # ==========================================================
            # Training loop
            # ==========================================================
            for epoch in range(args.epochs):
                # ---- Train ----
                model.train()
                epoch_loss = correct = total = 0
                for xb, yb in train_loader:
                    optimizer.zero_grad(set_to_none=True)
                    with autocast(device.type):
                        out  = model(xb)
                        loss = criterion(out, yb)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer); scaler.update()

                    apply_max_norm_refined(max_norm_val=max_norm_value,
                                           modules_to_apply=[model],
                                           layers=(nn.Conv1d, nn.Conv2d, nn.Linear))
                    epoch_loss += loss.item() * xb.size(0)
                    total      += yb.size(0)
                    correct    += (out.argmax(dim=1) == yb).sum().item()

                train_loss, train_acc = epoch_loss / total, correct / total

                # ---- Validation ----
                model.eval()
                v_loss = v_correct = v_total = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        out  = model(xb)
                        loss = criterion(out, yb)
                        v_loss  += loss.item() * xb.size(0)
                        v_total += yb.size(0)
                        v_correct += (out.argmax(dim=1) == yb).sum().item()
                val_loss, val_acc = v_loss / v_total, v_correct / v_total
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss, best_epoch = val_loss, epoch
                    best_state = copy.deepcopy(model.state_dict())

                if epoch % 100 == 0 or epoch == args.epochs-1:
                    logger.info(
                        f"[Seed {seed} Sub {test_sub} Ep {epoch}] "
                        f"TL:{train_loss:.4f} TA:{train_acc*100:5.2f}% | "
                        f"VL:{val_loss:.4f} VA:{val_acc*100:5.2f}% | "
                        f"LR:{optimizer.param_groups[0]['lr']:.6f}")

            # ==========================================================
            # Test
            # ==========================================================
            model.load_state_dict(best_state); model.eval()
            with torch.no_grad():
                out  = model(X_test)
                preds = out.argmax(dim=1)
                acc   = (preds == y_test).float().mean().item() * 100
                kappa = cohen_kappa_score(y_test.cpu().numpy(), preds.cpu().numpy())

            results[seed]['accuracy'][test_sub] = acc
            results[seed]['kappa'][test_sub]    = kappa
            logger.info(
                f"Seed {seed} Sub {test_sub}: best VL {best_val_loss:.4f} "
                f"(epoch {best_epoch}) | Test Acc {acc:.2f}% | κ {kappa:.4f}")

            del model; torch.cuda.empty_cache()


    # ------------------------------------------------------------------
    # Pretty print aggregated results (same table style as originals)
    # ------------------------------------------------------------------
    col_w, lbl_w = 10, 15
    hdr          = [f"sub_{s}" for s in args.subjects] + ["average"]
    header_line  = " "*lbl_w + "".join(f"{h:>{col_w}}" for h in hdr)
    print("\n---------------------------------\nTest Performance (Accuracy):\n---------------------------------")
    print(header_line)
    for seed in args.seeds:
        accs      = [results[seed]['accuracy'].get(s, np.nan) for s in args.subjects]
        avg_acc   = np.nanmean(accs)
        acc_str   = "".join(f"{a:>{col_w}.2f}" if not np.isnan(a) else f"{'N/A':>{col_w}}" for a in accs)
        print(f"{'Seed '+str(seed)+':':<{lbl_w}}{acc_str}{avg_acc:>{col_w}.2f}")

    # Per‑subject averages
    subj_means = [np.nanmean([results[seed]['accuracy'].get(s, np.nan) for seed in args.seeds])
                  for s in args.subjects]
    subj_str   = "".join(f"{m:>{col_w}.2f}" if not np.isnan(m) else f"{'N/A':>{col_w}}" for m in subj_means)
    print(f"\n{'Subject avg:':<{lbl_w}}{subj_str}{np.nanmean(subj_means):>{col_w}.2f}")

    # Repeat for Kappa
    print("\n---------------------------------\nTest Performance (Cohen's Kappa):\n---------------------------------")
    print(header_line)
    for seed in args.seeds:
        kappas   = [results[seed]['kappa'].get(s, np.nan) for s in args.subjects]
        avg_kap  = np.nanmean(kappas)
        kappa_str= "".join(f"{k:>{col_w}.3f}" if not np.isnan(k) else f"{'N/A':>{col_w}}" for k in kappas)
        print(f"{'Seed '+str(seed)+':':<{lbl_w}}{kappa_str}{avg_kap:>{col_w}.3f}")
    subj_km = [np.nanmean([results[seed]['kappa'].get(s, np.nan) for seed in args.seeds])
               for s in args.subjects]
    subj_kstr = "".join(f"{k:>{col_w}.3f}" if not np.isnan(k) else f"{'N/A':>{col_w}}" for k in subj_km)
    print(f"\n{'Subject avg:':<{lbl_w}}{subj_kstr}{np.nanmean(subj_km):>{col_w}.3f}")

def run_experiment(args, logger):
    """
    Runs a full training and evaluation experiment for a given configuration.
    """
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Model selection strategy: {args.model_selection}")
    if args.model_selection == 'early_stopping':
        logger.info(f"Early stopping patience: {args.early_stopping_patience}")


    num_class = 4 if args.dataset == "a" else 2
    max_norm_value = 0.6

    results = {seed: {'accuracy': {}, 'kappa': {}} for seed in args.seeds}

    for seed in args.seeds:
        logger.info(f"\n===== Seed {seed} =====")
        torch.manual_seed(seed); torch.cuda.manual_seed(seed)
        np.random.seed(seed);     random.seed(seed)

        for test_sub in args.subjects:
            logger.info(f"\n--- Subject {test_sub} ---")

            # ... (data loading and splitting code remains the same) ...
            if args.training_mode == "within":
                train_subjects = [test_sub]
                test_subjects  = [test_sub]
            else:
                train_subjects = args.subjects.copy()
                train_subjects.remove(test_sub)
                test_subjects  = [test_sub]
                logger.info(f"Train list: {train_subjects}")
                logger.info(f"Test  list: {test_subjects}")

            X_full, y_full = load_dataset(args.dataset, train_subjects, "T", args.npz_data)
            X_test_data, y_test_data = load_dataset(args.dataset, test_subjects, "E",  args.npz_data)
            logger.info("Data loaded")

            validation_mode = getattr(args, 'validation_mode', 'train_val_test')
            if validation_mode == 'train_test_only':
                logger.info("Mode: train-test-only. Using test set for validation.")
                X_train, y_train_np = X_full, y_full
                X_val, y_val_np = X_test_data, y_test_data
            else:
                logger.info("Mode: train-val-test. Splitting training data for validation.")
                X_train, X_val, y_train_np, y_val_np = train_test_split(
                    X_full, y_full, test_size=0.2, random_state=seed, stratify=y_full)

            X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
            X_val   = torch.tensor(X_val,   dtype=torch.float32, device=device)
            X_test  = torch.tensor(X_test_data,  dtype=torch.float32, device=device)
            y_train = torch.tensor(y_train_np, dtype=torch.long, device=device)
            y_val   = torch.tensor(y_val_np,   dtype=torch.long, device=device)
            y_test  = torch.tensor(y_test_data,  dtype=torch.long, device=device)

            train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True, drop_last=True)
            val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)

            # ... (model setup code remains the same) ...
            model_ctor = MODEL_ZOO[args.model]
            model = model_ctor(X_train.size(1), num_class).to(device)
            model.apply(initialize_weights_keras_style)
            criterion  = nn.CrossEntropyLoss()
            optimizer  = get_optimizer(model, lr=1e-3, weight_decay=True, verbose=False)
            scaler     = GradScaler(device.type)
            scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=20, factor=0.9, min_lr=1e-4)

            # --- MODIFICATION: Initialize variables for model selection ---
            best_state, best_val_loss, best_epoch = copy.deepcopy(model.state_dict()), float("inf"), 0
            epochs_without_improvement = 0

            # ==========================================================
            # Training loop
            # ==========================================================
            for epoch in range(args.epochs):
                # ---- Train ----
                model.train()
                # ... (training step code remains the same) ...
                epoch_loss = correct = total = 0
                for xb, yb in train_loader:
                    optimizer.zero_grad(set_to_none=True)
                    with autocast(device.type):
                        out  = model(xb)
                        loss = criterion(out, yb)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer); scaler.update()
                    apply_max_norm_refined(max_norm_val=max_norm_value, modules_to_apply=[model], layers=(nn.Conv1d, nn.Conv2d, nn.Linear))
                    epoch_loss += loss.item() * xb.size(0)
                    total      += yb.size(0)
                    correct    += (out.argmax(dim=1) == yb).sum().item()
                train_loss, train_acc = epoch_loss / total, correct / total

                # ---- Validation ----
                model.eval()
                v_loss = v_correct = v_total = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        out  = model(xb)
                        loss = criterion(out, yb)
                        v_loss  += loss.item() * xb.size(0)
                        v_total += yb.size(0)
                        v_correct += (out.argmax(dim=1) == yb).sum().item()
                val_loss, val_acc = v_loss / v_total, v_correct / v_total
                scheduler.step(val_loss)

                # --- MODIFICATION: Implement model selection and early stopping logic ---
                if val_loss < best_val_loss:
                    best_val_loss, best_epoch = val_loss, epoch
                    best_state = copy.deepcopy(model.state_dict())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if args.model_selection == 'early_stopping' and epochs_without_improvement >= args.early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch} after {args.early_stopping_patience} epochs without improvement.")
                    break  # Exit the training loop

                if epoch % 100 == 0 or epoch == args.epochs-1:
                    logger.info(
                        f"[Seed {seed} Sub {test_sub} Ep {epoch}] "
                        f"TL:{train_loss:.4f} TA:{train_acc*100:5.2f}% | "
                        f"VL:{val_loss:.4f} VA:{val_acc*100:5.2f}% | "
                        f"LR:{optimizer.param_groups[0]['lr']:.6f}")

            # ==========================================================
            # Test
            # ==========================================================
            # --- MODIFICATION: Conditionally load the best model state ---
            if args.model_selection in ['best_val_loss', 'early_stopping']:
                logger.info(f"Loading best model state from epoch {best_epoch} (Val Loss: {best_val_loss:.4f}) for testing.")
                model.load_state_dict(best_state)
            else: # 'final_epoch'
                logger.info(f"Using final model state from last epoch for testing.")
                # No action needed, the model is already in its final state

            model.eval()
            with torch.no_grad():
                out  = model(X_test)
                preds = out.argmax(dim=1)
                acc   = (preds == y_test).float().mean().item() * 100
                kappa = cohen_kappa_score(y_test.cpu().numpy(), preds.cpu().numpy())

            results[seed]['accuracy'][test_sub] = acc
            results[seed]['kappa'][test_sub]    = kappa
            if args.model_selection in ['best_val_loss', 'early_stopping']:
                logger.info(
                    f"Seed {seed} Sub {test_sub}: best VL {best_val_loss:.4f} "
                    f"(epoch {best_epoch}) | Test Acc {acc:.2f}% | κ {kappa:.4f}")
            else:
                logger.info(f" Test Acc {acc:.2f}% | κ {kappa:.4f}")

            del model; torch.cuda.empty_cache()

    # ... (results printing code remains the same) ...
    col_w, lbl_w = 10, 15
    hdr          = [f"sub_{s}" for s in args.subjects] + ["average"]
    header_line  = " "*lbl_w + "".join(f"{h:>{col_w}}" for h in hdr)
    print("\n---------------------------------\nTest Performance (Accuracy):\n---------------------------------")
    print(header_line)
    for seed in args.seeds:
        accs      = [results[seed]['accuracy'].get(s, np.nan) for s in args.subjects]
        avg_acc   = np.nanmean(accs)
        acc_str   = "".join(f"{a:>{col_w}.2f}" if not np.isnan(a) else f"{'N/A':>{col_w}}" for a in accs)
        print(f"{'Seed '+str(seed)+':':<{lbl_w}}{acc_str}{avg_acc:>{col_w}.2f}")
    subj_means = [np.nanmean([results[seed]['accuracy'].get(s, np.nan) for seed in args.seeds]) for s in args.subjects]
    subj_str   = "".join(f"{m:>{col_w}.2f}" if not np.isnan(m) else f"{'N/A':>{col_w}}" for m in subj_means)
    print(f"\n{'Subject avg:':<{lbl_w}}{subj_str}{np.nanmean(subj_means):>{col_w}.2f}")
    print("\n---------------------------------\nTest Performance (Cohen's Kappa):\n---------------------------------")
    print(header_line)
    for seed in args.seeds:
        kappas   = [results[seed]['kappa'].get(s, np.nan) for s in args.subjects]
        avg_kap  = np.nanmean(kappas)
        kappa_str= "".join(f"{k:>{col_w}.3f}" if not np.isnan(k) else f"{'N/A':>{col_w}}" for k in kappas)
        print(f"{'Seed '+str(seed)+':':<{lbl_w}}{kappa_str}{avg_kap:>{col_w}.3f}")
    subj_km = [np.nanmean([results[seed]['kappa'].get(s, np.nan) for seed in args.seeds]) for s in args.subjects]
    subj_kstr = "".join(f"{k:>{col_w}.3f}" if not np.isnan(k) else f"{'N/A':>{col_w}}" for k in subj_km)
    print(f"\n{'Subject avg:':<{lbl_w}}{subj_kstr}{np.nanmean(subj_km):>{col_w}.3f}")
# ----------------------------------------------------------------------
if __name__ == "__main__":
    args   = build_argparser().parse_args()
    logger = setup_logging()
    run_experiment(args, logger)
