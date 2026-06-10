import os
import glob
import json
import time
import random

import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import confusion_matrix


# =========================================================
# Path Configuration
# =========================================================

BASE_DIR = r"F:\My Folder\study\2026\ICT\project_Data"
SPLIT_DIR = os.path.join(BASE_DIR, "processed", "dl_split_chunked")

TRAIN_DIR = os.path.join(SPLIT_DIR, "train")
VAL_DIR = os.path.join(SPLIT_DIR, "val")
TEST_DIR = os.path.join(SPLIT_DIR, "test")

# Reuse scaler and feature columns from the previous preprocessing/training pipeline
SCALER_PATH = os.path.join(SPLIT_DIR, "mlp_chunked_scaler.pkl")
FEATURE_COLS_PATH = os.path.join(SPLIT_DIR, "mlp_feature_cols.pkl")

# PyTorch output files
TORCH_MODEL_PATH = os.path.join(SPLIT_DIR, "pytorch_mlp_model.pt")
TORCH_CONFIG_PATH = os.path.join(SPLIT_DIR, "pytorch_mlp_config.json")
TORCH_THRESHOLD_RESULT_PATH = os.path.join(SPLIT_DIR, "pytorch_threshold_tuning_results.csv")
TORCH_TEST_RESULT_PATH = os.path.join(SPLIT_DIR, "pytorch_test_result.json")


# =========================================================
# Training Configuration
# =========================================================

RANDOM_SEED = 42

EPOCHS = 10
BATCH_SIZE = 8192
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Weighted loss: increase attack sensitivity.
# Higher ATTACK_WEIGHT may improve attack recall but can increase false positives.
BENIGN_WEIGHT = 1.0
ATTACK_WEIGHT = 1.5

THRESHOLDS = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10]
TARGET_ATTACK_RECALL = 0.95

# Full validation/test evaluation is expensive.
# For epoch monitoring, only a subset is used by default.
# Final threshold tuning and final test evaluation use all parts unless changed.
MAX_VAL_PARTS_PER_EPOCH = 30
MAX_VAL_PARTS_FINAL = None
MAX_TEST_PARTS_FINAL = None


# =========================================================
# Reproducibility / Device
# =========================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] {DEVICE}")


# =========================================================
# Model Definition
# =========================================================

class PyTorchMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


# =========================================================
# Data Utilities
# =========================================================

def get_parquet_files(folder: str, max_parts=None):
    files = sorted(glob.glob(os.path.join(folder, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {folder}")
    if max_parts is not None:
        files = files[:max_parts]
    return files


def load_xy_from_parquet(path: str, feature_cols, scaler):
    df = pd.read_parquet(path)

    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    X = X.astype(np.float32)

    y = df["Label"].astype(np.float32).to_numpy()

    X_scaled = scaler.transform(X).astype(np.float32)
    return X_scaled, y


def make_loader(X, y, batch_size: int, shuffle: bool):
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


# =========================================================
# Metrics
# =========================================================

def calc_metrics_from_cm(cm):
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp

    accuracy = (tn + tp) / max(total, 1)

    benign_precision = tn / max(tn + fn, 1)
    benign_recall = tn / max(tn + fp, 1)
    benign_f1 = 2 * benign_precision * benign_recall / max(benign_precision + benign_recall, 1e-12)

    attack_precision = tp / max(tp + fp, 1)
    attack_recall = tp / max(tp + fn, 1)
    attack_f1 = 2 * attack_precision * attack_recall / max(attack_precision + attack_recall, 1e-12)

    macro_precision = (benign_precision + attack_precision) / 2
    macro_recall = (benign_recall + attack_recall) / 2
    macro_f1 = (benign_f1 + attack_f1) / 2

    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy),
        "benign_precision": float(benign_precision),
        "benign_recall": float(benign_recall),
        "benign_f1": float(benign_f1),
        "attack_precision": float(attack_precision),
        "attack_recall": float(attack_recall),
        "attack_f1": float(attack_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
    }


def evaluate_thresholds(model, files, feature_cols, scaler, thresholds, title):
    model.eval()
    cms = {th: np.zeros((2, 2), dtype=np.int64) for th in thresholds}

    with torch.no_grad():
        for i, path in enumerate(files):
            X, y = load_xy_from_parquet(path, feature_cols, scaler)
            loader = make_loader(X, y, batch_size=BATCH_SIZE, shuffle=False)

            probs_list = []
            y_list = []

            for xb, yb in loader:
                xb = xb.to(DEVICE, non_blocking=True)
                logits = model(xb)
                probs = torch.sigmoid(logits).cpu().numpy()
                probs_list.append(probs)
                y_list.append(yb.numpy())

            probs = np.concatenate(probs_list)
            y_true = np.concatenate(y_list).astype(np.int8)

            for th in thresholds:
                y_pred = (probs >= th).astype(np.int8)
                cms[th] += confusion_matrix(y_true, y_pred, labels=[0, 1])

            if i % 10 == 0:
                print(f"[{title}] evaluated part {i + 1}/{len(files)}")

    rows = []
    for th in thresholds:
        metrics = calc_metrics_from_cm(cms[th])
        metrics["threshold"] = th
        rows.append(metrics)

    return sorted(rows, key=lambda x: x["threshold"], reverse=True)


def print_threshold_table(rows, title):
    print("\n" + "=" * 110)
    print(f"[{title}]")
    print("=" * 110)
    print(
        f"{'threshold':>10} "
        f"{'accuracy':>10} "
        f"{'atk_prec':>10} "
        f"{'atk_rec':>10} "
        f"{'atk_f1':>10} "
        f"{'ben_rec':>10} "
        f"{'FP':>12} "
        f"{'FN':>12}"
    )
    for r in rows:
        print(
            f"{r['threshold']:>10.2f} "
            f"{r['accuracy']:>10.4f} "
            f"{r['attack_precision']:>10.4f} "
            f"{r['attack_recall']:>10.4f} "
            f"{r['attack_f1']:>10.4f} "
            f"{r['benign_recall']:>10.4f} "
            f"{r['fp']:>12,} "
            f"{r['fn']:>12,}"
        )


def choose_best_threshold(rows):
    candidates = [r for r in rows if r["attack_recall"] >= TARGET_ATTACK_RECALL]
    if candidates:
        best = max(candidates, key=lambda r: r["macro_f1"])
        reason = f"macro_f1 max among thresholds with attack_recall >= {TARGET_ATTACK_RECALL}"
    else:
        best = max(rows, key=lambda r: r["macro_f1"])
        reason = "macro_f1 max because no threshold satisfied target attack recall"
    return best, reason


# =========================================================
# Training
# =========================================================

def train_one_epoch(model, train_files, feature_cols, scaler, optimizer, epoch):
    model.train()
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    total_loss = 0.0
    total_count = 0
    start_time = time.time()

    benign_weight_tensor = torch.tensor(BENIGN_WEIGHT, device=DEVICE)
    attack_weight_tensor = torch.tensor(ATTACK_WEIGHT, device=DEVICE)

    for part_idx, path in enumerate(train_files):
        X, y = load_xy_from_parquet(path, feature_cols, scaler)
        loader = make_loader(X, y, batch_size=BATCH_SIZE, shuffle=True)

        part_loss = 0.0
        part_count = 0

        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            logits = model(xb)

            loss_raw = criterion(logits, yb)
            weights = torch.where(yb == 1, attack_weight_tensor, benign_weight_tensor)
            loss = (loss_raw * weights).mean()

            loss.backward()
            optimizer.step()

            n = len(yb)
            part_loss += loss.item() * n
            part_count += n

        total_loss += part_loss
        total_count += part_count

        if part_idx % 10 == 0:
            print(
                f"[EPOCH {epoch}] train part {part_idx + 1}/{len(train_files)} "
                f"part_loss={part_loss / max(part_count, 1):.6f}"
            )

    avg_loss = total_loss / max(total_count, 1)
    elapsed = time.time() - start_time
    print(f"[EPOCH {epoch}] train_loss={avg_loss:.6f}, rows={total_count:,}, elapsed={elapsed:.1f}s")
    return avg_loss


# =========================================================
# Main
# =========================================================

def main():
    print("[LOAD] scaler")
    scaler = joblib.load(SCALER_PATH)

    print("[LOAD] feature columns")
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    input_dim = len(feature_cols)

    print(f"[INFO] input_dim={input_dim}")
    print(f"[INFO] benign_weight={BENIGN_WEIGHT}, attack_weight={ATTACK_WEIGHT}")

    train_files = get_parquet_files(TRAIN_DIR)
    val_files_epoch = get_parquet_files(VAL_DIR, max_parts=MAX_VAL_PARTS_PER_EPOCH)
    val_files_final = get_parquet_files(VAL_DIR, max_parts=MAX_VAL_PARTS_FINAL)
    test_files_final = get_parquet_files(TEST_DIR, max_parts=MAX_TEST_PARTS_FINAL)

    print(f"[INFO] train parts={len(train_files)}")
    print(f"[INFO] val parts per epoch={len(val_files_epoch)}")
    print(f"[INFO] val parts final={len(val_files_final)}")
    print(f"[INFO] test parts final={len(test_files_final)}")

    model = PyTorchMLP(input_dim=input_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_macro_f1 = -1.0
    best_state = None
    best_epoch = None

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_files, feature_cols, scaler, optimizer, epoch)

        val_rows = evaluate_thresholds(
            model=model,
            files=val_files_epoch,
            feature_cols=feature_cols,
            scaler=scaler,
            thresholds=[0.25],
            title=f"VAL epoch {epoch}"
        )

        val_macro_f1 = val_rows[0]["macro_f1"]
        val_attack_recall = val_rows[0]["attack_recall"]

        print(
            f"[EPOCH {epoch}] train_loss={train_loss:.6f}, "
            f"VAL threshold=0.25 macro_f1={val_macro_f1:.4f}, attack_recall={val_attack_recall:.4f}"
        )

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"[BEST] epoch={best_epoch}, val_macro_f1={best_val_macro_f1:.4f}")

    if best_state is None:
        raise RuntimeError("No best model state was saved.")

    print("\n[LOAD BEST MODEL]")
    model.load_state_dict(best_state)

    print("\n[FINAL VALIDATION THRESHOLD TUNING]")
    val_threshold_rows = evaluate_thresholds(
        model=model,
        files=val_files_final,
        feature_cols=feature_cols,
        scaler=scaler,
        thresholds=THRESHOLDS,
        title="VAL THRESHOLD"
    )
    print_threshold_table(val_threshold_rows, "VALIDATION THRESHOLD TUNING")

    best_threshold_row, reason = choose_best_threshold(val_threshold_rows)
    best_threshold = best_threshold_row["threshold"]

    print("\n[BEST THRESHOLD]")
    print(f"threshold={best_threshold}")
    print(f"reason={reason}")
    print(best_threshold_row)

    pd.DataFrame(val_threshold_rows).to_csv(TORCH_THRESHOLD_RESULT_PATH, index=False, encoding="utf-8-sig")

    print("\n[FINAL TEST EVALUATION]")
    test_rows = evaluate_thresholds(
        model=model,
        files=test_files_final,
        feature_cols=feature_cols,
        scaler=scaler,
        thresholds=[best_threshold],
        title="TEST"
    )
    print_threshold_table(test_rows, "FINAL TEST RESULT")
    test_result = test_rows[0]

    save_obj = {
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden_layers": [256, 128, 64],
        "dropout": [0.20, 0.20, 0.10],
        "best_epoch": best_epoch,
        "best_threshold": best_threshold,
        "benign_weight": BENIGN_WEIGHT,
        "attack_weight": ATTACK_WEIGHT,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "feature_cols_path": FEATURE_COLS_PATH,
        "scaler_path": SCALER_PATH,
    }

    torch.save(save_obj, TORCH_MODEL_PATH)

    config = {
        "model": "PyTorch MLP",
        "input_dim": input_dim,
        "hidden_layers": [256, 128, 64],
        "optimizer": "AdamW",
        "loss": "Weighted BCEWithLogitsLoss",
        "benign_weight": BENIGN_WEIGHT,
        "attack_weight": ATTACK_WEIGHT,
        "best_epoch": best_epoch,
        "best_threshold": best_threshold,
        "threshold_selection_reason": reason,
        "validation_best_threshold_metrics": best_threshold_row,
        "test_metrics": test_result,
    }

    with open(TORCH_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    with open(TORCH_TEST_RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(test_result, f, indent=2)

    print("\n[SAVED]")
    print(TORCH_MODEL_PATH)
    print(TORCH_CONFIG_PATH)
    print(TORCH_THRESHOLD_RESULT_PATH)
    print(TORCH_TEST_RESULT_PATH)


if __name__ == "__main__":
    main()
