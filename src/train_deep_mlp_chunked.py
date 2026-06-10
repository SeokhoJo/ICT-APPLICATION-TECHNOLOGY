import os
import glob
import time
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix


# =========================================================
# Path 설정
# =========================================================

BASE_DIR = r"F:\My Folder\study\2026\ICT\project_Data"
SPLIT_DIR = os.path.join(BASE_DIR, "processed", "dl_split_chunked")

TRAIN_DIR = os.path.join(SPLIT_DIR, "train")
VAL_DIR = os.path.join(SPLIT_DIR, "val")
TEST_DIR = os.path.join(SPLIT_DIR, "test")

MODEL_OUT = os.path.join(SPLIT_DIR, "mlp_chunked_model.pkl")
SCALER_OUT = os.path.join(SPLIT_DIR, "mlp_chunked_scaler.pkl")
FEATURE_COLS_OUT = os.path.join(SPLIT_DIR, "mlp_feature_cols.pkl")


# =========================================================
# 학습 설정
# =========================================================

EPOCHS = 5
RANDOM_SEED = 42

DROP_FEATURES = [
    "Dst Port"
]

CLASSES = np.array([0, 1], dtype=np.int8)


# =========================================================
# Utility
# =========================================================

def get_parquet_files(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {folder}")
    return files


def get_feature_columns(sample_df):
    meta_cols = ["DatasetYear", "SourceFile"]

    feature_cols = [
        c for c in sample_df.columns
        if c not in ["Label"] + meta_cols
    ]

    feature_cols = [
        c for c in feature_cols
        if c not in DROP_FEATURES
    ]

    return feature_cols


def load_xy(path, feature_cols):
    df = pd.read_parquet(path)

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df["Label"].astype(np.int8)

    X = X.astype(np.float32)

    return X, y


def print_confusion_report(cm, title):
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / max(cm.sum(), 1)

    benign_precision = tn / max(tn + fn, 1)
    benign_recall = tn / max(tn + fp, 1)
    benign_f1 = 2 * benign_precision * benign_recall / max(benign_precision + benign_recall, 1e-12)

    attack_precision = tp / max(tp + fp, 1)
    attack_recall = tp / max(tp + fn, 1)
    attack_f1 = 2 * attack_precision * attack_recall / max(attack_precision + attack_recall, 1e-12)

    macro_precision = (benign_precision + attack_precision) / 2
    macro_recall = (benign_recall + attack_recall) / 2
    macro_f1 = (benign_f1 + attack_f1) / 2

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print(f"Accuracy: {accuracy:.4f}")

    print("\nConfusion Matrix")
    print(cm)

    print("\nClassification Report")
    print(f"{'class':<10} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")
    print(f"{'BENIGN':<10} {benign_precision:>10.4f} {benign_recall:>10.4f} {benign_f1:>10.4f} {tn + fp:>10}")
    print(f"{'ATTACK':<10} {attack_precision:>10.4f} {attack_recall:>10.4f} {attack_f1:>10.4f} {tp + fn:>10}")
    print()
    print(f"{'macro avg':<10} {macro_precision:>10.4f} {macro_recall:>10.4f} {macro_f1:>10.4f} {cm.sum():>10}")


def evaluate_chunked(model, scaler, files, feature_cols, title):
    total_cm = np.zeros((2, 2), dtype=np.int64)

    for i, path in enumerate(files):
        X, y = load_xy(path, feature_cols)
        X_scaled = scaler.transform(X)

        pred = model.predict(X_scaled)

        cm = confusion_matrix(y, pred, labels=[0, 1])
        total_cm += cm

        if i % 10 == 0:
            print(f"[EVAL] {title} part {i + 1}/{len(files)}")

    print_confusion_report(total_cm, title)

    return total_cm


# =========================================================
# Main
# =========================================================

def main():
    np.random.seed(RANDOM_SEED)

    train_files = get_parquet_files(TRAIN_DIR)
    val_files = get_parquet_files(VAL_DIR)
    test_files = get_parquet_files(TEST_DIR)

    print("[DATA]")
    print(f"Train parts: {len(train_files)}")
    print(f"Val parts  : {len(val_files)}")
    print(f"Test parts : {len(test_files)}")

    sample_df = pd.read_parquet(train_files[0])
    feature_cols = get_feature_columns(sample_df)

    print("\n[FEATURE]")
    print(f"Feature count: {len(feature_cols)}")
    print(f"Drop features: {DROP_FEATURES}")

    # =====================================================
    # 1단계: StandardScaler partial_fit
    # =====================================================

    print("\n" + "=" * 70)
    print("[STEP 1] StandardScaler partial_fit")
    print("=" * 70)

    scaler = StandardScaler()

    start_time = time.time()

    for i, path in enumerate(train_files):
        X, y = load_xy(path, feature_cols)
        scaler.partial_fit(X)

        if i % 10 == 0:
            elapsed = time.time() - start_time
            print(f"[SCALER] part {i + 1}/{len(train_files)} | elapsed {elapsed:.1f}s")

    print("[SCALER] Done")

    # =====================================================
    # 2단계: MLP 모델 생성
    # =====================================================

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        batch_size=4096,
        learning_rate_init=0.001,
        max_iter=1,
        warm_start=False,
        random_state=RANDOM_SEED,
        verbose=False,
        early_stopping=False
    )

    print("\n" + "=" * 70)
    print("[STEP 2] Train Deep MLP")
    print("=" * 70)
    print(f"Model: MLPClassifier")
    print(f"Hidden layers: 128 -> 64 -> 32")
    print(f"Epochs: {EPOCHS}")
    print(f"Optimizer: Adam")
    print(f"Backpropagation: Used")

    first_fit = True

    # =====================================================
    # 3단계: Epoch 반복 학습
    # =====================================================

    for epoch in range(1, EPOCHS + 1):
        print("\n" + "-" * 70)
        print(f"[EPOCH {epoch}/{EPOCHS}]")
        print("-" * 70)

        epoch_start = time.time()

        shuffled_files = train_files.copy()
        np.random.shuffle(shuffled_files)

        for i, path in enumerate(shuffled_files):
            X, y = load_xy(path, feature_cols)
            X_scaled = scaler.transform(X)

            if first_fit:
                model.partial_fit(X_scaled, y, classes=CLASSES)
                first_fit = False
            else:
                model.partial_fit(X_scaled, y, classes=CLASSES)

            if i % 10 == 0:
                elapsed = time.time() - epoch_start
                print(
                    f"[TRAIN] epoch {epoch} | "
                    f"part {i + 1}/{len(shuffled_files)} | "
                    f"elapsed {elapsed:.1f}s"
                )

        # Epoch마다 validation 평가
        print(f"\n[VALIDATION AFTER EPOCH {epoch}]")
        evaluate_chunked(
            model=model,
            scaler=scaler,
            files=val_files,
            feature_cols=feature_cols,
            title=f"VALIDATION RESULT - EPOCH {epoch}"
        )

    # =====================================================
    # 4단계: 최종 Test 평가
    # =====================================================

    print("\n" + "=" * 70)
    print("[STEP 3] Final Test Evaluation")
    print("=" * 70)

    evaluate_chunked(
        model=model,
        scaler=scaler,
        files=test_files,
        feature_cols=feature_cols,
        title="FINAL TEST RESULT"
    )

    # =====================================================
    # 5단계: 저장
    # =====================================================

    joblib.dump(model, MODEL_OUT)
    joblib.dump(scaler, SCALER_OUT)
    joblib.dump(feature_cols, FEATURE_COLS_OUT)

    print("\n" + "=" * 70)
    print("[SAVED]")
    print("=" * 70)
    print(MODEL_OUT)
    print(SCALER_OUT)
    print(FEATURE_COLS_OUT)


if __name__ == "__main__":
    main()