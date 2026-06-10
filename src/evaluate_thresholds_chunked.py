import os
import glob
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import confusion_matrix


# =========================================================
# Path 설정
# =========================================================

BASE_DIR = r"F:\My Folder\study\2026\ICT\project_Data"
SPLIT_DIR = os.path.join(BASE_DIR, "processed", "dl_split_chunked")

TEST_DIR = os.path.join(SPLIT_DIR, "test")

MODEL_PATH = os.path.join(SPLIT_DIR, "mlp_chunked_model.pkl")
SCALER_PATH = os.path.join(SPLIT_DIR, "mlp_chunked_scaler.pkl")
FEATURE_COLS_PATH = os.path.join(SPLIT_DIR, "mlp_feature_cols.pkl")

OUTPUT_CSV = os.path.join(SPLIT_DIR, "threshold_tuning_results.csv")


# =========================================================
# Threshold 후보
# =========================================================

THRESHOLDS = [
    0.50,
    0.45,
    0.40,
    0.35,
    0.30,
    0.25,
    0.20
]


# =========================================================
# Utility
# =========================================================

def get_parquet_files(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {folder}")
    return files


def load_xy(path, feature_cols):
    df = pd.read_parquet(path)

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df["Label"].astype(np.int8)

    X = X.astype(np.float32)

    return X, y


def calc_metrics_from_cm(cm):
    tn, fp, fn, tp = cm.ravel()

    total = tn + fp + fn + tp

    accuracy = (tn + tp) / max(total, 1)

    benign_precision = tn / max(tn + fn, 1)
    benign_recall = tn / max(tn + fp, 1)
    benign_f1 = (
        2 * benign_precision * benign_recall
        / max(benign_precision + benign_recall, 1e-12)
    )

    attack_precision = tp / max(tp + fp, 1)
    attack_recall = tp / max(tp + fn, 1)
    attack_f1 = (
        2 * attack_precision * attack_recall
        / max(attack_precision + attack_recall, 1e-12)
    )

    macro_precision = (benign_precision + attack_precision) / 2
    macro_recall = (benign_recall + attack_recall) / 2
    macro_f1 = (benign_f1 + attack_f1) / 2

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": accuracy,
        "benign_precision": benign_precision,
        "benign_recall": benign_recall,
        "benign_f1": benign_f1,
        "attack_precision": attack_precision,
        "attack_recall": attack_recall,
        "attack_f1": attack_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


def print_result_table(results):
    print("\n" + "=" * 100)
    print("[THRESHOLD TUNING RESULT]")
    print("=" * 100)

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

    for r in results:
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


def main():
    print("[LOAD] model")
    model = joblib.load(MODEL_PATH)

    print("[LOAD] scaler")
    scaler = joblib.load(SCALER_PATH)

    print("[LOAD] feature columns")
    feature_cols = joblib.load(FEATURE_COLS_PATH)

    test_files = get_parquet_files(TEST_DIR)

    print("\n[INFO]")
    print(f"Test parts: {len(test_files)}")
    print(f"Feature count: {len(feature_cols)}")
    print(f"Thresholds: {THRESHOLDS}")

    cms = {
        th: np.zeros((2, 2), dtype=np.int64)
        for th in THRESHOLDS
    }

    print("\n[EVALUATE]")

    for i, path in enumerate(test_files):
        X, y = load_xy(path, feature_cols)
        X_scaled = scaler.transform(X)

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_scaled)
            attack_prob = prob[:, 1]
        else:
            raise RuntimeError("현재 모델은 predict_proba를 지원하지 않습니다.")

        y_np = y.to_numpy()

        for th in THRESHOLDS:
            pred = (attack_prob >= th).astype(np.int8)
            cm = confusion_matrix(y_np, pred, labels=[0, 1])
            cms[th] += cm

        if i % 10 == 0:
            print(f"[EVAL] part {i + 1}/{len(test_files)}")

    results = []

    for th in THRESHOLDS:
        metrics = calc_metrics_from_cm(cms[th])
        metrics["threshold"] = th
        results.append(metrics)

    results = sorted(results, key=lambda x: x["threshold"], reverse=True)

    print_result_table(results)

    result_df = pd.DataFrame(results)
    result_df = result_df[
        [
            "threshold",
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "benign_precision",
            "benign_recall",
            "benign_f1",
            "attack_precision",
            "attack_recall",
            "attack_f1",
            "tn",
            "fp",
            "fn",
            "tp",
        ]
    ]

    result_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n[SAVED]")
    print(OUTPUT_CSV)

    print("\n[RECOMMENDATION]")
    print("보안 탐지에서는 FN, 즉 ATTACK을 BENIGN으로 놓치는 수를 줄이는 것이 중요합니다.")
    print("따라서 attack_recall이 충분히 높으면서 accuracy와 attack_precision이 크게 무너지지 않는 threshold를 선택하세요.")
    print("일반적으로 0.35~0.45 구간이 후보가 될 가능성이 높습니다.")


if __name__ == "__main__":
    main()