import os
import glob
import shutil
import numpy as np
import pandas as pd

BASE_DIR = r"F:\My Folder\study\2026\ICT\project_Data"

DATA_DIRS = {
    "2017": os.path.join(BASE_DIR, "2017"),
    "2018": os.path.join(BASE_DIR, "2018"),
    "2019": os.path.join(BASE_DIR, "2019"),
}

OUTPUT_DIR = os.path.join(BASE_DIR, "processed", "dl_split_chunked")
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_DIR = os.path.join(OUTPUT_DIR, "val")
TEST_DIR = os.path.join(OUTPUT_DIR, "test")

CHUNK_SIZE = 200_000
RANDOM_SEED = 42

RENAME_DICT = {
    "Destination Port": "Dst Port",
    "Total Fwd Packets": "Tot Fwd Pkts",
    "Total Backward Packets": "Tot Bwd Pkts",
    "Total Length of Fwd Packets": "TotLen Fwd Pkts",
    "Total Length of Bwd Packets": "TotLen Bwd Pkts",
    "Fwd Packet Length Max": "Fwd Pkt Len Max",
    "Fwd Packet Length Min": "Fwd Pkt Len Min",
    "Fwd Packet Length Mean": "Fwd Pkt Len Mean",
    "Fwd Packet Length Std": "Fwd Pkt Len Std",
    "Bwd Packet Length Max": "Bwd Pkt Len Max",
    "Bwd Packet Length Min": "Bwd Pkt Len Min",
    "Bwd Packet Length Mean": "Bwd Pkt Len Mean",
    "Bwd Packet Length Std": "Bwd Pkt Len Std",
    "Flow Bytes/s": "Flow Byts/s",
    "Flow Packets/s": "Flow Pkts/s",
    "Fwd IAT Total": "Fwd IAT Tot",
    "Bwd IAT Total": "Bwd IAT Tot",
    "Fwd Header Length": "Fwd Header Len",
    "Fwd Header Length.1": "Fwd Header Len",
    "Bwd Header Length": "Bwd Header Len",
    "Fwd Packets/s": "Fwd Pkts/s",
    "Bwd Packets/s": "Bwd Pkts/s",
    "Min Packet Length": "Pkt Len Min",
    "Max Packet Length": "Pkt Len Max",
    "Packet Length Mean": "Pkt Len Mean",
    "Packet Length Std": "Pkt Len Std",
    "Packet Length Variance": "Pkt Len Var",
    "FIN Flag Count": "FIN Flag Cnt",
    "SYN Flag Count": "SYN Flag Cnt",
    "RST Flag Count": "RST Flag Cnt",
    "PSH Flag Count": "PSH Flag Cnt",
    "ACK Flag Count": "ACK Flag Cnt",
    "URG Flag Count": "URG Flag Cnt",
    "ECE Flag Count": "ECE Flag Cnt",
    "Average Packet Size": "Pkt Size Avg",
    "Avg Fwd Segment Size": "Fwd Seg Size Avg",
    "Avg Bwd Segment Size": "Bwd Seg Size Avg",
    "Fwd Avg Bytes/Bulk": "Fwd Byts/b Avg",
    "Fwd Avg Packets/Bulk": "Fwd Pkts/b Avg",
    "Fwd Avg Bulk Rate": "Fwd Blk Rate Avg",
    "Bwd Avg Bytes/Bulk": "Bwd Byts/b Avg",
    "Bwd Avg Packets/Bulk": "Bwd Pkts/b Avg",
    "Bwd Avg Bulk Rate": "Bwd Blk Rate Avg",
    "Subflow Fwd Packets": "Subflow Fwd Pkts",
    "Subflow Fwd Bytes": "Subflow Fwd Byts",
    "Subflow Bwd Packets": "Subflow Bwd Pkts",
    "Subflow Bwd Bytes": "Subflow Bwd Byts",
    "Init_Win_bytes_forward": "Init Fwd Win Byts",
    "Init_Win_bytes_backward": "Init Bwd Win Byts",
    "act_data_pkt_fwd": "Fwd Act Data Pkts",
    "min_seg_size_forward": "Fwd Seg Size Min",
}

DROP_COLUMNS = [
    "Unnamed: 0", "Flow ID", "Source IP", "Source Port",
    "Destination IP", "Protocol", "Timestamp",
    "SimillarHTTP", "Inbound"
]

COMMON_COLUMNS = [
    "Dst Port", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
    "TotLen Fwd Pkts", "TotLen Bwd Pkts",
    "Fwd Pkt Len Max", "Fwd Pkt Len Min", "Fwd Pkt Len Mean", "Fwd Pkt Len Std",
    "Bwd Pkt Len Max", "Bwd Pkt Len Min", "Bwd Pkt Len Mean", "Bwd Pkt Len Std",
    "Flow Byts/s", "Flow Pkts/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Tot", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Len", "Bwd Header Len",
    "Fwd Pkts/s", "Bwd Pkts/s",
    "Pkt Len Min", "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var",
    "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt",
    "ACK Flag Cnt", "URG Flag Cnt", "CWE Flag Count", "ECE Flag Cnt",
    "Down/Up Ratio", "Pkt Size Avg", "Fwd Seg Size Avg", "Bwd Seg Size Avg",
    "Fwd Byts/b Avg", "Fwd Pkts/b Avg", "Fwd Blk Rate Avg",
    "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg",
    "Subflow Fwd Pkts", "Subflow Fwd Byts", "Subflow Bwd Pkts", "Subflow Bwd Byts",
    "Init Fwd Win Byts", "Init Bwd Win Byts",
    "Fwd Act Data Pkts", "Fwd Seg Size Min",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
    "Label"
]


def reset_output_dirs():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)


def normalize_label(x):
    return 0 if str(x).strip().upper() == "BENIGN" else 1


def process_chunk(chunk, year, source_file):
    chunk.columns = chunk.columns.str.strip()
    chunk = chunk.rename(columns=RENAME_DICT)
    chunk = chunk.loc[:, ~chunk.columns.duplicated()]
    chunk = chunk.drop(columns=[c for c in DROP_COLUMNS if c in chunk.columns], errors="ignore")

    missing = [c for c in COMMON_COLUMNS if c not in chunk.columns]
    if missing:
        return None, missing

    chunk = chunk[COMMON_COLUMNS]
    chunk = chunk.replace([np.inf, -np.inf], np.nan)

    for c in COMMON_COLUMNS:
        if c != "Label":
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce")

    chunk = chunk.dropna()

    if len(chunk) == 0:
        return None, []

    chunk["Label"] = chunk["Label"].apply(normalize_label).astype(np.int8)
    chunk["DatasetYear"] = str(year)
    chunk["SourceFile"] = source_file

    # 용량 줄이기
    for c in COMMON_COLUMNS:
        if c != "Label":
            chunk[c] = chunk[c].astype(np.float32)

    return chunk, []


def save_split_parquet(df, file_index):
    r = np.random.rand(len(df))

    train_df = df[r < 0.7]
    val_df = df[(r >= 0.7) & (r < 0.9)]
    test_df = df[r >= 0.9]

    if len(train_df) > 0:
        train_df.to_parquet(
            os.path.join(TRAIN_DIR, f"train_part_{file_index:05d}.parquet"),
            index=False
        )

    if len(val_df) > 0:
        val_df.to_parquet(
            os.path.join(VAL_DIR, f"val_part_{file_index:05d}.parquet"),
            index=False
        )

    if len(test_df) > 0:
        test_df.to_parquet(
            os.path.join(TEST_DIR, f"test_part_{file_index:05d}.parquet"),
            index=False
        )

    return len(train_df), len(val_df), len(test_df)


def main():
    np.random.seed(RANDOM_SEED)

    reset_output_dirs()

    total_train = 0
    total_val = 0
    total_test = 0
    part_index = 0

    total_label_count = {0: 0, 1: 0}

    for year, folder in DATA_DIRS.items():
        csv_files = glob.glob(os.path.join(folder, "*.csv"))

        print("\n" + "=" * 70)
        print(f"[{year}] CSV files: {len(csv_files)}")
        print("=" * 70)

        for path in csv_files:
            source_file = os.path.basename(path)
            print(f"\n[READ] {year} | {source_file}")

            try:
                reader = pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False)

                for chunk_idx, chunk in enumerate(reader):
                    processed, missing = process_chunk(chunk, year, source_file)

                    if missing:
                        print(f"[SKIP FILE] missing columns: {missing}")
                        break

                    if processed is None or len(processed) == 0:
                        continue

                    label_counts = processed["Label"].value_counts().to_dict()
                    total_label_count[0] += label_counts.get(0, 0)
                    total_label_count[1] += label_counts.get(1, 0)

                    tr_n, va_n, te_n = save_split_parquet(processed, part_index)
                    total_train += tr_n
                    total_val += va_n
                    total_test += te_n

                    if part_index % 10 == 0:
                        print(
                            f"part={part_index:05d} | "
                            f"train={total_train:,}, val={total_val:,}, test={total_test:,}"
                        )

                    part_index += 1

            except Exception as e:
                print(f"[ERROR] {source_file}")
                print(e)

    print("\n" + "=" * 70)
    print("[DONE]")
    print("=" * 70)

    print(f"Train rows: {total_train:,}")
    print(f"Val rows  : {total_val:,}")
    print(f"Test rows : {total_test:,}")
    print(f"Total rows: {total_train + total_val + total_test:,}")

    print("\n[Total Label Count]")
    print(f"BENIGN: {total_label_count[0]:,}")
    print(f"ATTACK: {total_label_count[1]:,}")

    print("\n[SAVED DIR]")
    print(TRAIN_DIR)
    print(VAL_DIR)
    print(TEST_DIR)


if __name__ == "__main__":
    main()