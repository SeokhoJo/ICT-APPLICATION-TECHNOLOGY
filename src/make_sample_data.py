import os
import glob
import pandas as pd


# =========================
# 1. Path Settings
# =========================

# 원본 프로젝트 데이터 경로
BASE_DIR = r"F:\My Folder\study\2026\ICT\project_Data"

# 전처리 완료된 split parquet 파일들이 있는 경로
SPLIT_DIR = os.path.join(BASE_DIR, "processed", "dl_split_chunked")

# 샘플 데이터 저장 위치
OUTPUT_DIR = os.path.join(BASE_DIR, "sample")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "sample_data.csv")


# =========================
# 2. Sample Settings
# =========================

# 최종 샘플 데이터 개수
SAMPLE_SIZE = 3000

# 재현 가능한 샘플링을 위한 seed
RANDOM_SEED = 42

# 어떤 split에서 샘플을 뽑을지 선택
# train, val, test 중 선택 가능
TARGET_SPLIT = "test"


# =========================
# 3. Load Parquet Files
# =========================

def find_parquet_files(split_dir, target_split):
    """
    전처리된 parquet 파일 목록을 찾는 함수.

    현재 예상 구조:
    dl_split_chunked/
      test/
        part_000.parquet
        part_001.parquet
      train/
      val/
    """

    target_dir = os.path.join(split_dir, target_split)

    patterns = [
        os.path.join(target_dir, "*.parquet"),
        os.path.join(target_dir, "part_*.parquet"),
        os.path.join(target_dir, f"{target_split}_part_*.parquet"),
    ]

    files = []

    for pattern in patterns:
        files = sorted(glob.glob(pattern))
        if files:
            print(f"Parquet pattern matched: {pattern}")
            break

    if not files:
        raise FileNotFoundError(
            "No parquet files found.\n"
            f"Checked directory: {target_dir}\n"
            f"Checked patterns:\n"
            + "\n".join(patterns)
        )

    return files


def load_partial_data(parquet_files, sample_size):
    """
    전체 parquet를 모두 읽지 않고,
    앞쪽 파일 몇 개에서 필요한 양보다 조금 많이 읽어오는 함수.
    """
    dfs = []
    current_rows = 0

    for file in parquet_files:
        print(f"Loading: {file}")

        df_part = pd.read_parquet(file)
        dfs.append(df_part)

        current_rows += len(df_part)

        # 샘플링에 충분한 양을 읽었으면 중단
        if current_rows >= sample_size * 3:
            break

    df = pd.concat(dfs, ignore_index=True)

    return df


# =========================
# 4. Create Sample Data
# =========================

def create_sample_data(df, sample_size, random_seed):
    """
    전처리된 데이터에서 단순 랜덤 샘플을 생성한다.
    Label 컬럼의 NaN 여부와 분포를 확인한다.
    """

    sample_size = min(sample_size, len(df))

    sample_df = df.sample(
        n=sample_size,
        random_state=random_seed
    ).reset_index(drop=True)

    label_col = detect_label_column(sample_df)

    if label_col is not None:
        print(f"Label column detected: {label_col}")

        print("\nSample label distribution:")
        print(sample_df[label_col].value_counts(dropna=False))

        # Label이 비어 있는 행이 있는지 확인
        missing_labels = sample_df[label_col].isna().sum()
        print(f"\nMissing Label count: {missing_labels}")

        if missing_labels > 0:
            raise ValueError(
                f"Sample data contains {missing_labels} missing labels. "
                "Please check the original parquet file or label column."
            )

    else:
        print("No label column detected.")

    return sample_df


def detect_label_column(df):
    """
    Label 컬럼 이름을 자동으로 찾는 함수.
    """
    for candidate in ["Label", "label", "LABEL"]:
        if candidate in df.columns:
            return candidate
    return None

# =========================
# 5. Save Sample Data
# =========================

def save_sample_data(sample_df, output_dir, output_file):
    os.makedirs(output_dir, exist_ok=True)

    sample_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("\nSample data saved successfully.")
    print(f"Output file: {output_file}")
    print(f"Shape: {sample_df.shape}")
    print("\nColumns:")
    print(list(sample_df.columns))


# =========================
# 6. Main
# =========================

def main():
    parquet_files = find_parquet_files(SPLIT_DIR, TARGET_SPLIT)

    print(f"Found {len(parquet_files)} parquet files.")
    print(f"Target split: {TARGET_SPLIT}")

    df = load_partial_data(parquet_files, SAMPLE_SIZE)

    print(f"\nLoaded data shape: {df.shape}")

    sample_df = create_sample_data(
        df=df,
        sample_size=SAMPLE_SIZE,
        random_seed=RANDOM_SEED
    )

    save_sample_data(
        sample_df=sample_df,
        output_dir=OUTPUT_DIR,
        output_file=OUTPUT_FILE
    )


if __name__ == "__main__":
    main()