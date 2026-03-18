import argparse
import os
from pathlib import Path

import pandas as pd


def sanitize_kpi_id(kpi_id: str) -> str:
    return str(kpi_id).replace("/", "_")


def split_single_kpi(
    kpi_df: pd.DataFrame,
    pretrain_ratio: float,
    finetune_ratio: float,
    min_points: int,
    min_eval_points: int,
):
    kpi_df = kpi_df.sort_values("timestamp").reset_index(drop=True)
    total = len(kpi_df)
    if total < min_points:
        return None

    pretrain_end = int(total * pretrain_ratio)
    finetune_end = int(total * finetune_ratio)

    # Keep at least one point in each split.
    pretrain_end = max(1, min(pretrain_end, total - 3))
    finetune_end = max(pretrain_end + 1, min(finetune_end, total - 2))
    remain = total - finetune_end
    finetune_test_size = remain // 2
    test_size = remain - finetune_test_size
    if finetune_test_size < min_eval_points or test_size < min_eval_points:
        return None

    finetune_test_end = finetune_end + finetune_test_size

    train_df = kpi_df.iloc[:pretrain_end][["value"]]
    finetune_train_df = kpi_df.iloc[pretrain_end:finetune_end][["value"]]
    finetune_test_df = kpi_df.iloc[finetune_end:finetune_test_end][["value", "label"]]
    test_df = kpi_df.iloc[finetune_test_end:][["value", "label"]]

    if (
        len(train_df) == 0
        or len(finetune_train_df) == 0
        or len(finetune_test_df) == 0
        or len(test_df) == 0
    ):
        return None

    return train_df, finetune_train_df, finetune_test_df, test_df


def main(args):
    input_path = Path(args.input_csv)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    required_columns = {"timestamp", "value", "label", "KPI ID"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    summary_rows = []
    produced = 0

    for kpi_id, kpi_df in df.groupby("KPI ID"):
        split_result = split_single_kpi(
            kpi_df,
            pretrain_ratio=args.pretrain_ratio,
            finetune_ratio=args.finetune_ratio,
            min_points=args.min_points,
            min_eval_points=args.min_eval_points,
        )
        if split_result is None:
            continue

        train_df, finetune_train_df, finetune_test_df, test_df = split_result
        kpi_dir = output_root / sanitize_kpi_id(kpi_id)
        kpi_dir.mkdir(parents=True, exist_ok=True)

        train_df.to_csv(kpi_dir / "train.csv", index=False)
        finetune_train_df.to_csv(kpi_dir / "finetune_train.csv", index=False)
        finetune_test_df.to_csv(kpi_dir / "finetune_test.csv", index=False)
        test_df.to_csv(kpi_dir / "test.csv", index=False)

        summary_rows.append(
            {
                "kpi_id": kpi_id,
                "total_points": len(kpi_df),
                "train_points": len(train_df),
                "finetune_train_points": len(finetune_train_df),
                "finetune_test_points": len(finetune_test_df),
                "test_points": len(test_df),
                "anomalies_finetune_test": int(finetune_test_df["label"].sum()),
                "anomalies_test": int(test_df["label"].sum()),
                "output_dir": str(kpi_dir),
            }
        )
        produced += 1

    summary_df = pd.DataFrame(summary_rows).sort_values("kpi_id")
    summary_path = output_root / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"Input: {input_path}")
    print(f"Output root: {output_root}")
    print(f"KPI directories produced: {produced}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert KPI-Anomaly-Detection dataset into TS-AD per-KPI split format."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="KPI-Anomaly-Detection-master/Finals_dataset/unpacked/phase2_train.csv",
        help="Path to labeled source CSV with columns: timestamp,value,label,KPI ID",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/kpi_splits",
        help="Output directory containing one folder per KPI ID.",
    )
    parser.add_argument(
        "--pretrain_ratio",
        type=float,
        default=0.7,
        help="Ratio used for train.csv split end.",
    )
    parser.add_argument(
        "--finetune_ratio",
        type=float,
        default=0.85,
        help="Ratio used for finetune_train.csv split end.",
    )
    parser.add_argument(
        "--min_points",
        type=int,
        default=500,
        help="Skip KPI series with fewer than this number of points.",
    )
    parser.add_argument(
        "--min_eval_points",
        type=int,
        default=200,
        help="Minimum required points for both finetune_test.csv and test.csv.",
    )
    main(parser.parse_args())
