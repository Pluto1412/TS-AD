import argparse
from pathlib import Path

import pandas as pd


def sanitize_kpi_id(kpi_id: str) -> str:
    return str(kpi_id).replace("/", "_")


def split_single_kpi(
    train_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    finetune_train_ratio: float,
    min_train_points: int,
    min_eval_points: int,
):
    train_df = train_df.sort_values("timestamp").reset_index(drop=True)
    gt_df = gt_df.sort_values("timestamp").reset_index(drop=True)

    train_total = len(train_df)
    gt_total = len(gt_df)
    if train_total < min_train_points or gt_total < (min_eval_points * 2 + 1):
        return None

    finetune_train_end = int(gt_total * finetune_train_ratio)

    # Keep at least one point in finetune train and enough room for two eval splits.
    finetune_train_end = max(1, min(finetune_train_end, gt_total - 2 * min_eval_points))
    remain = gt_total - finetune_train_end
    finetune_test_size = remain // 2
    test_size = remain - finetune_test_size
    if finetune_test_size < min_eval_points or test_size < min_eval_points:
        return None

    finetune_test_end = finetune_train_end + finetune_test_size

    pretrain_df = train_df[["value"]]
    finetune_train_df = gt_df.iloc[:finetune_train_end][["value"]]
    finetune_test_df = gt_df.iloc[finetune_train_end:finetune_test_end][["value", "label"]]
    test_df = gt_df.iloc[finetune_test_end:][["value", "label"]]

    if (
        len(pretrain_df) == 0
        or len(finetune_train_df) == 0
        or len(finetune_test_df) == 0
        or len(test_df) == 0
    ):
        return None

    return pretrain_df, finetune_train_df, finetune_test_df, test_df


def main(args):
    train_path = Path(args.train_csv)
    ground_truth_path = Path(args.ground_truth_hdf)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    train_source = pd.read_csv(train_path)
    ground_truth = pd.read_hdf(ground_truth_path, key=args.ground_truth_key)
    required_columns = {"timestamp", "value", "label", "KPI ID"}
    missing_train = required_columns - set(train_source.columns)
    missing_ground_truth = required_columns - set(ground_truth.columns)
    if missing_train:
        raise ValueError(f"Train source missing columns: {sorted(missing_train)}")
    if missing_ground_truth:
        raise ValueError(f"Ground truth missing columns: {sorted(missing_ground_truth)}")

    train_groups = {
        str(kpi_id): kpi_df.copy() for kpi_id, kpi_df in train_source.groupby("KPI ID")
    }
    gt_groups = {
        str(kpi_id): kpi_df.copy() for kpi_id, kpi_df in ground_truth.groupby("KPI ID")
    }

    summary_rows = []
    produced = 0

    for kpi_id in sorted(set(train_groups) & set(gt_groups)):
        split_result = split_single_kpi(
            train_groups[kpi_id],
            gt_groups[kpi_id],
            finetune_train_ratio=args.finetune_train_ratio,
            min_train_points=args.min_train_points,
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
                "train_source_points": len(train_groups[kpi_id]),
                "ground_truth_points": len(gt_groups[kpi_id]),
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

    print(f"Train source: {train_path}")
    print(f"Ground truth: {ground_truth_path} (key={args.ground_truth_key})")
    print(f"Output root: {output_root}")
    print(f"KPI directories produced: {produced}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert KPI-Anomaly-Detection phase2 train + ground truth into TS-AD per-KPI split format."
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default="KPI-Anomaly-Detection-master/Finals_dataset/unpacked/phase2_train.csv",
        help="Path to official phase2 train CSV with columns: timestamp,value,label,KPI ID",
    )
    parser.add_argument(
        "--ground_truth_hdf",
        type=str,
        default="KPI-Anomaly-Detection-master/Finals_dataset/unpacked/phase2_ground_truth.hdf",
        help="Path to official phase2 ground truth HDF.",
    )
    parser.add_argument(
        "--ground_truth_key",
        type=str,
        default="data",
        help="HDF key containing the ground truth table.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/kpi_splits",
        help="Output directory containing one folder per KPI ID.",
    )
    parser.add_argument(
        "--finetune_train_ratio",
        type=float,
        default=0.5,
        help="Ratio used for finetune_train.csv split end inside the ground truth segment.",
    )
    parser.add_argument(
        "--min_train_points",
        type=int,
        default=500,
        help="Skip KPI series whose official train segment has fewer than this number of points.",
    )
    parser.add_argument(
        "--min_eval_points",
        type=int,
        default=200,
        help="Minimum required points for both finetune_test.csv and test.csv after splitting the ground truth segment.",
    )
    main(parser.parse_args())
