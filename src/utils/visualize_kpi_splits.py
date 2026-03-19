import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def find_anomaly_segments(labels: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start = None

    for idx, label in enumerate(labels):
        if label == 1 and start is None:
            start = idx
        elif label == 0 and start is not None:
            segments.append((start, idx - 1))
            start = None

    if start is not None:
        segments.append((start, len(labels) - 1))

    return segments


def rolling_trend(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(values)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def build_full_kpi_frame(
    kpi_id: str,
    train_source: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    train_part = (
        train_source.loc[train_source["KPI ID"].astype(str) == kpi_id]
        .sort_values("timestamp")
        .copy()
    )
    train_part["segment"] = "train"

    gt_part = (
        ground_truth.loc[ground_truth["KPI ID"].astype(str) == kpi_id]
        .sort_values("timestamp")
        .copy()
    )
    gt_part["segment"] = "ground_truth"

    full_df = pd.concat([train_part, gt_part], ignore_index=True)
    full_df["datetime"] = pd.to_datetime(full_df["timestamp"], unit="s")

    train_end = len(train_part)
    gt_total = len(gt_part)
    finetune_train_end = gt_total // 2
    finetune_test_end = finetune_train_end + (gt_total - finetune_train_end) // 2

    split_positions = {
        "train_end": train_end,
        "finetune_train_end": train_end + finetune_train_end,
        "finetune_test_end": train_end + finetune_test_end,
    }
    return full_df, split_positions


def add_split_lines(fig, full_df: pd.DataFrame, split_positions: dict[str, int]) -> None:
    x_values = full_df["datetime"].to_numpy()
    for label, pos in split_positions.items():
        if 0 < pos < len(x_values):
            x = x_values[pos]
            fig.add_vline(x=x, line_width=1, line_dash="dash", line_color="gray")
            fig.add_annotation(
                x=x,
                y=1.0,
                yref="paper",
                text=label,
                showarrow=False,
                yshift=8,
                font=dict(size=10, color="gray"),
            )


def create_plot(
    kpi_id: str,
    full_df: pd.DataFrame,
    split_positions: dict[str, int],
    output_path: Path,
) -> None:
    values = full_df["value"].to_numpy(dtype=float)
    labels = full_df["label"].to_numpy(dtype=int)
    datetimes = full_df["datetime"]
    trend_window = max(11, len(full_df) // 150)
    if trend_window % 2 == 0:
        trend_window += 1
    trend = rolling_trend(values, trend_window)
    anomaly_segments = find_anomaly_segments(labels)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("Trajectory and Trend", "Label Timeline"),
        row_heights=[0.78, 0.22],
    )

    fig.add_trace(
        go.Scatter(
            x=datetimes,
            y=values,
            name="Value",
            mode="lines",
            line=dict(color="#1f77b4", width=1.2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=datetimes,
            y=trend,
            name=f"Trend (rolling={trend_window})",
            mode="lines",
            line=dict(color="#ff7f0e", width=2),
        ),
        row=1,
        col=1,
    )

    anomaly_mask = labels == 1
    if anomaly_mask.any():
        fig.add_trace(
            go.Scatter(
                x=datetimes[anomaly_mask],
                y=values[anomaly_mask],
                name="Anomaly points",
                mode="markers",
                marker=dict(color="crimson", size=5),
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=datetimes,
            y=labels,
            name="Label",
            mode="lines",
            line=dict(color="crimson", width=1.5, dash="dot"),
            fill="tozeroy",
            fillcolor="rgba(220,20,60,0.15)",
        ),
        row=2,
        col=1,
    )

    for start, end in anomaly_segments:
        fig.add_vrect(
            x0=datetimes.iloc[start],
            x1=datetimes.iloc[end],
            fillcolor="rgba(220,20,60,0.12)",
            line_width=0,
            layer="below",
            row="all",
            col=1,
        )

    add_split_lines(fig, full_df, split_positions)

    anomaly_points = int(labels.sum())
    normal_points = int((labels == 0).sum())
    fig.update_layout(
        title=(
            f"{kpi_id} | total={len(full_df)} | normal={normal_points} | "
            f"anomaly={anomaly_points}"
        ),
        height=760,
        legend_traceorder="normal",
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(
        title_text="Label",
        row=2,
        col=1,
        tickmode="array",
        tickvals=[0, 1],
        ticktext=["Normal (0)", "Anomaly (1)"],
    )
    fig.update_xaxes(title_text="Timestamp", row=2, col=1)
    fig.write_html(output_path)


def main(args):
    output_root = Path(args.output_dir)
    train_source = pd.read_csv(args.train_csv)
    ground_truth = pd.read_hdf(args.ground_truth_hdf, key=args.ground_truth_key)

    produced = 0
    for kpi_dir in sorted(output_root.iterdir()):
        if not kpi_dir.is_dir():
            continue

        kpi_id = kpi_dir.name
        full_df, split_positions = build_full_kpi_frame(kpi_id, train_source, ground_truth)
        if full_df.empty:
            continue

        output_path = kpi_dir / args.output_name
        create_plot(kpi_id, full_df, split_positions, output_path)
        produced += 1
        print(f"Saved {output_path}")

    print(f"Generated plots for {produced} KPI directories.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize each KPI split with trajectory, trend, labels, and split boundaries."
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default="KPI-Anomaly-Detection-master/Finals_dataset/unpacked/phase2_train.csv",
        help="Official phase2 train CSV.",
    )
    parser.add_argument(
        "--ground_truth_hdf",
        type=str,
        default="KPI-Anomaly-Detection-master/Finals_dataset/unpacked/phase2_ground_truth.hdf",
        help="Official phase2 ground truth HDF.",
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
        help="Directory containing one folder per KPI ID.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="overview.html",
        help="HTML filename written inside each KPI directory.",
    )
    main(parser.parse_args())
