#!/usr/bin/env python3
"""
Plot sweep results for quick config comparisons.

Reads sweep_results.csv, picks the best run per (config_id, power_cap) based on
lowest control_p99, and generates:
  - Control-class P99 latency vs power cap
  - Power peak vs power cap (cap compliance proxy)
  - Throughput (class0+class1) vs power cap
  - Control latency percentiles (P50/P95/P99) vs power cap per config

Example:
  python3 scripts/plot_sweep_results.py --csv sweep_results.csv --outdir plots
"""

import argparse
import os
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import pandas as pd


NUMERIC_COLUMNS = [
    "power_cap",
    "control_p50",
    "control_p95",
    "control_p99",
    "batch_p50",
    "batch_p95",
    "batch_p99",
    "total_power_avg",
    "total_power_peak",
    "throughput_flits_class0",
    "throughput_flits_class1",
]


def load_sweep(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize useful numeric columns
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            # Ensure missing columns exist (filled with NaN) so plotting code can skip gracefully
            df[col] = pd.NA
    return df


def pick_best(group: pd.DataFrame) -> pd.Series:
    """Pick the best row in a group (lowest control_p99)."""
    if group.empty:
        return pd.Series()
    if "control_p99" in group.columns and group["control_p99"].notna().any():
        idx = group["control_p99"].idxmin()
    else:
        # Fallback: lowest total_power_peak
        idx = group["total_power_peak"].idxmin()
    return group.loc[idx]


def best_by_config_cap(df: pd.DataFrame) -> pd.DataFrame:
    if "config_id" not in df.columns or "power_cap" not in df.columns:
        raise ValueError("Expected columns config_id and power_cap in sweep_results.csv")
    grouped = df.groupby(["config_id", "power_cap"], dropna=False)
    best_rows = grouped.apply(pick_best)
    return best_rows.reset_index(drop=True)


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_control_p99(best: pd.DataFrame, outdir: str) -> str:
    plt.figure(figsize=(6, 4))
    if "control_p99" not in best.columns or best["control_p99"].isna().all():
        return ""
    for cfg, sub in best.groupby("config_id"):
        sub = sub.sort_values("power_cap")
        if sub["control_p99"].notna().any():
            plt.plot(sub["power_cap"], sub["control_p99"], marker="o", label=cfg)
    plt.xlabel("Power cap")
    plt.ylabel("Control P99 latency")
    plt.title("Control-class P99 vs power cap (best per config)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    out = os.path.join(outdir, "control_p99_vs_power_cap.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_power_compliance(best: pd.DataFrame, outdir: str) -> str:
    plt.figure(figsize=(6, 4))
    if "total_power_peak" not in best.columns or best["total_power_peak"].isna().all():
        return ""
    for cfg, sub in best.groupby("config_id"):
        sub = sub.sort_values("power_cap")
        plt.plot(sub["power_cap"], sub["total_power_peak"], marker="o", label=f"{cfg} peak")
    plt.plot(best["power_cap"], best["power_cap"], color="k", linestyle="--", label="cap")
    plt.xlabel("Power cap")
    plt.ylabel("Peak power")
    plt.title("Power vs cap (peak as compliance proxy)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    out = os.path.join(outdir, "power_peak_vs_cap.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_throughput(best: pd.DataFrame, outdir: str) -> str:
    plt.figure(figsize=(6, 4))
    if ("throughput_flits_class0" not in best.columns
            and "throughput_flits_class1" not in best.columns):
        return ""
    for cfg, sub in best.groupby("config_id"):
        sub = sub.sort_values("power_cap")
        thr = sub.get("throughput_flits_class0", 0).fillna(0) + sub.get("throughput_flits_class1", 0).fillna(0)
        if thr.notna().any():
            plt.plot(sub["power_cap"], thr, marker="o", label=cfg)
    plt.xlabel("Power cap")
    plt.ylabel("Throughput (flits/cycle)")
    plt.title("Throughput vs power cap (best per config)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    out = os.path.join(outdir, "throughput_vs_power_cap.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_latency_percentiles(best: pd.DataFrame, outdir: str) -> str:
    out_paths = []
    for cfg, sub in best.groupby("config_id"):
        sub = sub.sort_values("power_cap")
        plt.figure(figsize=(6, 4))
        for col, label in [
            ("control_p50", "P50"),
            ("control_p95", "P95"),
            ("control_p99", "P99"),
        ]:
            if col in sub.columns and sub[col].notna().any():
                plt.plot(sub["power_cap"], sub[col], marker="o", label=label)
        plt.xlabel("Power cap")
        plt.ylabel("Control latency")
        plt.title(f"Control latency percentiles vs power cap ({cfg})")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        out = os.path.join(outdir, f"latency_percentiles_{cfg}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        out_paths.append(out)
    return out_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot sweep_results.csv comparisons.")
    parser.add_argument("--csv", default="sweep_results.csv", help="Path to sweep_results.csv")
    parser.add_argument("--outdir", default="plots", help="Directory to write plots")
    args = parser.parse_args()

    df = load_sweep(args.csv)
    if df.empty:
        print("No data in CSV; nothing to plot.")
        return
    ensure_outdir(args.outdir)

    best = best_by_config_cap(df)
    outputs = []
    outputs.append(plot_control_p99(best, args.outdir))
    outputs.append(plot_power_compliance(best, args.outdir))
    outputs.append(plot_throughput(best, args.outdir))
    outputs.extend(plot_latency_percentiles(best, args.outdir))

    print("Wrote plots:")
    for p in outputs:
        print("  ", p)


if __name__ == "__main__":
    main()
