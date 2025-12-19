#!/usr/bin/env python3
"""
Plot sweep results for quick config comparisons.

Reads sweep_results.csv, picks the best run per (config_id, power_cap) based on
lowest control_p99 (configurable), and generates:
  - Control-class P99 latency vs power cap
  - Power peak vs power cap (cap compliance proxy)
  - Throughput (class0+class1) vs power cap
  - Control latency percentiles (P50/P95/P99) vs power cap per config

Example:
  python3 scripts/plot_sweep_results.py --csv sweep_results.csv --outdir plots
"""

import argparse
import os
import re
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


NUMERIC_COLUMNS = [
    "power_cap",
    "exit_code",
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


def _series_key(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    if not cols:
        return pd.Series(["all"] * len(df), index=df.index)
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.Series(["all"] * len(df), index=df.index)
    return df[cols].astype(str).agg("|".join, axis=1)


def pick_best(group: pd.DataFrame, metric: str, mode: str) -> pd.Series:
    """Pick the best row in a group by a metric (min/max)."""
    if group.empty:
        return pd.Series()
    if metric in group.columns and group[metric].notna().any():
        idx = group[metric].idxmin() if mode == "min" else group[metric].idxmax()
    elif "total_power_peak" in group.columns and group["total_power_peak"].notna().any():
        idx = group["total_power_peak"].idxmin()
    else:
        idx = group.index[0]
    return group.loc[idx]


def best_by_series_cap(
    df: pd.DataFrame,
    series_cols: List[str],
    metric: str,
    mode: str,
    require_success: bool,
) -> pd.DataFrame:
    if "power_cap" not in df.columns:
        raise ValueError("Expected column power_cap in sweep_results.csv")
    work = df.copy()
    if require_success and "exit_code" in work.columns:
        work = work[work["exit_code"].fillna(1) == 0]
    work["__series__"] = _series_key(work, series_cols)
    grouped = work.groupby(["__series__", "power_cap"], dropna=False)
    best_rows = grouped.apply(lambda g: pick_best(g, metric=metric, mode=mode))
    return best_rows.reset_index(drop=True)


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _safe_filename(s: str, max_len: int = 140) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")
    if len(s) <= max_len:
        return s
    return s[:max_len]

def _read_epoch_metrics(run_dir: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Return (active_power_avg, active_throughput_avg) computed from epoch.csv
    using only epochs where total_power > 0.
    """
    def find_in_tree(root: str, filename: str, max_depth: int = 3) -> Optional[str]:
        if os.path.exists(os.path.join(root, filename)):
            return os.path.join(root, filename)
        root_depth = root.rstrip(os.sep).count(os.sep)
        for cur, _, files in os.walk(root):
            depth = cur.rstrip(os.sep).count(os.sep) - root_depth
            if depth > max_depth:
                continue
            if filename in files:
                return os.path.join(cur, filename)
        return None

    epoch_path = find_in_tree(run_dir, "epoch.csv")
    if not epoch_path:
        return None, None
    try:
        df = pd.read_csv(epoch_path)
    except Exception:
        return None, None
    if "total_power" not in df.columns:
        return None, None
    df["total_power"] = pd.to_numeric(df["total_power"], errors="coerce")
    thr_col = None
    for cand in ["class0_throughput", "class0_throughput_flits", "class0_accepted_rate"]:
        if cand in df.columns:
            thr_col = cand
            break
    if thr_col is None:
        return None, None
    df[thr_col] = pd.to_numeric(df[thr_col], errors="coerce")
    active = df[(df["total_power"] > 0) & df["total_power"].notna() & df[thr_col].notna()]
    if active.empty:
        return None, None
    return float(active["total_power"].mean()), float(active[thr_col].mean())


def _read_powerlog_freq(run_dir: str) -> Optional[float]:
    """
    Return average domain0 frequency scale over power_log entries with total_power > 0.
    """
    def find_in_tree(root: str, filename: str, max_depth: int = 3) -> Optional[str]:
        if os.path.exists(os.path.join(root, filename)):
            return os.path.join(root, filename)
        root_depth = root.rstrip(os.sep).count(os.sep)
        for cur, _, files in os.walk(root):
            depth = cur.rstrip(os.sep).count(os.sep) - root_depth
            if depth > max_depth:
                continue
            if filename in files:
                return os.path.join(cur, filename)
        return None

    path = find_in_tree(run_dir, "power_log")
    if not path:
        return None
    total_power_re = re.compile(r"total_power=([0-9.+eE-]+)")
    freq_re = re.compile(r"domains\{0:freq=([0-9.+eE-]+)")
    freqs: List[float] = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m_tp = total_power_re.search(line)
                m_f = freq_re.search(line)
                if not m_tp or not m_f:
                    continue
                try:
                    tp = float(m_tp.group(1))
                    fr = float(m_f.group(1))
                except ValueError:
                    continue
                if tp > 0:
                    freqs.append(fr)
    except Exception:
        return None
    if not freqs:
        return None
    return float(sum(freqs) / len(freqs))


def enrich_with_run_metrics(best: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-run, file-derived metrics used for more meaningful tradeoff plots:
      - active_power_avg (epoch.csv, total_power>0)
      - active_throughput_avg (epoch.csv, total_power>0)
      - active_freq_avg (power_log, total_power>0)
    """
    work = best.copy()
    if "output_dir" not in work.columns or "run_name" not in work.columns:
        return work
    active_power = []
    active_thr = []
    active_freq = []
    for _, row in work.iterrows():
        run_dir = os.path.join(str(row.get("output_dir", "sims")), str(row.get("run_name", "")))
        p_avg, t_avg = _read_epoch_metrics(run_dir)
        active_power.append(p_avg)
        active_thr.append(t_avg)
        active_freq.append(_read_powerlog_freq(run_dir))
    work["active_power_avg"] = pd.to_numeric(active_power, errors="coerce")
    work["active_throughput_avg"] = pd.to_numeric(active_thr, errors="coerce")
    work["active_freq_avg"] = pd.to_numeric(active_freq, errors="coerce")
    return work


def plot_control_p99(best: pd.DataFrame, outdir: str, label: str) -> str:
    plt.figure(figsize=(6, 4))
    if "control_p99" not in best.columns or best["control_p99"].isna().all():
        return ""
    for cfg, sub in best.groupby(label):
        sub = sub.sort_values("power_cap")
        if sub["control_p99"].notna().any():
            plt.plot(sub["power_cap"], sub["control_p99"], marker="o", label=cfg)
    plt.xlabel("Power cap")
    plt.ylabel("Control P99 latency")
    plt.title("Control-class P99 vs power cap (best per series)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    out = os.path.join(outdir, "control_p99_vs_power_cap.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_power_compliance(best: pd.DataFrame, outdir: str, label: str) -> str:
    plt.figure(figsize=(6, 4))
    if "total_power_peak" not in best.columns or best["total_power_peak"].isna().all():
        return ""
    for cfg, sub in best.groupby(label):
        sub = sub.sort_values("power_cap")
        plt.plot(sub["power_cap"], sub["total_power_peak"], marker="o", label=f"{cfg} peak")
    caps = sorted([c for c in best["power_cap"].dropna().unique().tolist()])
    if caps:
        plt.plot(caps, caps, color="k", linestyle="--", label="cap")
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


def plot_throughput(best: pd.DataFrame, outdir: str, label: str) -> str:
    plt.figure(figsize=(6, 4))
    if ("throughput_flits_class0" not in best.columns
            and "throughput_flits_class1" not in best.columns):
        return ""
    for cfg, sub in best.groupby(label):
        sub = sub.sort_values("power_cap")
        thr = sub.get("throughput_flits_class0", 0).fillna(0) + sub.get("throughput_flits_class1", 0).fillna(0)
        if thr.notna().any():
            plt.plot(sub["power_cap"], thr, marker="o", label=cfg)
    plt.xlabel("Power cap")
    plt.ylabel("Throughput (flits/cycle)")
    plt.title("Throughput vs power cap (best per series)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    out = os.path.join(outdir, "throughput_vs_power_cap.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_throughput_vs_power(best: pd.DataFrame, outdir: str, label: str) -> str:
    plt.figure(figsize=(6, 4))
    if "active_power_avg" not in best.columns or "active_throughput_avg" not in best.columns:
        return ""
    if best["active_power_avg"].isna().all() or best["active_throughput_avg"].isna().all():
        return ""
    for cfg, sub in best.groupby(label):
        sub = sub.sort_values("active_power_avg")
        if sub["active_power_avg"].notna().any() and sub["active_throughput_avg"].notna().any():
            plt.plot(sub["active_power_avg"], sub["active_throughput_avg"], marker="o", label=cfg)
    plt.xlabel("Avg power (active epochs)")
    plt.ylabel("Throughput (active epochs)")
    plt.title("Throughput vs observed power (active epochs)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    out = os.path.join(outdir, "throughput_vs_active_power.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_throughput_vs_freq(best: pd.DataFrame, outdir: str, label: str) -> str:
    plt.figure(figsize=(6, 4))
    if "active_freq_avg" not in best.columns or "active_throughput_avg" not in best.columns:
        return ""
    if best["active_freq_avg"].isna().all() or best["active_throughput_avg"].isna().all():
        return ""
    for cfg, sub in best.groupby(label):
        sub = sub.sort_values("active_freq_avg")
        if sub["active_freq_avg"].notna().any() and sub["active_throughput_avg"].notna().any():
            plt.plot(sub["active_freq_avg"], sub["active_throughput_avg"], marker="o", label=cfg)
    plt.xlabel("Avg freq scale (active epochs)")
    plt.ylabel("Throughput (active epochs)")
    plt.title("Throughput vs observed frequency (active epochs)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    out = os.path.join(outdir, "throughput_vs_active_freq.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_latency_percentiles(best: pd.DataFrame, outdir: str, label: str) -> List[str]:
    out_paths = []
    for cfg, sub in best.groupby(label):
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
        out = os.path.join(outdir, f"latency_percentiles_{_safe_filename(str(cfg))}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        out_paths.append(out)
    return out_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot sweep_results.csv comparisons.")
    parser.add_argument("--csv", default="sweep_results.csv", help="Path to sweep_results.csv")
    parser.add_argument("--outdir", default="plots", help="Directory to write plots")
    parser.add_argument("--series", default="config_id",
                        help="Comma-separated columns to define a series (default: config_id). "
                             "Example: config_id,policy,dvfs_epoch")
    parser.add_argument("--metric", default="control_p99",
                        help="Metric used to pick the best row per (series,power_cap).")
    parser.add_argument("--mode", choices=["min", "max"], default="min",
                        help="Whether to minimize or maximize --metric when selecting best runs.")
    parser.add_argument("--where", action="append", default=[],
                        help="Filter rows in the form col=value (repeatable).")
    parser.add_argument("--include-failures", action="store_true",
                        help="Include non-zero exit_code rows in selection/plots.")
    args = parser.parse_args()

    df = load_sweep(args.csv)
    if df.empty:
        print("No data in CSV; nothing to plot.")
        return

    # Apply filters
    for clause in args.where:
        if "=" not in clause:
            raise ValueError(f"Bad --where format (expected col=value): {clause}")
        col, val = clause.split("=", 1)
        if col not in df.columns:
            raise ValueError(f"--where column not found: {col}")
        df = df[df[col].astype(str) == val]

    ensure_outdir(args.outdir)

    series_cols = [c.strip() for c in args.series.split(",") if c.strip()]
    best = best_by_series_cap(
        df,
        series_cols=series_cols,
        metric=args.metric,
        mode=args.mode,
        require_success=not args.include_failures,
    )
    label_col = "__series__"
    best = enrich_with_run_metrics(best)
    outputs = []
    outputs.append(plot_control_p99(best, args.outdir, label=label_col))
    outputs.append(plot_power_compliance(best, args.outdir, label=label_col))
    outputs.append(plot_throughput(best, args.outdir, label=label_col))
    outputs.append(plot_throughput_vs_power(best, args.outdir, label=label_col))
    outputs.append(plot_throughput_vs_freq(best, args.outdir, label=label_col))
    outputs.extend(plot_latency_percentiles(best, args.outdir, label=label_col))

    print("Wrote plots:")
    for p in outputs:
        if p:
            print("  ", p)


if __name__ == "__main__":
    main()
