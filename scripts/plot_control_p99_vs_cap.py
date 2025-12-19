#!/usr/bin/env python3
"""
Clean plot: Control P99 vs Power Cap for all controllers.

Features:
  - One line per policy (hw_reactive, queue_pid, perf_target, uniform)
  - Uniform line is dotted and uses the "best feasible" point per cap
    (filters to total_power_peak <= power_cap, then picks best).
  - Optional filtering by dvfs_epoch and injection_rate for a clean single plot.

Example:
  python3 scripts/plot_control_p99_vs_cap.py \
    --csv sweep_results_all_policies.csv --csv sweep_results_uniform_tuning.csv \
    --dvfs-epoch 10000 --injection-rate 0.5 \
    --slo-p99 232 --out plots
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd


POLICY_ORDER = ["hw_reactive", "queue_pid", "perf_target", "uniform"]


def _norm_policy(p: object) -> str:
    s = str(p or "").strip().lower()
    if not s:
        return "unknown"
    if "uniform" in s:
        return "uniform"
    if "hw" in s and "react" in s:
        return "hw_reactive"
    if "queue" in s and "pid" in s:
        return "queue_pid"
    if "perf" in s and "target" in s:
        return "perf_target"
    return s


def _load_csvs(paths: List[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            continue
        f = pd.read_csv(path)
        f["source_csv"] = str(path)
        frames.append(f)
    if not frames:
        raise SystemExit("No CSVs found. Provide --csv sweep_results_all_policies.csv ...")
    df = pd.concat(frames, ignore_index=True)
    df["policy_norm"] = df.get("policy", pd.Series(["unknown"] * len(df))).apply(_norm_policy)
    for c in [
        "power_cap",
        "dvfs_epoch",
        "injection_rate",
        "exit_code",
        "control_p99",
        "total_power_peak",
        "uniform_target",
        "throughput_flits_class0",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _require_single_injection(df: pd.DataFrame, inj: Optional[float]) -> pd.DataFrame:
    if "injection_rate" not in df.columns or df["injection_rate"].isna().all():
        return df
    if inj is not None:
        return df[df["injection_rate"] == inj]
    vals = sorted(df["injection_rate"].dropna().unique().tolist())
    if len(vals) <= 1:
        return df
    raise SystemExit(
        f"Multiple injection_rate values present {vals}. "
        "Use --injection-rate <val> to select one for a clean plot."
    )


def _pick_per_policy_cap(
    df: pd.DataFrame,
    *,
    uniform_cap_metric: str,
    uniform_cap_eps: float,
    uniform_pick: str,
    uniform_source_csv: Optional[str],
) -> pd.DataFrame:
    rows = []
    group_cols = ["policy_norm", "power_cap"]
    saw_uniform = False
    kept_uniform = 0

    for (_, _), g in df.groupby(group_cols, dropna=False):
        if g.empty:
            continue
        pol = str(g["policy_norm"].iloc[0])
        cap = g["power_cap"].iloc[0]

        # Default: pick minimal control_p99 (best latency) for a given cap
        candidates = g.dropna(subset=["control_p99"])

        if pol == "uniform":
            saw_uniform = True
            ug = g
            if uniform_source_csv is not None and "source_csv" in ug.columns:
                ug = ug[ug["source_csv"] == uniform_source_csv]
            elif "uniform_target" in ug.columns and ug["uniform_target"].notna().any():
                # If a uniform_target sweep is present, prefer those rows (avoids mixing in
                # uniform runs from other sweep CSVs that didn't tune uniform_target).
                ug = ug[ug["uniform_target"].notna()]

            if uniform_cap_metric in ug.columns and not pd.isna(cap):
                feas = ug[pd.to_numeric(ug[uniform_cap_metric], errors="coerce") <= (float(cap) + uniform_cap_eps)]
                candidates = feas.dropna(subset=["control_p99"])
            if candidates.empty:
                # No feasible uniform point -> gap in curve
                continue
            if uniform_pick == "max_uniform_target" and "uniform_target" in candidates.columns:
                ut = candidates["uniform_target"]
                if ut.notna().any():
                    rows.append(candidates.loc[ut.idxmax()].to_dict())
                    kept_uniform += 1
                    continue
            if uniform_pick == "max_throughput" and "throughput_flits_class0" in candidates.columns:
                t = candidates["throughput_flits_class0"]
                if t.notna().any():
                    rows.append(candidates.loc[t.idxmax()].to_dict())
                    kept_uniform += 1
                    continue

        # Fallback: min p99
        rows.append(candidates.loc[candidates["control_p99"].idxmin()].to_dict())
        if pol == "uniform":
            kept_uniform += 1

    out = pd.DataFrame(rows)
    if saw_uniform and kept_uniform == 0:
        print(
            "WARNING: uniform policy present in CSVs but no cap-feasible uniform points were found. "
            "Did you include a uniform_target sweep CSV and sweep low enough targets for these caps?"
        )
    return out


def _style(pol: str):
    if pol == "hw_reactive":
        return dict(color="#1f77b4", linestyle="-", marker="o", label="HW reactive")
    if pol == "queue_pid":
        return dict(color="#ff7f0e", linestyle="-", marker="o", label="Queue PID")
    if pol == "perf_target":
        return dict(color="#2ca02c", linestyle="-", marker="o", label="Perf target")
    if pol == "uniform":
        return dict(color="#111111", linestyle=":", marker="o", label="Uniform (best feasible)")
    return dict(linestyle="-", marker="o", label=pol)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", action="append", default=[], help="Input sweep_results CSV (repeatable).")
    ap.add_argument("--out", default="control_p99_vs_power_cap.png", help="Output image path.")
    ap.add_argument("--title", default=None, help="Optional plot title.")
    ap.add_argument("--dvfs-epoch", type=int, default=None, help="Filter to a single dvfs_epoch.")
    ap.add_argument("--injection-rate", type=float, default=None, help="Select a single injection_rate.")
    ap.add_argument("--slo-p99", type=float, default=None, help="Draw a horizontal SLO line at this p99.")
    ap.add_argument("--include-failures", action="store_true", help="Include non-zero exit_code runs.")
    ap.add_argument("--uniform-cap-metric", default="total_power_peak", help="Cap compliance metric for uniform.")
    ap.add_argument("--uniform-cap-eps", type=float, default=1e-9, help="Slack for cap comparison.")
    ap.add_argument(
        "--uniform-csv",
        default=None,
        help="If set, use uniform points only from this CSV path (recommended: sweep_results_uniform_tuning.csv).",
    )
    ap.add_argument(
        "--uniform-pick",
        choices=["min_p99", "max_uniform_target", "max_throughput"],
        default="max_uniform_target",
        help="How to choose the best feasible uniform point per cap.",
    )
    args = ap.parse_args()

    csvs = args.csv or ["sweep_results_all_policies.csv", "sweep_results_uniform_tuning.csv"]
    df = _load_csvs(csvs)

    if not args.include_failures and "exit_code" in df.columns:
        df = df[df["exit_code"].fillna(1) == 0]

    if args.dvfs_epoch is not None and "dvfs_epoch" in df.columns:
        df = df[df["dvfs_epoch"] == args.dvfs_epoch]

    df = _require_single_injection(df, args.injection_rate)

    df = df.dropna(subset=["power_cap", "control_p99"])
    if df.empty:
        raise SystemExit("No rows to plot after filtering (check dvfs_epoch/injection_rate and exit_code).")

    picked = _pick_per_policy_cap(
        df,
        uniform_cap_metric=args.uniform_cap_metric,
        uniform_cap_eps=args.uniform_cap_eps,
        uniform_pick=args.uniform_pick,
        uniform_source_csv=args.uniform_csv,
    )
    if picked.empty:
        raise SystemExit("No rows after selection. Did you include the uniform_target sweep CSV?")

    # Plot
    plt.figure(figsize=(7.2, 4.6), dpi=140)
    for pol in POLICY_ORDER:
        s = picked[picked["policy_norm"] == pol]
        if s.empty:
            continue
        s = s.sort_values("power_cap")
        plt.plot(s["power_cap"], s["control_p99"], **_style(pol))

    if args.slo_p99 is not None:
        plt.axhline(args.slo_p99, color="black", linewidth=1.6, alpha=0.7, label=f"SLO={args.slo_p99:g}")

    plt.xlabel("Power cap")
    plt.ylabel("Control P99 latency (cycles)")
    if args.title:
        plt.title(args.title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.close()
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
