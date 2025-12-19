#!/usr/bin/env python3
"""
Clean plot: Peak Power vs Power Cap (non-uniform policies only).

Purpose: quick sanity check that each DVFS policy respects the cap.

Plots one line per policy (excluding uniform):
  - hw_reactive, queue_pid, perf_target (whatever exists in the CSV)

Uses:
  x = power_cap
  y = total_power_peak

Example:
  python3 scripts/plot_peak_power_vs_cap.py \
    --csv sweep_results.csv \
    --dvfs-epoch 10000 --injection-rate 0.5 \
    --out plots/peak_power_vs_cap.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd


POLICY_ORDER = ["hw_reactive", "queue_pid", "perf_target"]


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
        raise SystemExit("No CSVs found. Provide --csv sweep_results.csv ...")
    df = pd.concat(frames, ignore_index=True)
    df["policy_norm"] = df.get("policy", pd.Series(["unknown"] * len(df))).apply(_norm_policy)
    for c in ["power_cap", "dvfs_epoch", "injection_rate", "exit_code", "total_power_peak"]:
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


def _style(pol: str):
    if pol == "hw_reactive":
        return dict(color="#1f77b4", linestyle="-", marker="o", label="HW reactive")
    if pol == "queue_pid":
        return dict(color="#ff7f0e", linestyle="-", marker="o", label="Queue PID")
    if pol == "perf_target":
        return dict(color="#2ca02c", linestyle="-", marker="o", label="Perf target")
    return dict(linestyle="-", marker="o", label=pol)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", action="append", default=[], help="Input sweep_results CSV (repeatable).")
    ap.add_argument("--out", default="peak_power_vs_cap.png", help="Output image path.")
    ap.add_argument("--title", default=None, help="Optional plot title.")
    ap.add_argument("--dvfs-epoch", type=int, default=None, help="Filter to a single dvfs_epoch.")
    ap.add_argument("--injection-rate", type=float, default=None, help="Select a single injection_rate.")
    ap.add_argument("--include-failures", action="store_true", help="Include non-zero exit_code runs.")
    ap.add_argument(
        "--pick",
        choices=["max_peak", "min_peak", "mean_peak"],
        default="max_peak",
        help="How to collapse multiple runs per (policy, cap).",
    )
    args = ap.parse_args()

    csvs = args.csv or ["sweep_results.csv"]
    df = _load_csvs(csvs)

    if not args.include_failures and "exit_code" in df.columns:
        df = df[df["exit_code"].fillna(1) == 0]

    if args.dvfs_epoch is not None and "dvfs_epoch" in df.columns:
        df = df[df["dvfs_epoch"] == args.dvfs_epoch]

    df = _require_single_injection(df, args.injection_rate)
    df = df.dropna(subset=["power_cap", "total_power_peak"])
    df = df[df["policy_norm"] != "uniform"]

    if df.empty:
        raise SystemExit("No non-uniform rows to plot after filtering (need total_power_peak).")

    rows = []
    for (pol, cap), g in df.groupby(["policy_norm", "power_cap"], dropna=False):
        if g.empty:
            continue
        if args.pick == "min_peak":
            rows.append(g.loc[g["total_power_peak"].idxmin()].to_dict())
        elif args.pick == "mean_peak":
            r = g.iloc[0].to_dict()
            r["total_power_peak"] = float(g["total_power_peak"].mean())
            rows.append(r)
        else:
            rows.append(g.loc[g["total_power_peak"].idxmax()].to_dict())

    pts = pd.DataFrame(rows)
    if pts.empty:
        raise SystemExit("No points selected.")

    plt.figure(figsize=(7.2, 4.6), dpi=140)
    for pol in POLICY_ORDER:
        s = pts[pts["policy_norm"] == pol]
        if s.empty:
            continue
        s = s.sort_values("power_cap")
        plt.plot(s["power_cap"], s["total_power_peak"], **_style(pol))

    # Plot any other non-uniform policies that might appear.
    extras = [p for p in sorted(pts["policy_norm"].unique().tolist()) if p not in POLICY_ORDER and p != "uniform"]
    for pol in extras:
        s = pts[pts["policy_norm"] == pol].sort_values("power_cap")
        plt.plot(s["power_cap"], s["total_power_peak"], **_style(pol))

    plt.xlabel("Power cap")
    plt.ylabel("Peak power (total_power_peak)")
    if args.title:
        plt.title(args.title)
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    if out.suffix.lower() != ".pdf":
        plt.savefig(out.with_suffix(".pdf"))
    print("Wrote", out, "and", out.with_suffix(".pdf"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

