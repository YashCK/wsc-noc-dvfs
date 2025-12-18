#!/usr/bin/env python3
"""
Clean plot: Control P99 vs Avg Power (Efficiency / Pareto) for all controllers.

Features:
  - Scatter only (no lines)
  - One color per policy (hw_reactive, queue_pid, perf_target, uniform)
  - Uniform points are selected as "best feasible" under the cap:
      total_power_peak <= power_cap (configurable via --uniform-cap-metric)
    and then picked by --uniform-pick (default: max_uniform_target).
  - Optional Pareto highlighting (down-left is better).

Example:
  python3 scripts/plot_control_p99_vs_avg_power.py \
    --csv sweep_results_all_policies.csv --csv sweep_results_uniform_fast.csv \
    --uniform-csv sweep_results_uniform_fast.csv \
    --dvfs-epoch 10000 --injection-rate 0.5 \
    --slo-p99 232 --pareto \
    --out plots/control_p99_vs_avg_power.png
"""

from __future__ import annotations

import argparse
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
        "total_power_avg",
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


def _pick_uniform_best_feasible(
    g: pd.DataFrame,
    *,
    uniform_cap_metric: str,
    uniform_cap_eps: float,
    uniform_pick: str,
    uniform_source_csv: Optional[str],
) -> Optional[pd.Series]:
    cap = g["power_cap"].iloc[0]
    ug = g
    if uniform_source_csv is not None and "source_csv" in ug.columns:
        ug = ug[ug["source_csv"] == uniform_source_csv]
    elif "uniform_target" in ug.columns and ug["uniform_target"].notna().any():
        ug = ug[ug["uniform_target"].notna()]

    if uniform_cap_metric in ug.columns and not pd.isna(cap):
        feas = ug[pd.to_numeric(ug[uniform_cap_metric], errors="coerce") <= (float(cap) + uniform_cap_eps)]
    else:
        feas = ug
    feas = feas.dropna(subset=["control_p99", "total_power_avg"])
    if feas.empty:
        return None

    if uniform_pick == "max_uniform_target" and "uniform_target" in feas.columns:
        ut = feas["uniform_target"]
        if ut.notna().any():
            return feas.loc[ut.idxmax()]
    if uniform_pick == "max_throughput" and "throughput_flits_class0" in feas.columns:
        t = feas["throughput_flits_class0"]
        if t.notna().any():
            return feas.loc[t.idxmax()]
    return feas.loc[feas["control_p99"].idxmin()]


def _select_points(
    df: pd.DataFrame,
    *,
    uniform_cap_metric: str,
    uniform_cap_eps: float,
    uniform_pick: str,
    uniform_source_csv: Optional[str],
) -> pd.DataFrame:
    rows = []
    saw_uniform = False
    kept_uniform = 0

    for (pol, cap), g in df.groupby(["policy_norm", "power_cap"], dropna=False):
        if g.empty:
            continue
        pol = str(pol)
        if pol == "uniform":
            saw_uniform = True
            s = _pick_uniform_best_feasible(
                g,
                uniform_cap_metric=uniform_cap_metric,
                uniform_cap_eps=uniform_cap_eps,
                uniform_pick=uniform_pick,
                uniform_source_csv=uniform_source_csv,
            )
            if s is None:
                continue
            rows.append(s.to_dict())
            kept_uniform += 1
        else:
            cand = g.dropna(subset=["control_p99", "total_power_avg"])
            if cand.empty:
                continue
            rows.append(cand.loc[cand["control_p99"].idxmin()].to_dict())

    out = pd.DataFrame(rows)
    if saw_uniform and kept_uniform == 0:
        print(
            "WARNING: uniform policy present in CSVs but no cap-feasible uniform points were found. "
            "Include a uniform_target sweep CSV and sweep low enough targets."
        )
    return out


def _pareto_mask(x: List[float], y: List[float]) -> List[bool]:
    keep = [True] * len(x)
    for i in range(len(x)):
        if not keep[i]:
            continue
        for j in range(len(x)):
            if i == j:
                continue
            if (x[j] <= x[i] and y[j] <= y[i]) and (x[j] < x[i] or y[j] < y[i]):
                keep[i] = False
                break
    return keep


def _style(pol: str):
    if pol == "hw_reactive":
        return dict(color="#1f77b4", label="HW reactive")
    if pol == "queue_pid":
        return dict(color="#ff7f0e", label="Queue PID")
    if pol == "perf_target":
        return dict(color="#2ca02c", label="Perf target")
    if pol == "uniform":
        return dict(color="#111111", label="Uniform (best feasible)")
    return dict(label=pol)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", action="append", default=[], help="Input sweep_results CSV (repeatable).")
    ap.add_argument("--out", default="control_p99_vs_avg_power.png", help="Output image path.")
    ap.add_argument("--title", default=None, help="Optional plot title.")
    ap.add_argument("--dvfs-epoch", type=int, default=None, help="Filter to a single dvfs_epoch.")
    ap.add_argument("--injection-rate", type=float, default=None, help="Select a single injection_rate.")
    ap.add_argument("--slo-p99", type=float, default=None, help="Draw a horizontal SLO line at this p99.")
    ap.add_argument("--include-failures", action="store_true", help="Include non-zero exit_code runs.")
    ap.add_argument("--pareto", action="store_true", help="Highlight global Pareto-optimal points (hollow circles).")
    ap.add_argument("--uniform-cap-metric", default="total_power_peak", help="Cap compliance metric for uniform.")
    ap.add_argument("--uniform-cap-eps", type=float, default=1e-9, help="Slack for cap comparison.")
    ap.add_argument(
        "--uniform-pick",
        choices=["min_p99", "max_uniform_target", "max_throughput"],
        default="max_uniform_target",
        help="How to choose the best feasible uniform point per cap.",
    )
    ap.add_argument(
        "--uniform-csv",
        default=None,
        help="If set, use uniform points only from this CSV path (recommended).",
    )
    args = ap.parse_args()

    csvs = args.csv or ["sweep_results_all_policies.csv", "sweep_results_uniform_fast.csv"]
    df = _load_csvs(csvs)

    if not args.include_failures and "exit_code" in df.columns:
        df = df[df["exit_code"].fillna(1) == 0]

    if args.dvfs_epoch is not None and "dvfs_epoch" in df.columns:
        df = df[df["dvfs_epoch"] == args.dvfs_epoch]

    df = _require_single_injection(df, args.injection_rate)
    df = df.dropna(subset=["power_cap", "control_p99", "total_power_avg"])
    if df.empty:
        raise SystemExit("No rows to plot after filtering (need total_power_avg and control_p99).")

    pts = _select_points(
        df,
        uniform_cap_metric=args.uniform_cap_metric,
        uniform_cap_eps=args.uniform_cap_eps,
        uniform_pick=args.uniform_pick,
        uniform_source_csv=args.uniform_csv,
    )
    if pts.empty:
        raise SystemExit("No points selected (did you include the uniform sweep CSV?).")

    # Plot
    plt.figure(figsize=(7.2, 4.8), dpi=140)
    for pol in POLICY_ORDER:
        s = pts[pts["policy_norm"] == pol]
        if s.empty:
            continue
        st = _style(pol)
        marker = "x" if pol == "uniform" else "o"
        plt.scatter(s["total_power_avg"], s["control_p99"], marker=marker, s=70, alpha=0.9, **st)

    if args.pareto:
        pareto = pts.dropna(subset=["total_power_avg", "control_p99"]).copy()
        x = pareto["total_power_avg"].tolist()
        y = pareto["control_p99"].tolist()
        mask = _pareto_mask(x, y)
        pf = pareto[mask]
        plt.scatter(
            pf["total_power_avg"],
            pf["control_p99"],
            facecolors="none",
            edgecolors="black",
            linewidths=1.6,
            s=120,
            label="Pareto-optimal",
        )

    if args.slo_p99 is not None:
        plt.axhline(args.slo_p99, color="black", linewidth=1.6, alpha=0.7, label=f"SLO={args.slo_p99:g}")

    plt.xlabel("Average power (total_power_avg)")
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

