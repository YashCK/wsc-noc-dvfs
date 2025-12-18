#!/usr/bin/env python3
"""
Clean plot: Tail Latency Fairness (Batch / Control) vs Power Cap.

One line per policy:
  - hw_reactive, queue_pid, perf_target: solid
  - uniform: dotted, using the best *cap-feasible* uniform point per cap

Fairness metric:
  ratio = batch_p99 / control_p99

Selection:
  - For DVFS policies: pick one run per (policy, cap) (default: min control_p99).
  - For uniform: filter to cap-feasible (total_power_peak <= power_cap) and then
    pick best by --uniform-pick (default: max_uniform_target).

Example:
  python3 scripts/plot_tail_fairness_vs_cap.py \
    --csv sweep_results_all_policies.csv --csv sweep_results_uniform_fast.csv \
    --uniform-csv sweep_results_uniform_fast.csv \
    --dvfs-epoch 10000 --injection-rate 0.5 \
    --out plots/tail_fairness_vs_cap.png
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
        "batch_p99",
        "tail_latency_ratio_batch_over_control",
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


def _compute_ratio(df: pd.DataFrame) -> pd.Series:
    # Prefer the precomputed ratio if present; fallback to batch_p99/control_p99.
    if "tail_latency_ratio_batch_over_control" in df.columns:
        r = pd.to_numeric(df["tail_latency_ratio_batch_over_control"], errors="coerce")
    else:
        r = pd.Series([float("nan")] * len(df), index=df.index)
    bp = pd.to_numeric(df.get("batch_p99"), errors="coerce")
    cp = pd.to_numeric(df.get("control_p99"), errors="coerce")
    computed = bp / cp
    r = r.where(r.notna(), computed)
    return r


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

    feas = feas.copy()
    feas["ratio"] = _compute_ratio(feas)
    feas = feas.dropna(subset=["ratio", "control_p99", "batch_p99"])
    # avoid division artifacts when batch traffic is absent
    feas = feas[(feas["control_p99"] > 0) & (feas["batch_p99"] > 0)]
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
    # fallback: minimize control_p99 (stays consistent with latency-first comparisons)
    return feas.loc[feas["control_p99"].idxmin()]


def _select_points(
    df: pd.DataFrame,
    *,
    uniform_cap_metric: str,
    uniform_cap_eps: float,
    uniform_pick: str,
    uniform_source_csv: Optional[str],
    pick_mode: str,
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
            d = s.to_dict()
            d["ratio"] = _as_ratio(d)
            rows.append(d)
            kept_uniform += 1
            continue

        cand = g.copy()
        cand["ratio"] = _compute_ratio(cand)
        cand = cand.dropna(subset=["ratio", "control_p99", "batch_p99"])
        cand = cand[(cand["control_p99"] > 0) & (cand["batch_p99"] > 0)]
        if cand.empty:
            continue
        if pick_mode == "min_ratio":
            rows.append(cand.loc[cand["ratio"].idxmin()].to_dict())
        else:
            rows.append(cand.loc[cand["control_p99"].idxmin()].to_dict())

    out = pd.DataFrame(rows)
    if saw_uniform and kept_uniform == 0:
        print(
            "WARNING: uniform policy present in CSVs but no cap-feasible uniform points were found. "
            "Include a uniform_target sweep CSV and sweep low enough targets."
        )
    return out


def _as_ratio(d: dict) -> float:
    r = d.get("tail_latency_ratio_batch_over_control")
    try:
        if r is not None and not (isinstance(r, float) and pd.isna(r)):
            return float(r)
    except Exception:
        pass
    try:
        bp = float(d.get("batch_p99"))
        cp = float(d.get("control_p99"))
        return bp / cp
    except Exception:
        return float("nan")


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
    ap.add_argument("--out", default="tail_fairness_vs_cap.png", help="Output image path.")
    ap.add_argument("--title", default=None, help="Optional plot title.")
    ap.add_argument("--dvfs-epoch", type=int, default=None, help="Filter to a single dvfs_epoch.")
    ap.add_argument("--injection-rate", type=float, default=None, help="Select a single injection_rate.")
    ap.add_argument("--include-failures", action="store_true", help="Include non-zero exit_code runs.")
    ap.add_argument("--uniform-cap-metric", default="total_power_peak", help="Cap compliance metric for uniform.")
    ap.add_argument("--uniform-cap-eps", type=float, default=1e-9, help="Slack for cap comparison.")
    ap.add_argument(
        "--uniform-pick",
        choices=["max_uniform_target", "max_throughput", "min_control_p99"],
        default="max_uniform_target",
        help="How to choose the best feasible uniform point per cap.",
    )
    ap.add_argument(
        "--uniform-csv",
        default=None,
        help="If set, use uniform points only from this CSV path (recommended).",
    )
    ap.add_argument(
        "--pick",
        choices=["min_control_p99", "min_ratio"],
        default="min_control_p99",
        help="How to pick runs for DVFS policies when multiple exist per cap.",
    )
    args = ap.parse_args()

    csvs = args.csv or ["sweep_results_all_policies.csv", "sweep_results_uniform_fast.csv"]
    df = _load_csvs(csvs)

    if not args.include_failures and "exit_code" in df.columns:
        df = df[df["exit_code"].fillna(1) == 0]

    if args.dvfs_epoch is not None and "dvfs_epoch" in df.columns:
        df = df[df["dvfs_epoch"] == args.dvfs_epoch]

    df = _require_single_injection(df, args.injection_rate)
    df = df.dropna(subset=["power_cap", "control_p99", "batch_p99"])
    if df.empty:
        raise SystemExit("No rows to plot after filtering (need batch_p99 and control_p99).")

    pts = _select_points(
        df,
        uniform_cap_metric=args.uniform_cap_metric,
        uniform_cap_eps=args.uniform_cap_eps,
        uniform_pick=args.uniform_pick,
        uniform_source_csv=args.uniform_csv,
        pick_mode=args.pick,
    )
    if pts.empty:
        raise SystemExit("No points selected (did you include the uniform sweep CSV and batch traffic?).")

    plt.figure(figsize=(7.2, 4.6), dpi=140)
    for pol in POLICY_ORDER:
        s = pts[pts["policy_norm"] == pol]
        if s.empty:
            continue
        s = s.sort_values("power_cap")
        style = _style(pol)
        plt.plot(s["power_cap"], s["ratio"], **style)

    plt.xlabel("Power cap")
    plt.ylabel("Tail latency ratio (batch P99 / control P99)")
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

