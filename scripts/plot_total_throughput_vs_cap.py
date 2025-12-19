#!/usr/bin/env python3
"""
Clean plot: Total Throughput vs Power Cap (single subplot).

One line per policy:
  - hw_reactive, queue_pid, perf_target: solid
  - uniform: dotted, using the best *cap-feasible* uniform point per cap

Total throughput = throughput_flits_class0 + throughput_flits_class1
(as written by scripts/sweep_params.py from throughput.csv).

Selection:
  - For DVFS policies: pick one run per (policy, cap) (default: max total throughput).
  - For uniform: filter to cap-feasible (total_power_peak <= power_cap) and then
    pick best by --uniform-pick (default: max_uniform_target).

Example:
  python3 scripts/plot_total_throughput_vs_cap.py \
    --csv sweep_results.csv --csv sweep_results_uniform_tuning.csv \
    --uniform-csv sweep_results_uniform_tuning.csv \
    --dvfs-epoch 10000 --injection-rate 0.5 \
    --out plots/total_throughput_vs_cap.png
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
        raise SystemExit("No CSVs found. Provide --csv sweep_results.csv ...")
    df = pd.concat(frames, ignore_index=True)
    df["policy_norm"] = df.get("policy", pd.Series(["unknown"] * len(df))).apply(_norm_policy)
    for c in [
        "power_cap",
        "dvfs_epoch",
        "injection_rate",
        "exit_code",
        "throughput_flits_class0",
        "throughput_flits_class1",
        "total_power_peak",
        "uniform_target",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Total throughput; tolerate missing class1 (treat as 0) so mixed CSVs still plot.
    t0 = df["throughput_flits_class0"] if "throughput_flits_class0" in df.columns else pd.Series([pd.NA] * len(df))
    t1 = df["throughput_flits_class1"] if "throughput_flits_class1" in df.columns else pd.Series([0.0] * len(df))
    df["throughput_total"] = pd.to_numeric(t0, errors="coerce").fillna(0.0) + pd.to_numeric(t1, errors="coerce").fillna(0.0)
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

    feas = feas.dropna(subset=["throughput_total"])
    if feas.empty:
        return None

    if uniform_pick == "max_uniform_target" and "uniform_target" in feas.columns:
        ut = feas["uniform_target"]
        if ut.notna().any():
            return feas.loc[ut.idxmax()]
    if uniform_pick == "max_throughput":
        t = feas["throughput_total"]
        if t.notna().any():
            return feas.loc[t.idxmax()]
    return feas.loc[feas["throughput_total"].idxmax()]


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
            rows.append(s.to_dict())
            kept_uniform += 1
            continue

        cand = g.dropna(subset=["throughput_total"])
        if cand.empty:
            continue
        if pick_mode == "max_control" and "throughput_flits_class0" in cand.columns:
            t0 = cand["throughput_flits_class0"].fillna(0.0)
            rows.append(cand.loc[t0.idxmax()].to_dict())
        elif pick_mode == "max_batch" and "throughput_flits_class1" in cand.columns:
            t1 = cand["throughput_flits_class1"].fillna(0.0)
            rows.append(cand.loc[t1.idxmax()].to_dict())
        else:
            rows.append(cand.loc[cand["throughput_total"].idxmax()].to_dict())

    out = pd.DataFrame(rows)
    if saw_uniform and kept_uniform == 0:
        print(
            "NOTE: uniform policy present in CSVs, but no cap-feasible uniform points were found; "
            "plotting only the DVFS policies. (Resweep uniform_target lower and/or pass --uniform-csv.)"
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
    ap.add_argument("--out", default="total_throughput_vs_cap.png", help="Output image path.")
    ap.add_argument("--title", default=None, help="Optional plot title.")
    ap.add_argument("--dvfs-epoch", type=int, default=None, help="Filter to a single dvfs_epoch.")
    ap.add_argument("--injection-rate", type=float, default=None, help="Select a single injection_rate.")
    ap.add_argument("--include-failures", action="store_true", help="Include non-zero exit_code runs.")
    ap.add_argument("--uniform-cap-metric", default="total_power_peak", help="Cap compliance metric for uniform.")
    ap.add_argument("--uniform-cap-eps", type=float, default=1e-9, help="Slack for cap comparison.")
    ap.add_argument(
        "--uniform-pick",
        choices=["max_uniform_target", "max_throughput"],
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
        choices=["max_total", "max_control", "max_batch"],
        default="max_total",
        help="How to pick runs for DVFS policies when multiple exist per cap.",
    )
    args = ap.parse_args()

    csvs = args.csv or ["sweep_results.csv", "sweep_results_uniform_tuning.csv"]
    df = _load_csvs(csvs)

    if not args.include_failures and "exit_code" in df.columns:
        df = df[df["exit_code"].fillna(1) == 0]

    if args.dvfs_epoch is not None and "dvfs_epoch" in df.columns:
        df = df[df["dvfs_epoch"] == args.dvfs_epoch]

    df = _require_single_injection(df, args.injection_rate)
    df = df.dropna(subset=["power_cap"])
    df = df[df["throughput_total"].notna()]
    if df.empty:
        raise SystemExit("No rows to plot after filtering (need throughput_flits_class0/1).")

    pts = _select_points(
        df,
        uniform_cap_metric=args.uniform_cap_metric,
        uniform_cap_eps=args.uniform_cap_eps,
        uniform_pick=args.uniform_pick,
        uniform_source_csv=args.uniform_csv,
        pick_mode=args.pick,
    )
    if pts.empty:
        raise SystemExit("No points selected (did you include the uniform sweep CSV?).")

    present = sorted(pts["policy_norm"].dropna().unique().tolist()) if "policy_norm" in pts.columns else []
    if present:
        print("Policies plotted:", ", ".join(present))
    if present == ["uniform"]:
        print(
            "NOTE: only uniform was available after filtering. "
            "This usually means your non-uniform sweep CSV doesn't match --dvfs-epoch/--injection-rate "
            "(or is missing throughput columns)."
        )

    plt.figure(figsize=(7.2, 4.6), dpi=140)
    for pol in POLICY_ORDER:
        s = pts[pts["policy_norm"] == pol]
        if s.empty:
            continue
        s = s.sort_values("power_cap")
        plt.plot(s["power_cap"], s["throughput_total"], **_style(pol))

    plt.xlabel("Power cap")
    plt.ylabel("Total throughput (flits/cycle, class0 + class1)")
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
