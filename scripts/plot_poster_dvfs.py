#!/usr/bin/env python3
"""
Poster-quality plots for DVFS policy comparisons.

Generates (Tier 1–3) the exact plots you described from sweep_results*.csv:
  1) Control P99 vs Power Cap (lines, SLO line, dashed uniform)
  2) Control P99 vs Avg Power (scatter, marker=inj_rate, Pareto highlight)
  3) Tail latency fairness ratio (batch/control) vs Power Cap (lines, band)
  4) Throughput vs Power Cap (2 subplots: class0 and class1)

This script expects sweeps produced by scripts/sweep_params.py, i.e. rows with:
  policy, power_cap, injection_rate (if applicable), output_dir, run_name,
  control_p99, batch_p99, throughput_flits_class0/1, exit_code, ...

Note: When use_netrace=1, "injection_rate" usually does not control offered load.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_POLICY_ORDER = ["hw_reactive", "queue_pid", "perf_target", "uniform"]

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


def _as_float(x) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _run_dir(row: pd.Series) -> Optional[Path]:
    out = row.get("output_dir")
    name = row.get("run_name")
    if not isinstance(out, str) or not out:
        return None
    if not isinstance(name, str) or not name:
        return None
    return Path(out) / name


def _read_epoch_csv_metrics(run_dir: Path, control_class: int, batch_class: int) -> Dict[str, Optional[float]]:
    """
    Compute "active" averages from epoch.csv:
      - active_power_avg: mean(total_power) for epochs with total_power>0 and control throughput>0
      - active_control_throughput: mean(class{control}_throughput) on active epochs
      - active_batch_throughput: mean(class{batch}_throughput) on active epochs
    Returns None values if epoch.csv isn't found or no active epochs.
    """
    epoch_path = run_dir / "epoch.csv"
    if not epoch_path.exists():
        # sweep_params.py always writes epoch.csv when configured, but be robust
        # (some runs may not enable epoch_csv).
        return {
            "active_power_avg": None,
            "active_control_throughput": None,
            "active_batch_throughput": None,
        }

    try:
        df = pd.read_csv(epoch_path)
    except Exception:
        return {
            "active_power_avg": None,
            "active_control_throughput": None,
            "active_batch_throughput": None,
        }

    def col_series(name: str) -> pd.Series:
        if name not in df.columns:
            return pd.Series([float("nan")] * len(df))
        return pd.to_numeric(df[name], errors="coerce")

    tp = col_series("total_power")
    ctrl_thr = col_series(f"class{control_class}_throughput")
    batch_thr = col_series(f"class{batch_class}_throughput")

    active = (tp > 0.0) & (ctrl_thr > 0.0)
    if active.sum() == 0:
        return {
            "active_power_avg": None,
            "active_control_throughput": None,
            "active_batch_throughput": None,
        }

    return {
        "active_power_avg": float(tp[active].mean()),
        "active_control_throughput": float(ctrl_thr[active].mean()),
        "active_batch_throughput": float(batch_thr[active].mean()),
    }


def _pareto_mask(x: Iterable[float], y: Iterable[float]) -> List[bool]:
    """
    Pareto-optimal for minimizing both x and y.
    Returns a boolean mask of length n where True means "not dominated".
    """
    pts = [(float(a), float(b)) for a, b in zip(x, y)]
    keep = [True] * len(pts)
    for i, (xi, yi) in enumerate(pts):
        if not keep[i]:
            continue
        for j, (xj, yj) in enumerate(pts):
            if i == j:
                continue
            # j dominates i if <= in both and strictly < in at least one
            if (xj <= xi and yj <= yi) and (xj < xi or yj < yi):
                keep[i] = False
                break
    return keep


def _style_for_policy(policy: str) -> Tuple[str, str]:
    """
    Returns (linestyle, pretty_label).
    """
    p = (policy or "").strip().lower()
    label = p
    if p == "hw_reactive":
        label = "HW reactive"
    elif p == "queue_pid":
        label = "Queue PID"
    elif p == "perf_target":
        label = "Perf target"
    elif p == "uniform":
        label = "Uniform"
    linestyle = "--" if p == "uniform" else "-"
    return linestyle, label


def _apply_poster_rcparams() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 15,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "lines.linewidth": 2.5,
            "lines.markersize": 6,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
        }
    )


@dataclass
class PlotConfig:
    control_class: int
    batch_class: int
    slo_p99: Optional[float]
    dvfs_epoch: Optional[int]
    pick_mode: str
    require_success: bool
    uniform_cap_metric: str
    uniform_cap_eps: float


def _pick_best(df: pd.DataFrame, cfg: PlotConfig) -> pd.DataFrame:
    """
    Pick one row per (policy, injection_rate, power_cap) (and optionally dvfs_epoch)
    using cfg.pick_mode on control_p99.
    """
    work = df.copy()
    work["policy_norm"] = work["policy"].apply(_norm_policy)
    if cfg.require_success and "exit_code" in work.columns:
        work = work[pd.to_numeric(work["exit_code"], errors="coerce").fillna(1) == 0]
    if cfg.dvfs_epoch is not None and "dvfs_epoch" in work.columns:
        work = work[pd.to_numeric(work["dvfs_epoch"], errors="coerce") == cfg.dvfs_epoch]

    # Ensure numeric for grouping/sorting
    for c in [
        "power_cap",
        "injection_rate",
        "control_p99",
        "batch_p99",
        "total_power_avg",
        "total_power_peak",
        "throughput_flits_class0",
        "throughput_flits_class1",
    ]:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")

    group_cols = ["policy_norm", "power_cap"]
    if "injection_rate" in work.columns:
        group_cols.insert(1, "injection_rate")

    def pick(g: pd.DataFrame) -> pd.Series:
        if g.empty:
            return g.iloc[0]
        # Special case: uniform baseline should respect the cap. Expect the user to
        # sweep uniform_target; then pick the best feasible point per (cap,inj).
        if g["policy_norm"].iloc[0] == "uniform":
            metric = cfg.uniform_cap_metric
            if metric in g.columns and g["power_cap"].notna().any():
                cap = float(g["power_cap"].iloc[0])
                feas = g[pd.to_numeric(g[metric], errors="coerce") <= (cap + cfg.uniform_cap_eps)]
                if not feas.empty:
                    feas = feas.copy()
                    feas["uniform_feasible"] = 1
                    g = feas
                else:
                    # No feasible uniform point for this cap (given the swept uniform_target set).
                    # Return a row with NaNs for plotted metrics so the curve has a gap instead
                    # of silently picking an over-cap run.
                    row = g.iloc[0].copy()
                    row["uniform_feasible"] = 0
                    for k in ["control_p99", "control_p95", "control_p50", "batch_p99", "batch_p95", "batch_p50",
                              "throughput_flits_class0", "throughput_flits_class1", "total_power_avg", "total_power_peak"]:
                        if k in row.index:
                            row[k] = float("nan")
                    return row
        if "control_p99" in g.columns and g["control_p99"].notna().any():
            if cfg.pick_mode == "min_p99":
                return g.loc[g["control_p99"].idxmin()]
            if cfg.pick_mode == "max_throughput":
                t = g.get("throughput_flits_class0")
                if t is not None and t.notna().any():
                    return g.loc[t.idxmax()]
        return g.iloc[0]

    rows: List[Dict[str, object]] = []
    for _, g in work.groupby(group_cols, dropna=False):
        s = pick(g)
        if isinstance(s, pd.Series):
            rows.append(s.to_dict())
        else:
            rows.append(pd.Series(s).to_dict())
    return pd.DataFrame(rows)


def _augment_active_power(df: pd.DataFrame, cfg: PlotConfig) -> pd.DataFrame:
    """
    Adds active_power_avg to each row (computed from epoch.csv).
    Falls back to total_power_avg if epoch.csv is unavailable.
    """
    active_vals: List[Optional[float]] = []
    for _, row in df.iterrows():
        run_dir = _run_dir(row)
        if run_dir is None:
            active_vals.append(_as_float(row.get("total_power_avg")))
            continue
        m = _read_epoch_csv_metrics(run_dir, cfg.control_class, cfg.batch_class)
        ap = m["active_power_avg"]
        if ap is None:
            ap = _as_float(row.get("total_power_avg"))
        active_vals.append(ap)
    out = df.copy()
    out["active_power_avg"] = pd.to_numeric(active_vals, errors="coerce")
    return out


def plot_control_p99_vs_cap(df: pd.DataFrame, outdir: Path, cfg: PlotConfig) -> List[Path]:
    outs: List[Path] = []
    inj_vals = sorted(df["injection_rate"].dropna().unique().tolist()) if "injection_rate" in df.columns else [None]

    n = len(inj_vals)
    cols = min(3, n) if n else 1
    rows = int(math.ceil(n / cols)) if n else 1
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 4.2 * rows), squeeze=False)

    for idx, inj in enumerate(inj_vals):
        ax = axes[idx // cols][idx % cols]
        sub = df if inj is None else df[df["injection_rate"] == inj]
        for pol in DEFAULT_POLICY_ORDER:
            s = sub[sub["policy_norm"] == pol]
            if s.empty:
                continue
            s = s.sort_values("power_cap")
            ls, label = _style_for_policy(pol)
            ax.plot(s["power_cap"], s["control_p99"], linestyle=ls, marker="o", label=label)
        if cfg.slo_p99 is not None:
            ax.axhline(cfg.slo_p99, color="black", linewidth=1.8, alpha=0.8, label=f"SLO={cfg.slo_p99:g}")
        ax.set_xlabel("Power cap")
        ax.set_ylabel("Control P99 latency (cycles)")
        title = f"injection_rate={inj:g}" if inj is not None else "all injection rates"
        ax.set_title(title)

    # Hide unused subplots
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)), frameon=False)
    fig.suptitle("Control P99 vs Power Cap", y=0.98)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))

    png = outdir / "tier1_control_p99_vs_power_cap.png"
    pdf = outdir / "tier1_control_p99_vs_power_cap.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    outs += [png, pdf]
    return outs


def plot_control_p99_vs_power(df: pd.DataFrame, outdir: Path, cfg: PlotConfig) -> List[Path]:
    outs: List[Path] = []
    work = df.dropna(subset=["control_p99", "active_power_avg"]).copy()
    if work.empty:
        return outs

    inj_vals = sorted(work["injection_rate"].dropna().unique().tolist()) if "injection_rate" in work.columns else []
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X"]
    inj_marker = {inj: marker_cycle[i % len(marker_cycle)] for i, inj in enumerate(inj_vals)}

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    policies = sorted(
        work["policy_norm"].dropna().unique().tolist(),
        key=lambda p: DEFAULT_POLICY_ORDER.index(p) if p in DEFAULT_POLICY_ORDER else 99,
    )

    for pol in policies:
        s = work[work["policy_norm"] == pol]
        _, label = _style_for_policy(str(pol))
        # scatter only (no lines)
        for inj, sub in s.groupby("injection_rate") if "injection_rate" in s.columns else [(None, s)]:
            mk = inj_marker.get(inj, "o")
            ax.scatter(
                sub["active_power_avg"],
                sub["control_p99"],
                label=f"{label} (inj={inj:g})" if inj is not None else label,
                marker=mk,
                alpha=0.85,
            )

    # Pareto highlight (global)
    pareto = work.copy()
    pareto = pareto.dropna(subset=["active_power_avg", "control_p99"])
    if not pareto.empty:
        mask = _pareto_mask(pareto["active_power_avg"].tolist(), pareto["control_p99"].tolist())
        pf = pareto[mask]
        ax.scatter(
            pf["active_power_avg"],
            pf["control_p99"],
            facecolors="none",
            edgecolors="black",
            linewidths=1.8,
            s=110,
            label="Pareto-optimal (global)",
        )

    if cfg.slo_p99 is not None:
        ax.axhline(cfg.slo_p99, color="black", linewidth=1.6, alpha=0.7)

    ax.set_xlabel("Avg power during active epochs")
    ax.set_ylabel("Control P99 latency (cycles)")
    ax.set_title("Control P99 vs Avg Power (Efficiency / Pareto)")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()

    png = outdir / "tier1b_control_p99_vs_avg_power.png"
    pdf = outdir / "tier1b_control_p99_vs_avg_power.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    outs += [png, pdf]
    return outs


def plot_tail_ratio_vs_cap(df: pd.DataFrame, outdir: Path, cfg: PlotConfig) -> List[Path]:
    outs: List[Path] = []
    work = df.copy()
    work["fairness_ratio"] = pd.to_numeric(work.get("tail_latency_ratio_batch_over_control"), errors="coerce")
    if "batch_p99" in work.columns and "control_p99" in work.columns:
        bp = pd.to_numeric(work["batch_p99"], errors="coerce")
        cp = pd.to_numeric(work["control_p99"], errors="coerce")
        computed = bp / cp
        # Prefer computed where valid (handles cases where the precomputed ratio is 0/blank).
        work["fairness_ratio"] = work["fairness_ratio"].where(work["fairness_ratio"].notna(), computed)

    work = work.dropna(subset=["fairness_ratio", "power_cap"])
    if work.empty:
        return outs

    inj_vals = sorted(work["injection_rate"].dropna().unique().tolist()) if "injection_rate" in work.columns else [None]
    n = len(inj_vals)
    cols = min(3, n) if n else 1
    rows = int(math.ceil(n / cols)) if n else 1
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 4.2 * rows), squeeze=False)

    for idx, inj in enumerate(inj_vals):
        ax = axes[idx // cols][idx % cols]
        ax.axhspan(1.0, 3.0, color="gray", alpha=0.15, label="acceptable (1–3×)")
        sub = work if inj is None else work[work["injection_rate"] == inj]
        for pol in DEFAULT_POLICY_ORDER:
            s = sub[sub["policy_norm"] == pol]
            if s.empty:
                continue
            s = s.sort_values("power_cap")
            ls, label = _style_for_policy(pol)
            ax.plot(s["power_cap"], s["fairness_ratio"], linestyle=ls, marker="o", label=label)
        ax.set_xlabel("Power cap")
        ax.set_ylabel("Tail latency ratio (batch / control)")
        title = f"injection_rate={inj:g}" if inj is not None else "all injection rates"
        ax.set_title(title)

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)), frameon=False)
    fig.suptitle("Tail Latency Fairness vs Power Cap", y=0.98)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))

    png = outdir / "tier2_tail_latency_ratio_vs_power_cap.png"
    pdf = outdir / "tier2_tail_latency_ratio_vs_power_cap.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    outs += [png, pdf]
    return outs


def plot_throughput_vs_cap(df: pd.DataFrame, outdir: Path, cfg: PlotConfig) -> List[Path]:
    outs: List[Path] = []
    if "throughput_flits_class0" not in df.columns and "throughput_flits_class1" not in df.columns:
        return outs

    work = df.copy()
    work["thr0"] = pd.to_numeric(work.get("throughput_flits_class0"), errors="coerce")
    work["thr1"] = pd.to_numeric(work.get("throughput_flits_class1"), errors="coerce")

    inj_vals = sorted(work["injection_rate"].dropna().unique().tolist()) if "injection_rate" in work.columns else [None]
    nrows = len(inj_vals) if inj_vals != [None] else 1
    fig, axes = plt.subplots(nrows, 2, figsize=(12.0, 3.8 * nrows), squeeze=False, sharex=True)

    for r, inj in enumerate(inj_vals if inj_vals else [None]):
        sub = work if inj is None else work[work["injection_rate"] == inj]
        for c, (thr_col, title) in enumerate([("thr0", "Class 0 throughput"), ("thr1", "Class 1 throughput")]):
            ax = axes[r][c]
            for pol in DEFAULT_POLICY_ORDER:
                s = sub[sub["policy_norm"] == pol]
                if s.empty:
                    continue
                s = s.sort_values("power_cap")
                ls, label = _style_for_policy(pol)
                ax.plot(s["power_cap"], s[thr_col], linestyle=ls, marker="o", label=label)
            ax.set_title(f"{title}" + (f" (inj={inj:g})" if inj is not None else ""))
            ax.set_xlabel("Power cap")
            ax.set_ylabel("Throughput (flits/cycle)")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)), frameon=False)
    fig.suptitle("Throughput vs Power Cap", y=0.98)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))

    png = outdir / "tier3_throughput_vs_power_cap.png"
    pdf = outdir / "tier3_throughput_vs_power_cap.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    outs += [png, pdf]
    return outs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", action="append", default=[], help="Input sweep_results CSV (repeatable).")
    ap.add_argument("--outdir", default="poster_plots", help="Output directory for figures.")
    ap.add_argument("--control-class", type=int, default=0, help="Control class id (default: 0).")
    ap.add_argument("--batch-class", type=int, default=1, help="Batch class id (default: 1).")
    ap.add_argument("--slo-p99", type=float, default=None, help="Draw horizontal SLO line at this P99 latency.")
    ap.add_argument("--dvfs-epoch", type=int, default=None, help="Filter to a single dvfs_epoch (recommended).")
    ap.add_argument(
        "--pick",
        choices=["min_p99", "max_throughput"],
        default="min_p99",
        help="When multiple runs share (policy,injection_rate,power_cap), pick which one to plot.",
    )
    ap.add_argument(
        "--uniform-cap-metric",
        default="total_power_peak",
        help="Metric used to enforce cap for uniform baseline selection (default: total_power_peak).",
    )
    ap.add_argument(
        "--uniform-cap-eps",
        type=float,
        default=1e-9,
        help="Epsilon slack when checking uniform cap compliance (default: 1e-9).",
    )
    ap.add_argument(
        "--include-failures",
        action="store_true",
        help="Include runs with non-zero exit_code (not recommended).",
    )
    args = ap.parse_args()

    csvs = args.csv or ["sweep_results_all_policies.csv", "sweep_results.csv"]
    frames: List[pd.DataFrame] = []
    for p in csvs:
        path = Path(p)
        if not path.exists():
            continue
        frames.append(pd.read_csv(path))
    if not frames:
        print("No CSVs found. Provide --csv sweep_results_all_policies.csv", flush=True)
        return 2

    df = pd.concat(frames, ignore_index=True)
    if "policy" not in df.columns:
        print("Expected a 'policy' column in the sweep CSV(s).", flush=True)
        return 2
    if "power_cap" not in df.columns:
        print("Expected a 'power_cap' column in the sweep CSV(s).", flush=True)
        return 2

    cfg = PlotConfig(
        control_class=args.control_class,
        batch_class=args.batch_class,
        slo_p99=args.slo_p99,
        dvfs_epoch=args.dvfs_epoch,
        pick_mode=args.pick,
        require_success=not args.include_failures,
        uniform_cap_metric=args.uniform_cap_metric,
        uniform_cap_eps=args.uniform_cap_eps,
    )

    _apply_poster_rcparams()
    outdir = Path(args.outdir)
    _safe_mkdir(outdir)

    picked = _pick_best(df, cfg)
    picked = _augment_active_power(picked, cfg)

    if "policy_norm" in picked.columns and (picked["policy_norm"] == "uniform").any():
        if "uniform_feasible" in picked.columns:
            infeas = picked[(picked["policy_norm"] == "uniform") & (picked["uniform_feasible"] == 0)]
            if not infeas.empty:
                caps = sorted(pd.to_numeric(infeas["power_cap"], errors="coerce").dropna().unique().tolist())
                print(
                    "WARNING: No cap-respecting uniform point found for some caps. "
                    "Uniform curve will have gaps there. "
                    f"Caps: {caps}"
                )

    outs: List[Path] = []
    outs += plot_control_p99_vs_cap(picked, outdir, cfg)
    outs += plot_control_p99_vs_power(picked, outdir, cfg)
    outs += plot_tail_ratio_vs_cap(picked, outdir, cfg)
    outs += plot_throughput_vs_cap(picked, outdir, cfg)

    print("Wrote:")
    for o in outs:
        print(f"  {o}")

    # Pitfall warnings
    if "use_netrace" in df.columns and pd.to_numeric(df["use_netrace"], errors="coerce").fillna(0).max() > 0:
        if "injection_rate" in df.columns:
            print(
                "NOTE: use_netrace=1 detected; injection_rate may not control offered load. "
                "Facet labels reflect the sweep parameter, not necessarily the trace intensity."
            )
    if "batch_p99" in df.columns:
        bp = pd.to_numeric(df["batch_p99"], errors="coerce")
        if bp.notna().any() and (bp.fillna(0) == 0).any():
            print(
                "NOTE: Some runs have batch_p99=0 (often means classes=1 or no batch traffic). "
                "Fairness ratio plots will drop those points."
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
