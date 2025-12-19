#!/usr/bin/env python3
"""
Sweep BookSim config parameters and collect results.

Example:
  python3 scripts/sweep_params.py \
    --config booksim2/runfiles/flatfly_dvfs_orion.cfg \
    --param injection_rate=0.4,0.5,0.6 \
    --param dvfs_epoch=5000,10000 \
    --metric average_packet_latency_0 \
    --booksim-bin booksim2/src/booksim

Notes:
- You can override any BookSim config key on the command line (param=value).
- Each run will get a unique run_name derived from the parameter combo.
- Results are written to sweep_results.csv (one row per run) plus stdout.
"""

import argparse
import csv
import itertools
import os
import re
import hashlib
import subprocess
import sys
from typing import Dict, List, Optional, Tuple


def parse_params(param_list: List[str]) -> Dict[str, List[str]]:
    params = {}
    for p in param_list:
        if "=" not in p:
            raise ValueError(f"Bad --param format (expected name=val1,val2): {p}")
        name, vals = p.split("=", 1)
        values = [v for v in vals.split(",") if v]
        if not values:
            raise ValueError(f"No values provided for param {name}")
        params[name] = values
    return params


def _sanitize_run_name(name: str, max_len: int = 180) -> str:
    if not name:
        return "run"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", name)
    safe = safe.replace(os.sep, "-").replace("/", "-")
    safe = re.sub(r"-{2,}", "-", safe).strip("-")
    if len(safe) <= max_len:
        return safe
    h = hashlib.sha1(safe.encode("utf-8")).hexdigest()[:10]
    keep = max(1, max_len - 11)
    return f"{safe[:keep]}-{h}"


def run_one(booksim_bin: str, config: str, overrides: Dict[str, str],
            run_name_parts: Optional[List[str]] = None) -> Tuple[int, str]:
    run_name = overrides.get("run_name")
    if not run_name:
        # Build run_name from sweeped params but ignore output_dir to avoid duplicating it in paths
        suffix_parts: List[str] = []
        if run_name_parts:
            suffix_parts.extend(run_name_parts)
        suffix_parts.extend([f"{k}-{v}" for k, v in overrides.items() if k not in {"output_dir"}])
        suffix = "_".join(suffix_parts)
        run_name = f"sweep_{suffix}" if suffix else "sweep_run"
    run_name = _sanitize_run_name(run_name)
    overrides["run_name"] = run_name
    # BookSim resolves netrace_file relative to the process CWD. For convenience, accept
    # either a path relative to the current working directory or (fallback) relative to
    # the config's directory.
    if "netrace_file" in overrides:
        nf = overrides["netrace_file"]
        if nf and not os.path.isabs(nf):
            nf_cwd = os.path.abspath(nf)
            nf_cfg = os.path.normpath(os.path.join(os.path.dirname(config), nf))
            if os.path.exists(nf_cwd):
                overrides["netrace_file"] = nf_cwd
            elif os.path.exists(nf_cfg):
                overrides["netrace_file"] = nf_cfg
            else:
                print(f"[ERROR] netrace_file not found: {nf} (tried {nf_cwd} and {nf_cfg})", file=sys.stderr)
                return 1, run_name
        elif nf and not os.path.exists(nf):
            print(f"[ERROR] netrace_file not found: {nf}", file=sys.stderr)
            return 1, run_name

    cmd = [booksim_bin, config] + [f"{k}={v}" for k, v in overrides.items()]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        print(f"[WARN] run {run_name} failed (exit {result.returncode}). Output:\n{result.stdout}")
    return result.returncode, run_name


def _to_float(val: Optional[str]) -> Optional[float]:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def read_summary(run_name: str, output_dir: str = "sims") -> Dict[str, str]:
    """Read summary plus additional per-run metrics."""
    run_dir = os.path.join(output_dir, run_name)
    summary = {}

    summary_path = os.path.join(run_dir, "summary.csv")
    if os.path.exists(summary_path):
        with open(summary_path, newline="") as f:
            reader = csv.DictReader(f)
            summary = next(reader, {})

    # Latency percentiles per class
    latency_path = os.path.join(run_dir, "latency.csv")
    if os.path.exists(latency_path):
        with open(latency_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cls = row.get("class")
                if cls == "0":
                    summary["control_p50"] = row.get("plat_p50", "")
                    summary["control_p95"] = row.get("plat_p95", "")
                    summary["control_p99"] = row.get("plat_p99", "")
                    summary["tail_latency_control"] = row.get("plat_p99", "")
                elif cls == "1":
                    summary["batch_p50"] = row.get("plat_p50", "")
                    summary["batch_p95"] = row.get("plat_p95", "")
                    summary["batch_p99"] = row.get("plat_p99", "")
                    summary["tail_latency_batch"] = row.get("plat_p99", "")

    # Time-varying power, queueing, and stalls
    epoch_path = os.path.join(run_dir, "epoch.csv")
    if os.path.exists(epoch_path):
        headroom_vals = []
        total_power_vals = []
        qdel0 = []
        qdel1 = []
        stall0 = []
        stall1 = []
        with open(epoch_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                hr = _to_float(row.get("headroom"))
                if hr is not None:
                    headroom_vals.append(hr)
                tp = _to_float(row.get("total_power"))
                if tp is not None:
                    total_power_vals.append(tp)
                c0_q = _to_float(row.get("class0_qdel_p99"))
                if c0_q is not None:
                    qdel0.append(c0_q)
                c1_q = _to_float(row.get("class1_qdel_p99"))
                if c1_q is not None:
                    qdel1.append(c1_q)
                c0_stall = _to_float(row.get("class0_stall_xbar_rate"))
                if c0_stall is not None:
                    stall0.append(c0_stall)
                c1_stall = _to_float(row.get("class1_stall_xbar_rate"))
                if c1_stall is not None:
                    stall1.append(c1_stall)
        if headroom_vals:
            summary["headroom_min"] = min(headroom_vals)
        if total_power_vals:
            summary["total_power_peak"] = max(total_power_vals)
        if qdel0:
            summary["control_qdel_p99"] = max(qdel0)
        if qdel1:
            summary["batch_qdel_p99"] = max(qdel1)
        if stall0:
            summary["stall_xbar_rate_class0"] = max(stall0)
        if stall1:
            summary["stall_xbar_rate_class1"] = max(stall1)

    # Throughput / offered load
    throughput_path = os.path.join(run_dir, "throughput.csv")
    if os.path.exists(throughput_path):
        with open(throughput_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cls = row.get("class")
                if cls == "0":
                    summary["offered_load_class0"] = row.get("offered_load", "")
                    summary["throughput_flits_class0"] = row.get("avg_throughput_flits_per_cycle", "")
                elif cls == "1":
                    summary["offered_load_class1"] = row.get("offered_load", "")
                    summary["throughput_flits_class1"] = row.get("avg_throughput_flits_per_cycle", "")

    # Tail latency blow-up (fairness)
    try:
        c_p99 = float(summary.get("control_p99", ""))
        b_p99 = float(summary.get("batch_p99", ""))
        if c_p99 > 0:
            summary["tail_latency_ratio_batch_over_control"] = b_p99 / c_p99
    except ValueError:
        pass

    return summary


def main():
    parser = argparse.ArgumentParser(description="Sweep BookSim parameters")
    parser.add_argument("--config", action="append", required=True,
                        help="Path to BookSim config (repeatable for config sweep)")
    parser.add_argument("--param", action="append", default=[],
                        help="Parameter sweep in the form name=val1,val2 (repeatable)")
    parser.add_argument("--metric", action="append", default=[],
                        help="Optional metric keys to extract from summary.csv")
    parser.add_argument("--booksim-bin", default="booksim2/src/booksim",
                        help="Path to booksim executable")
    parser.add_argument("--output", default="sweep_results.csv",
                        help="Output CSV for sweep summary")
    args = parser.parse_args()

    param_grid = parse_params(args.param)

    # Allow sweeping configs either via repeated --config or via --param config=cfg1,cfg2.
    config_list = list(args.config or [])
    if "config" in param_grid:
        config_list.extend(param_grid.pop("config"))
    if not config_list:
        print("No configs provided. Use --config path or --param config=...", file=sys.stderr)
        sys.exit(1)

    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys])) if keys else [()]

    rows = []
    had_failure = False
    for config in config_list:
        config_id = os.path.splitext(os.path.basename(config))[0] or config
        for combo in combos:
            overrides = dict(zip(keys, combo))
            rc, run_name = run_one(args.booksim_bin, config, overrides,
                                   run_name_parts=[f"cfg-{config_id}"])
            had_failure = had_failure or (rc != 0)
            output_dir = overrides.get("output_dir", "sims")
            summary = read_summary(run_name, output_dir)
            row = {"run_name": run_name, "exit_code": rc, "config": config, "config_id": config_id}
            for k, v in overrides.items():
                row[k] = v
            if summary:
                if args.metric:
                    for m in args.metric:
                        row[m] = summary.get(m, "")
                else:
                    # Store a richer default set of metrics to cover latency, power, throughput and stalls
                    default_metrics = [
                        "policy", "power_cap", "total_power_avg", "total_power_peak", "headroom_min",
                        "control_p50", "control_p95", "control_p99", "control_qdel_p99",
                        "batch_p50", "batch_p95", "batch_p99", "batch_qdel_p99",
                        "tail_latency_ratio_batch_over_control",
                        "offered_load_class0", "offered_load_class1",
                        "throughput_flits_class0", "throughput_flits_class1",
                        "stall_xbar_rate_class0", "stall_xbar_rate_class1",
                        "average_packet_latency_0", "accepted_packet_rate_0",
                        "average_packet_latency_1", "accepted_packet_rate_1",
                    ]
                    for m in default_metrics:
                        if m in summary:
                            row[m] = summary[m]
            rows.append(row)

    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output} ({len(rows)} runs)")
    if had_failure:
        sys.exit(1)


if __name__ == "__main__":
    main()
