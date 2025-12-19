#!/usr/bin/env python3
"""
Compute a "realistic" P99 SLO as a multiplier of the unconstrained baseline.

Method:
  1) Run the given config with max frequency (uniform_target=1.0) and a high cap
  2) Read class0 P99 latency from latency.csv
  3) Recommend SLO = multiplier * baseline_p99

Example (netrace):
  python3 scripts/compute_slo_from_baseline.py \
    --config booksim2/runfiles/flatfly_hw_reactive_tuned.cfg \
    --netrace-file sniper_trace_to_netrace/workloads/network_trace_merged.log.tra.bz2 \
    --mult 2.0
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Optional


def _read_class_p99(latency_csv: Path, cls: str = "0") -> Optional[float]:
    if not latency_csv.exists():
        return None
    with latency_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("class", "")).strip() != cls:
                continue
            v = row.get("plat_p99") or row.get("p99") or row.get("lat_p99")
            if v is None:
                return None
            try:
                return float(v)
            except ValueError:
                return None
    return None


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--booksim", type=Path, default=repo_root / "booksim2/src/booksim")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=repo_root / "booksim2/sims")
    ap.add_argument("--run-name", default="_baseline_maxfreq")
    ap.add_argument("--mult", type=float, default=2.0, help="SLO multiplier over baseline p99 (e.g. 1.5 or 2.0).")
    ap.add_argument("--power-cap", type=float, default=10.0, help="High cap for baseline run.")
    ap.add_argument("--dvfs-epoch", type=int, default=10_000, help="Epoch (only used to drive uniform policy updates/logging).")
    ap.add_argument("--max-samples", type=int, default=5, help="Keep this small; you only need a baseline p99.")

    # Netrace (optional)
    ap.add_argument("--netrace-file", type=Path, default=None)
    ap.add_argument("--netrace-use-addr-size", type=int, default=1)
    ap.add_argument("--netrace-class-from-node-types", type=int, default=1)
    args = ap.parse_args()

    if not args.booksim.exists():
        print(f"ERROR: booksim not found: {args.booksim}", file=sys.stderr)
        return 2
    if not args.config.exists():
        print(f"ERROR: config not found: {args.config}", file=sys.stderr)
        return 2
    if args.mult <= 1.0:
        print("ERROR: --mult should be > 1.0 (e.g. 1.5 or 2.0).", file=sys.stderr)
        return 2

    cmd = [
        str(args.booksim),
        str(args.config),
        "dvfs_policy=uniform",
        "uniform_target=1.0",
        f"power_cap={args.power_cap}",
        f"dvfs_epoch={args.dvfs_epoch}",
        f"dvfs_log_interval={args.dvfs_epoch}",
        f"max_samples={args.max_samples}",
        f"output_dir={args.output_dir}",
        f"run_name={args.run_name}",
    ]

    if args.netrace_file is not None:
        nf = args.netrace_file
        if not nf.is_absolute():
            nf = (repo_root / nf).resolve()
        if not nf.exists():
            print(f"ERROR: netrace_file not found: {nf}", file=sys.stderr)
            return 2
        cmd += [
            "use_netrace=1",
            f"netrace_file={nf}",
            f"netrace_use_addr_size={args.netrace_use_addr_size}",
            f"netrace_class_from_node_types={args.netrace_class_from_node_types}",
        ]

    print("Running baseline:", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-8000:], file=sys.stderr)
        print(f"ERROR: baseline run failed (exit {proc.returncode})", file=sys.stderr)
        return 1

    run_dir = args.output_dir / args.run_name
    latency_csv = run_dir / "latency.csv"
    p99 = _read_class_p99(latency_csv, cls="0")
    if p99 is None or p99 <= 0:
        print(f"ERROR: could not read valid class0 p99 from {latency_csv}", file=sys.stderr)
        return 1

    slo = args.mult * p99
    print(f"Baseline control p99: {p99:.6g} cycles")
    print(f"Recommended SLO (mult={args.mult:g}): {slo:.6g} cycles")
    print(f"Use in plotting: --slo-p99 {slo:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

