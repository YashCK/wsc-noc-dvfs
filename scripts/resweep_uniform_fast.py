#!/usr/bin/env python3
"""
Resweep the uniform baseline (uniform_target x power_cap) into a CSV that the
plotting scripts can use as the "best feasible uniform" curve.

This is intentionally biased toward *low* uniform_target values (plus 1.0 for
sanity) so you get cap-feasible points at low power caps without wasting time
on high targets that will never be cap-feasible.

Default output matches the plotting scripts' expected input:
  sweep_results_uniform_tuning.csv
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def _csv_list(values: list[float]) -> str:
    # BookSim's config override parser treats integer-looking literals as "int" typed,
    # and will reject float fields like power_cap when passed as e.g. `power_cap=1`.
    # Force a decimal point for whole numbers: `1.0`, `0.0`, etc.
    out = []
    for v in values:
        s = f"{v:.6g}"
        if "e" not in s and "." not in s:
            s = s + ".0"
        out.append(s)
    return ",".join(out)


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        return list(r)


def _to_float(s: str | None) -> float | None:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--booksim", type=Path, default=repo_root / "booksim2/src/booksim")
    ap.add_argument("--config", type=Path, default=repo_root / "booksim2/runfiles/flatfly_uniform_baseline.cfg")
    ap.add_argument("--output", type=Path, default=repo_root / "sweep_results_uniform_tuning.csv")
    ap.add_argument("--output-dir", type=Path, default=repo_root / "booksim2/sims")

    ap.add_argument("--dvfs-epoch", type=int, default=10_000)
    ap.add_argument("--max-samples", type=int, default=5)

    ap.add_argument(
        "--power-caps",
        default="0.3,0.4,0.5,0.6,0.75,0.9",
        help="Comma-separated caps to sweep.",
    )
    ap.add_argument(
        "--uniform-targets",
        default="0.01,0.02,0.03,0.04,0.05,0.075,0.1,0.15,0.2,0.3,0.4,0.6,1.0",
        help="Comma-separated uniform_target values to sweep.",
    )
    ap.add_argument(
        "--ultra-low",
        action="store_true",
        help="Override --uniform-targets with an ultra-low set (good for low caps like 0.3).",
    )

    ap.add_argument("--netrace-file", type=Path, default=repo_root / "sniper_trace_to_netrace/workloads/network_trace_merged.log.tra.bz2")
    ap.add_argument("--netrace-use-addr-size", type=int, default=1)
    ap.add_argument("--netrace-class-from-node-types", type=int, default=1)

    ap.add_argument("--dry-run", action="store_true", help="Print the sweep_params command and exit.")
    args = ap.parse_args()

    if not args.booksim.exists():
        print(f"ERROR: booksim not found: {args.booksim}", file=sys.stderr)
        return 2
    if not args.config.exists():
        print(f"ERROR: config not found: {args.config}", file=sys.stderr)
        return 2
    if not args.netrace_file.exists():
        print(f"ERROR: netrace_file not found: {args.netrace_file}", file=sys.stderr)
        return 2

    # Parse lists
    try:
        caps = [float(x) for x in args.power_caps.split(",") if x.strip()]
        if args.ultra_low:
            targets = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 1.0]
        else:
            targets = [float(x) for x in args.uniform_targets.split(",") if x.strip()]
    except ValueError as e:
        print(f"ERROR: bad --power-caps/--uniform-targets: {e}", file=sys.stderr)
        return 2

    caps = sorted(set(caps))
    targets = sorted(set(targets))

    cmd = [
        sys.executable,
        str(repo_root / "scripts/sweep_params.py"),
        "--config",
        str(args.config),
        "--booksim-bin",
        str(args.booksim),
        "--output",
        str(args.output),
        "--param",
        f"power_cap={_csv_list(caps)}",
        "--param",
        f"uniform_target={_csv_list(targets)}",
        "--param",
        f"dvfs_epoch={args.dvfs_epoch}",
        "--param",
        f"dvfs_log_interval={args.dvfs_epoch}",
        "--param",
        f"max_samples={args.max_samples}",
        "--param",
        f"output_dir={args.output_dir}",
        "--param",
        "use_netrace=1",
        "--param",
        f"netrace_file={args.netrace_file}",
        "--param",
        f"netrace_use_addr_size={args.netrace_use_addr_size}",
        "--param",
        f"netrace_class_from_node_types={args.netrace_class_from_node_types}",
    ]

    print("Running:", " ".join(cmd))
    if args.dry_run:
        return 0

    proc = subprocess.run(cmd)
    # sweep_params exits 1 if any run fails; we still want to summarize what we got.
    if proc.returncode not in (0, 1):
        return proc.returncode

    rows = _read_rows(args.output)
    if not rows:
        print(f"ERROR: no rows written to {args.output}", file=sys.stderr)
        return 1

    # Summarize feasibility for uniform baseline selection (best-feasible per cap).
    by_cap: dict[float, list[tuple[float, float]]] = {}
    for row in rows:
        cap = _to_float(row.get("power_cap"))
        ut = _to_float(row.get("uniform_target"))
        peak = _to_float(row.get("total_power_peak"))
        rc = int(_to_float(row.get("exit_code")) or 0)
        if cap is None or ut is None or peak is None:
            continue
        if rc != 0:
            continue
        if peak <= cap + 1e-12:
            by_cap.setdefault(cap, []).append((ut, peak))

    print(f"\nWrote {args.output} ({len(rows)} rows)")
    for cap in caps:
        feas = by_cap.get(cap, [])
        if not feas:
            print(f"cap={cap:.3g}: feasible=0  (sweep lower uniform_target values)")
            continue
        best_ut, best_peak = max(feas, key=lambda t: t[0])
        print(f"cap={cap:.3g}: feasible={len(feas)}  best_uniform_target={best_ut:.3g}  peak_power={best_peak:.3g}")

    # Preserve sweep_params semantics: return 1 if any run failed, else 0.
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
