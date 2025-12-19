#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class RunResult:
    power_cap: float
    ok: bool
    run_dir: Path
    active_power_mean: float | None
    active_headroom_mean: float | None
    active_throughput_mean: float | None
    freq_mean: float | None
    freq_min: float | None
    freq_max: float | None


def _mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _parse_epoch_csv(path: Path) -> tuple[float | None, float | None, float | None]:
    if not path.exists():
        return None, None, None

    active_power: list[float] = []
    active_headroom: list[float] = []
    active_throughput: list[float] = []

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return None, None, None

        throughput_fields = [h for h in reader.fieldnames if h.endswith("_throughput")]

        for row in reader:
            try:
                total_power = float(row.get("total_power", "0") or 0.0)
            except ValueError:
                continue

            if total_power <= 0.0:
                continue

            active_power.append(total_power)

            try:
                active_headroom.append(float(row.get("headroom", "0") or 0.0))
            except ValueError:
                pass

            thr = 0.0
            for field in throughput_fields:
                try:
                    thr += float(row.get(field, "0") or 0.0)
                except ValueError:
                    continue
            active_throughput.append(thr)

    return _mean(active_power), _mean(active_headroom), _mean(active_throughput)


_FREQ_RE = re.compile(r"domains\{0:freq=([0-9]*\.?[0-9]+)")


def _parse_power_log_freq(path: Path) -> tuple[float | None, float | None, float | None]:
    if not path.exists():
        return None, None, None

    freqs: list[float] = []
    for line in path.read_text(errors="ignore").splitlines():
        m = _FREQ_RE.search(line)
        if not m:
            continue
        try:
            freqs.append(float(m.group(1)))
        except ValueError:
            continue

    if not freqs:
        return None, None, None
    return _mean(freqs), min(freqs), max(freqs)


def _cap_to_token(cap: float) -> str:
    s = f"{cap:.6g}"
    return s.replace("-", "m").replace(".", "p")


def _run_one(
    *,
    booksim: Path,
    config: Path,
    output_dir: Path,
    run_prefix: str,
    power_cap: float,
    dvfs_epoch: int,
    max_samples: int,
    netrace_file: Path | None,
    netrace_use_addr_size: int,
    netrace_class_from_node_types: int,
) -> RunResult:
    run_name = f"{run_prefix}_cap{_cap_to_token(power_cap)}"
    run_dir = output_dir / run_name

    args: list[str] = [
        str(booksim),
        str(config),
        f"power_cap={power_cap}",
        f"dvfs_epoch={dvfs_epoch}",
        f"dvfs_log_interval={dvfs_epoch}",
        f"max_samples={max_samples}",
        f"output_dir={output_dir}",
        f"run_name={run_name}",
    ]

    if netrace_file is not None:
        args += [
            "use_netrace=1",
            f"netrace_file={netrace_file}",
            f"netrace_use_addr_size={netrace_use_addr_size}",
            f"netrace_class_from_node_types={netrace_class_from_node_types}",
        ]

    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    ok = proc.returncode == 0
    if not ok:
        sys.stderr.write(proc.stdout[-8000:] + "\n")

    epoch_csv = run_dir / "epoch.csv"
    power_log = run_dir / "power_log"
    active_power_mean, active_headroom_mean, active_throughput_mean = _parse_epoch_csv(epoch_csv)
    freq_mean, freq_min, freq_max = _parse_power_log_freq(power_log)

    return RunResult(
        power_cap=power_cap,
        ok=ok,
        run_dir=run_dir,
        active_power_mean=active_power_mean,
        active_headroom_mean=active_headroom_mean,
        active_throughput_mean=active_throughput_mean,
        freq_mean=freq_mean,
        freq_min=freq_min,
        freq_max=freq_max,
    )


def _write_results(path: Path, results: Iterable[RunResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "power_cap",
                "ok",
                "active_power_mean",
                "active_headroom_mean",
                "active_throughput_mean",
                "freq_mean",
                "freq_min",
                "freq_max",
                "run_dir",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.power_cap,
                    int(r.ok),
                    "" if r.active_power_mean is None else f"{r.active_power_mean:.6g}",
                    "" if r.active_headroom_mean is None else f"{r.active_headroom_mean:.6g}",
                    "" if r.active_throughput_mean is None else f"{r.active_throughput_mean:.6g}",
                    "" if r.freq_mean is None else f"{r.freq_mean:.6g}",
                    "" if r.freq_min is None else f"{r.freq_min:.6g}",
                    "" if r.freq_max is None else f"{r.freq_max:.6g}",
                    str(r.run_dir),
                ]
            )


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    p = argparse.ArgumentParser(description="Sweep power_cap for queue_pid and summarize freq/power/throughput.")
    p.add_argument(
        "--booksim",
        type=Path,
        default=repo_root / "booksim2/src/booksim",
        help="Path to booksim binary.",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=repo_root / "booksim2/runfiles/flatfly_queue_pid_tuned_netrace.cfg",
        help="Base config file.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "booksim2/sims",
        help="Output directory (each cap gets its own run_name subdir).",
    )
    p.add_argument("--run-prefix", default="queue_pid_tuned", help="Run name prefix.")
    p.add_argument("--dvfs-epoch", type=int, default=10_000)
    p.add_argument("--max-samples", type=int, default=5)
    p.add_argument(
        "--caps",
        type=float,
        nargs="+",
        required=True,
        help="List of power caps (e.g. --caps 0.3 0.4 0.5 0.6).",
    )
    p.add_argument(
        "--netrace-file",
        type=Path,
        default=repo_root / "sniper_trace_to_netrace/workloads/network_trace_merged.log.tra.bz2",
        help="If set and exists, forces use_netrace=1 and overrides netrace_file.",
    )
    p.add_argument("--netrace-use-addr-size", type=int, default=1)
    p.add_argument("--netrace-class-from-node-types", type=int, default=1)
    p.add_argument(
        "--results-csv",
        type=Path,
        default=repo_root / "booksim2/sims/queue_pid_cap_sweep.csv",
        help="Where to write the summarized sweep CSV.",
    )

    args = p.parse_args()

    if not args.booksim.exists():
        sys.stderr.write(f"ERROR: booksim binary not found: {args.booksim}\n")
        return 2
    if not args.config.exists():
        sys.stderr.write(f"ERROR: config not found: {args.config}\n")
        return 2

    netrace_file: Path | None = None
    if args.netrace_file and args.netrace_file.exists():
        netrace_file = args.netrace_file
    else:
        sys.stderr.write(f"WARNING: netrace file not found, running without netrace: {args.netrace_file}\n")

    results: list[RunResult] = []
    any_fail = False
    for cap in args.caps:
        r = _run_one(
            booksim=args.booksim,
            config=args.config,
            output_dir=args.output_dir,
            run_prefix=args.run_prefix,
            power_cap=cap,
            dvfs_epoch=args.dvfs_epoch,
            max_samples=args.max_samples,
            netrace_file=netrace_file,
            netrace_use_addr_size=args.netrace_use_addr_size,
            netrace_class_from_node_types=args.netrace_class_from_node_types,
        )
        results.append(r)
        any_fail |= not r.ok
        freq_span = "" if (r.freq_min is None or r.freq_max is None) else f"freq[{r.freq_min:.3g},{r.freq_max:.3g}]"
        pwr = "" if r.active_power_mean is None else f"pwr={r.active_power_mean:.3g}"
        thr = "" if r.active_throughput_mean is None else f"thr={r.active_throughput_mean:.3g}"
        sys.stdout.write(f"cap={cap:.6g} ok={int(r.ok)} {freq_span} {pwr} {thr} dir={r.run_dir}\n")

    _write_results(args.results_csv, results)
    sys.stdout.write(f"Wrote {args.results_csv}\n")
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

