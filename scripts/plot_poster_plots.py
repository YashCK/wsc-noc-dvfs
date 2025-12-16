#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

NUMERIC_COLS = [
    "power_cap", "control_p99", "control_p95", "control_p50",
    "total_power_avg", "total_power_peak", "tail_latency_ratio_batch_over_control",
    "injection_rate", "dvfs_epoch"
]

def load_best(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Ensure expected columns exist
    for key in ["config_id", "injection_rate", "power_cap", "dvfs_epoch"]:
        if key not in df.columns:
            df[key] = pd.NA
    for col in NUMERIC_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # pick best per config_id + injection_rate + power_cap + dvfs_epoch
    group_keys = ["config_id", "injection_rate", "power_cap", "dvfs_epoch"]
    def pick_best(g: pd.DataFrame) -> pd.Series:
        if g.empty:
            return pd.Series()
        if "control_p99" in g and g["control_p99"].notna().any():
            return g.loc[g["control_p99"].idxmin()]
        if "total_power_peak" in g:
            return g.loc[g["total_power_peak"].idxmin()]
        return g.iloc[0]
    best = df.groupby(group_keys, dropna=False).apply(pick_best).reset_index(drop=True)
    return best

def plot_control_p99(best: pd.DataFrame, outdir: str) -> str:
    plt.figure(figsize=(6, 4))
    for inj, sub in best.groupby("injection_rate"):
        for epoch, sub2 in sub.groupby("dvfs_epoch"):
            sub2 = sub2.sort_values("power_cap")
            if sub2["control_p99"].notna().any():
                label = f"inj={inj}, epoch={int(epoch)}"
                plt.plot(sub2["power_cap"], sub2["control_p99"], marker="o", label=label)
    plt.xlabel("Power cap")
    plt.ylabel("Control P99 latency")
    plt.title("Control P99 vs power cap (best per param combo)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    out = os.path.join(outdir, "control_p99_vs_power_cap.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out

def plot_power_latency(best: pd.DataFrame, outdir: str) -> str:
    plt.figure(figsize=(6, 4))
    markers = {5000: "o", 10000: "s"}
    for inj, sub in best.groupby("injection_rate"):
        for epoch, sub2 in sub.groupby("dvfs_epoch"):
            sub2 = sub2.dropna(subset=["control_p99", "total_power_avg"])
            if sub2.empty:
                continue
            plt.scatter(sub2["total_power_avg"], sub2["control_p99"],
                        label=f"inj={inj}, epoch={int(epoch)}",
                        marker=markers.get(epoch, "o"))
    plt.xlabel("Average power")
    plt.ylabel("Control P99 latency")
    plt.title("Power–latency frontier (best per param combo)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    out = os.path.join(outdir, "control_p99_vs_avg_power.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out

def plot_tail_ratio(best: pd.DataFrame, outdir: str) -> str:
    plt.figure(figsize=(6, 4))
    for inj, sub in best.groupby("injection_rate"):
        sub = sub.sort_values("power_cap")
        if sub["tail_latency_ratio_batch_over_control"].notna().any():
            plt.plot(sub["power_cap"], sub["tail_latency_ratio_batch_over_control"],
                     marker="o", label=f"inj={inj}")
    plt.xlabel("Power cap")
    plt.ylabel("Tail latency ratio (batch / control)")
    plt.title("Tail-latency fairness vs power cap")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    out = os.path.join(outdir, "tail_latency_ratio_vs_power_cap.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out

def main():
    ap = argparse.ArgumentParser(description="Poster plots from sweep_results.csv")
    ap.add_argument("--csv", default="sweep_results.csv")
    ap.add_argument("--outdir", default="plots")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    best = load_best(args.csv)
    if best.empty:
        print("No data to plot.")
        return

    outs = [
        plot_control_p99(best, args.outdir),
        plot_power_latency(best, args.outdir),
        plot_tail_ratio(best, args.outdir),
    ]
    print("Wrote:")
    for o in outs:
        print("  ", o)

if __name__ == "__main__":
    main()
