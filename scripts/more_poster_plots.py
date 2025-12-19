import pandas as pd
import matplotlib.pyplot as plt
import os

# -----------------------------
# Configuration
# -----------------------------
CSV_PATH = "sweep_results.csv"          # path to your CSV
OUTDIR = "plots"
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "legend.fontsize": 12,
    "lines.linewidth": 2,
})

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(CSV_PATH)

# Sort for clean lines
df = df.sort_values(by=["injection_rate", "dvfs_epoch", "power_cap"])

# -----------------------------
# Helper: line plot by groups
# -----------------------------
def plot_by_group(
    x, y, group_col, title, ylabel, filename,
    style_col=None
):
    plt.figure(figsize=(7, 5))
    for key, g in df.groupby(group_col):
        if style_col:
            for style, sg in g.groupby(style_col):
                plt.plot(
                    sg[x], sg[y],
                    marker="o",
                    label=f"{group_col}={key}, {style_col}={style}"
                )
        else:
            plt.plot(g[x], g[y], marker="o", label=f"{group_col}={key}")

    plt.xlabel(x.replace("_", " "))
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, filename))
    plt.close()

# ============================================================
# 1. Control P99 vs Power Cap (PRIMARY POSTER FIGURE)
# ============================================================
plot_by_group(
    x="power_cap",
    y="control_p99",
    group_col="injection_rate",
    style_col="dvfs_epoch",
    title="Control-Class P99 Latency vs Power Cap",
    ylabel="Control P99 Latency (cycles)",
    filename="control_p99_vs_power_cap.png"
)

# ============================================================
# 2. Power–Latency Efficiency Curve
# ============================================================
plt.figure(figsize=(7, 5))
for (inj, epoch), g in df.groupby(["injection_rate", "dvfs_epoch"]):
    plt.scatter(
        g["total_power_avg"],
        g["control_p99"],
        label=f"inj={inj}, epoch={epoch}",
        s=60
    )

plt.xlabel("Average Power")
plt.ylabel("Control P99 Latency (cycles)")
plt.title("Power–Latency Efficiency (Control Traffic)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "p99_vs_avg_power.png"))
plt.close()

# ============================================================
# 3. Tail Latency Fairness (Batch / Control)
# ============================================================
plot_by_group(
    x="power_cap",
    y="tail_latency_ratio_batch_over_control",
    group_col="injection_rate",
    title="Tail Latency Fairness: Batch / Control",
    ylabel="Batch P99 / Control P99",
    filename="tail_latency_ratio.png"
)

# ============================================================
# 4. Throughput vs Power Cap (Class 0 & 1)
# ============================================================
plt.figure(figsize=(7, 5))
for inj, g in df.groupby("injection_rate"):
    plt.plot(
        g["power_cap"], g["throughput_flits_class0"],
        marker="o", label=f"Class 0, inj={inj}"
    )
    plt.plot(
        g["power_cap"], g["throughput_flits_class1"],
        marker="s", linestyle="--", label=f"Class 1, inj={inj}"
    )

plt.xlabel("Power Cap")
plt.ylabel("Throughput (flits / cycle)")
plt.title("Throughput vs Power Cap")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "throughput_vs_power_cap.png"))
plt.close()

# ============================================================
# 5. Headroom vs Power Cap
# ============================================================
plot_by_group(
    x="power_cap",
    y="headroom_min",
    group_col="dvfs_epoch",
    title="Minimum Power Headroom vs Power Cap",
    ylabel="Minimum Headroom",
    filename="headroom_vs_power_cap.png"
)

# ============================================================
# 6. Crossbar Stall Rate vs Injection Rate
# ============================================================
plt.figure(figsize=(7, 5))
for cap, g in df.groupby("power_cap"):
    plt.scatter(
        g["injection_rate"],
        g["stall_xbar_rate_class0"],
        label=f"cap={cap} (class0)",
        marker="o"
    )
    plt.scatter(
        g["injection_rate"],
        g["stall_xbar_rate_class1"],
        label=f"cap={cap} (class1)",
        marker="x"
    )

plt.xlabel("Injection Rate")
plt.ylabel("Crossbar Stall Rate")
plt.title("Crossbar Contention vs Load")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "stall_rate_vs_injection.png"))
plt.close()

print(f"All plots saved to: {OUTDIR}/")
