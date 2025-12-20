#!/usr/bin/env python3
"""
Generate Final Paper Plots from Simulation Results

Creates plots for:
- Control/Batch P99 Latency vs Power Cap
- Control/Batch P99 Latency vs Injection Rate  
- Control/Batch Throughput vs Power Cap
- Control/Batch Throughput vs Injection Rate
- DVFS throttling heatmaps per policy
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path
import re
import os

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_CSV = PROJECT_DIR / "final_paper_sims" / "all_results.csv"
OUTPUT_DIR = PROJECT_DIR / "final_paper_plots"

# Plot styling
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'uniform': '#1f77b4',      # Blue
    'hw_reactive': '#ff7f0e',  # Orange  
    'queue_pid': '#2ca02c',    # Green
    'perf_target': '#d62728',  # Red
}
MARKERS = {
    'uniform': 'o',
    'hw_reactive': 's', 
    'queue_pid': '^',
    'perf_target': 'D',
}
POLICY_LABELS = {
    'uniform': 'Uniform',
    'hw_reactive': 'HW Reactive',
    'queue_pid': 'Queue PID',
    'perf_target': 'Perf Target',
}

def load_results():
    """Load and preprocess simulation results."""
    df = pd.read_csv(RESULTS_CSV)
    
    # Filter to converged runs only
    df = df[df['converged'] == True].copy()
    
    # Clean up any NaN values in key columns
    df['control_p99'] = pd.to_numeric(df['control_p99'], errors='coerce')
    df['batch_p99'] = pd.to_numeric(df['batch_p99'], errors='coerce')
    df['control_throughput'] = pd.to_numeric(df['control_throughput'], errors='coerce')
    df['batch_throughput'] = pd.to_numeric(df['batch_throughput'], errors='coerce')
    df['freq_avg'] = pd.to_numeric(df['freq_avg'], errors='coerce')
    df['freq_min'] = pd.to_numeric(df['freq_min'], errors='coerce')
    
    return df


def plot_metric_vs_power_cap(df, topology, metric_col, ylabel, title, filename, 
                              log_scale=False, ylim=None):
    """Plot a metric vs power cap with lines for each policy."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    topo_df = df[df['topology'] == topology]
    
    for policy in ['uniform', 'hw_reactive', 'queue_pid', 'perf_target']:
        policy_df = topo_df[topo_df['policy'] == policy]
        
        # Average across injection rates for each power cap
        grouped = policy_df.groupby('power_cap')[metric_col].mean().reset_index()
        grouped = grouped.dropna()
        
        if not grouped.empty:
            ax.plot(grouped['power_cap'], grouped[metric_col],
                   color=COLORS[policy], marker=MARKERS[policy],
                   label=POLICY_LABELS[policy], linewidth=2, markersize=8)
    
    ax.set_xlabel('Power Cap (W)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'{title} ({topology.capitalize()})', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if log_scale:
        ax.set_yscale('log')
    if ylim:
        ax.set_ylim(ylim)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_metric_vs_injection_rate(df, topology, metric_col, ylabel, title, filename,
                                   log_scale=False, ylim=None):
    """Plot a metric vs injection rate with lines for each policy."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    topo_df = df[df['topology'] == topology]
    
    for policy in ['uniform', 'hw_reactive', 'queue_pid', 'perf_target']:
        policy_df = topo_df[topo_df['policy'] == policy]
        
        # Average across power caps for each injection rate
        grouped = policy_df.groupby('injection_rate')[metric_col].mean().reset_index()
        grouped = grouped.dropna()
        
        if not grouped.empty:
            ax.plot(grouped['injection_rate'], grouped[metric_col],
                   color=COLORS[policy], marker=MARKERS[policy],
                   label=POLICY_LABELS[policy], linewidth=2, markersize=8)
    
    ax.set_xlabel('Injection Rate', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'{title} ({topology.capitalize()})', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if log_scale:
        ax.set_yscale('log')
    if ylim:
        ax.set_ylim(ylim)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_latency_vs_power_cap_by_injection(df, topology, class_type='control'):
    """Plot latency vs power cap with separate lines for each injection rate."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    topo_df = df[df['topology'] == topology]
    metric_col = f'{class_type}_p99'
    injection_rates = sorted(topo_df['injection_rate'].unique())
    
    colors_inj = plt.cm.viridis(np.linspace(0, 0.8, len(injection_rates)))
    
    for idx, policy in enumerate(['uniform', 'hw_reactive', 'queue_pid', 'perf_target']):
        ax = axes[idx]
        policy_df = topo_df[topo_df['policy'] == policy]
        
        for i, inj_rate in enumerate(injection_rates):
            inj_df = policy_df[policy_df['injection_rate'] == inj_rate]
            inj_df = inj_df.sort_values('power_cap')
            inj_df = inj_df.dropna(subset=[metric_col])
            
            if not inj_df.empty:
                ax.plot(inj_df['power_cap'], inj_df[metric_col],
                       color=colors_inj[i], marker='o',
                       label=f'inj={inj_rate}', linewidth=2, markersize=6)
        
        ax.set_xlabel('Power Cap (W)', fontsize=11)
        ax.set_ylabel(f'{class_type.capitalize()} P99 Latency (cycles)', fontsize=11)
        ax.set_title(f'{POLICY_LABELS[policy]}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.suptitle(f'{class_type.capitalize()} Class P99 Latency vs Power Cap ({topology.capitalize()})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{topology}_{class_type}_p99_vs_power_by_inj.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {topology}_{class_type}_p99_vs_power_by_inj.png")


def plot_dvfs_throttling_heatmap(df, topology, policy):
    """
    Plot DVFS throttling as a heatmap showing frequency scale vs power cap and injection rate.
    """
    topo_df = df[(df['topology'] == topology) & (df['policy'] == policy)]
    
    if topo_df.empty:
        print(f"  No data for {topology} {policy}")
        return
    
    # Create pivot table: injection_rate x power_cap -> freq_avg
    # Throttling = 1 - freq_avg (since freq_avg=1.0 means no throttling)
    pivot = topo_df.pivot_table(
        values='freq_avg', 
        index='injection_rate', 
        columns='power_cap',
        aggfunc='mean'
    )
    
    if pivot.empty:
        print(f"  No pivot data for {topology} {policy}")
        return
    
    # Convert to throttling percentage
    throttling = (1 - pivot) * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create heatmap
    im = ax.imshow(throttling.values, cmap='RdYlGn_r', aspect='auto',
                   vmin=0, vmax=50)
    
    # Set ticks
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f'{x:.1f}' for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f'{y:.1f}' for y in pivot.index])
    
    # Labels
    ax.set_xlabel('Power Cap (W)', fontsize=12)
    ax.set_ylabel('Injection Rate', fontsize=12)
    ax.set_title(f'DVFS Throttling % - {POLICY_LABELS[policy]} ({topology.capitalize()})', 
                 fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Throttling %', fontsize=11)
    
    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = throttling.values[i, j]
            if not np.isnan(val):
                text_color = 'white' if val > 25 else 'black'
                ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                       color=text_color, fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{topology}_{policy}_dvfs_throttling.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {topology}_{policy}_dvfs_throttling.png")


def plot_per_router_frequency(topology, policy):
    """
    Plot per-router frequency scaling from output logs.
    Reads the last DVFS epoch from output logs to get per-router frequencies.
    """
    sim_dir = PROJECT_DIR / "final_paper_sims" / topology
    
    # Find all runs for this policy
    runs = list(sim_dir.glob(f'{topology}_{policy}_*'))
    
    if not runs:
        print(f"  No runs found for {topology} {policy}")
        return
    
    # Collect frequency data from multiple runs
    all_freqs = {}
    
    for run_dir in runs:
        output_log = run_dir / "output.log"
        if not output_log.exists():
            continue
        
        # Parse run parameters from directory name
        name = run_dir.name
        match = re.search(r'inj([\d.]+)_cap([\d.]+)', name)
        if not match:
            continue
        
        inj_rate = float(match.group(1))
        power_cap = float(match.group(2))
        
        # Read last DVFS epoch from log
        with open(output_log, 'r') as f:
            content = f.read()
        
        # Find all DVFS epochs
        epochs = re.findall(r'DVFS epoch[^\n]+routers\{([^}]+)\}', content)
        if not epochs:
            continue
        
        # Parse last epoch's router frequencies
        last_epoch = epochs[-1]
        router_freqs = re.findall(r'(\d+):freq=([\d.]+),power', last_epoch)
        
        if router_freqs:
            freqs = {int(r[0]): float(r[1]) for r in router_freqs}
            key = (inj_rate, power_cap)
            all_freqs[key] = freqs
    
    if not all_freqs:
        print(f"  No frequency data found for {topology} {policy}")
        return
    
    # Create subplot grid based on number of runs
    n_runs = len(all_freqs)
    n_cols = min(4, n_runs)
    n_rows = (n_runs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    if n_runs == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Determine grid size for routers
    n_routers = len(list(all_freqs.values())[0])
    if topology == 'flatfly':
        grid_size = (4, 4)  # 16 routers
    else:
        grid_size = (8, 8)  # 64 routers
    
    sorted_keys = sorted(all_freqs.keys())
    
    for idx, (inj_rate, power_cap) in enumerate(sorted_keys):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        freqs = all_freqs[(inj_rate, power_cap)]
        
        # Create grid
        freq_grid = np.zeros(grid_size)
        for router_id, freq in freqs.items():
            row = router_id // grid_size[1]
            col = router_id % grid_size[1]
            if row < grid_size[0] and col < grid_size[1]:
                freq_grid[row, col] = freq
        
        im = ax.imshow(freq_grid, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='equal')
        ax.set_title(f'inj={inj_rate}, cap={power_cap}W', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide unused subplots
    for idx in range(n_runs, len(axes)):
        axes[idx].set_visible(False)
    
    # Add colorbar
    fig.colorbar(im, ax=axes, label='Frequency Scale', shrink=0.8)
    
    plt.suptitle(f'Per-Router Frequency Scale - {POLICY_LABELS[policy]} ({topology.capitalize()})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{topology}_{policy}_per_router_freq.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {topology}_{policy}_per_router_freq.png")


def plot_comparison_all_policies(df, topology):
    """Create a single comparison plot with all key metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    topo_df = df[df['topology'] == topology]
    
    # Plot 1: Control P99 vs Power Cap
    ax = axes[0, 0]
    for policy in ['uniform', 'hw_reactive', 'queue_pid', 'perf_target']:
        policy_df = topo_df[topo_df['policy'] == policy]
        grouped = policy_df.groupby('power_cap')['control_p99'].mean().reset_index()
        grouped = grouped.dropna()
        if not grouped.empty:
            ax.plot(grouped['power_cap'], grouped['control_p99'],
                   color=COLORS[policy], marker=MARKERS[policy],
                   label=POLICY_LABELS[policy], linewidth=2, markersize=6)
    ax.set_xlabel('Power Cap (W)')
    ax.set_ylabel('Control P99 Latency (cycles)')
    ax.set_title('Control P99 vs Power Cap')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 2: Batch P99 vs Power Cap
    ax = axes[0, 1]
    for policy in ['uniform', 'hw_reactive', 'queue_pid', 'perf_target']:
        policy_df = topo_df[topo_df['policy'] == policy]
        grouped = policy_df.groupby('power_cap')['batch_p99'].mean().reset_index()
        grouped = grouped.dropna()
        if not grouped.empty:
            ax.plot(grouped['power_cap'], grouped['batch_p99'],
                   color=COLORS[policy], marker=MARKERS[policy],
                   label=POLICY_LABELS[policy], linewidth=2, markersize=6)
    ax.set_xlabel('Power Cap (W)')
    ax.set_ylabel('Batch P99 Latency (cycles)')
    ax.set_title('Batch P99 vs Power Cap')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 3: Control Throughput vs Power Cap
    ax = axes[0, 2]
    for policy in ['uniform', 'hw_reactive', 'queue_pid', 'perf_target']:
        policy_df = topo_df[topo_df['policy'] == policy]
        grouped = policy_df.groupby('power_cap')['control_throughput'].mean().reset_index()
        grouped = grouped.dropna()
        if not grouped.empty:
            ax.plot(grouped['power_cap'], grouped['control_throughput'],
                   color=COLORS[policy], marker=MARKERS[policy],
                   label=POLICY_LABELS[policy], linewidth=2, markersize=6)
    ax.set_xlabel('Power Cap (W)')
    ax.set_ylabel('Control Throughput')
    ax.set_title('Control Throughput vs Power Cap')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Control P99 vs Injection Rate
    ax = axes[1, 0]
    for policy in ['uniform', 'hw_reactive', 'queue_pid', 'perf_target']:
        policy_df = topo_df[topo_df['policy'] == policy]
        grouped = policy_df.groupby('injection_rate')['control_p99'].mean().reset_index()
        grouped = grouped.dropna()
        if not grouped.empty:
            ax.plot(grouped['injection_rate'], grouped['control_p99'],
                   color=COLORS[policy], marker=MARKERS[policy],
                   label=POLICY_LABELS[policy], linewidth=2, markersize=6)
    ax.set_xlabel('Injection Rate')
    ax.set_ylabel('Control P99 Latency (cycles)')
    ax.set_title('Control P99 vs Injection Rate')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 5: Batch P99 vs Injection Rate
    ax = axes[1, 1]
    for policy in ['uniform', 'hw_reactive', 'queue_pid', 'perf_target']:
        policy_df = topo_df[topo_df['policy'] == policy]
        grouped = policy_df.groupby('injection_rate')['batch_p99'].mean().reset_index()
        grouped = grouped.dropna()
        if not grouped.empty:
            ax.plot(grouped['injection_rate'], grouped['batch_p99'],
                   color=COLORS[policy], marker=MARKERS[policy],
                   label=POLICY_LABELS[policy], linewidth=2, markersize=6)
    ax.set_xlabel('Injection Rate')
    ax.set_ylabel('Batch P99 Latency (cycles)')
    ax.set_title('Batch P99 vs Injection Rate')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 6: Average Frequency vs Power Cap (throttling indicator)
    ax = axes[1, 2]
    for policy in ['uniform', 'hw_reactive', 'queue_pid', 'perf_target']:
        policy_df = topo_df[topo_df['policy'] == policy]
        grouped = policy_df.groupby('power_cap')['freq_avg'].mean().reset_index()
        grouped = grouped.dropna()
        if not grouped.empty:
            ax.plot(grouped['power_cap'], grouped['freq_avg'],
                   color=COLORS[policy], marker=MARKERS[policy],
                   label=POLICY_LABELS[policy], linewidth=2, markersize=6)
    ax.set_xlabel('Power Cap (W)')
    ax.set_ylabel('Average Frequency Scale')
    ax.set_title('DVFS Frequency vs Power Cap')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.05)
    
    plt.suptitle(f'DVFS Policy Comparison - {topology.capitalize()} Topology',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{topology}_policy_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {topology}_policy_comparison.png")


def main():
    print("Loading results...")
    df = load_results()
    print(f"Loaded {len(df)} converged experiments")
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    for topology in ['flatfly', 'mesh']:
        print(f"\n=== Generating plots for {topology} ===")
        
        # 1. Control P99 Latency vs Power Cap
        print("\nLatency plots:")
        plot_metric_vs_power_cap(
            df, topology, 'control_p99',
            'Control Class P99 Latency (cycles)',
            'Control Class P99 Latency vs Power Cap',
            f'{topology}_control_p99_vs_power.png',
            log_scale=True
        )
        
        # 2. Control P99 Latency vs Injection Rate
        plot_metric_vs_injection_rate(
            df, topology, 'control_p99',
            'Control Class P99 Latency (cycles)',
            'Control Class P99 Latency vs Injection Rate',
            f'{topology}_control_p99_vs_injection.png',
            log_scale=True
        )
        
        # 3. Batch P99 Latency vs Power Cap
        plot_metric_vs_power_cap(
            df, topology, 'batch_p99',
            'Batch Class P99 Latency (cycles)',
            'Batch Class P99 Latency vs Power Cap',
            f'{topology}_batch_p99_vs_power.png',
            log_scale=True
        )
        
        # 4. Batch P99 Latency vs Injection Rate
        plot_metric_vs_injection_rate(
            df, topology, 'batch_p99',
            'Batch Class P99 Latency (cycles)',
            'Batch Class P99 Latency vs Injection Rate',
            f'{topology}_batch_p99_vs_injection.png',
            log_scale=True
        )
        
        # 5. Control Throughput vs Power Cap
        print("\nThroughput plots:")
        plot_metric_vs_power_cap(
            df, topology, 'control_throughput',
            'Control Class Throughput',
            'Control Class Throughput vs Power Cap',
            f'{topology}_control_throughput_vs_power.png'
        )
        
        # 6. Control Throughput vs Injection Rate
        plot_metric_vs_injection_rate(
            df, topology, 'control_throughput',
            'Control Class Throughput',
            'Control Class Throughput vs Injection Rate',
            f'{topology}_control_throughput_vs_injection.png'
        )
        
        # 7. Batch Throughput vs Power Cap
        plot_metric_vs_power_cap(
            df, topology, 'batch_throughput',
            'Batch Class Throughput',
            'Batch Class Throughput vs Power Cap',
            f'{topology}_batch_throughput_vs_power.png'
        )
        
        # 8. Batch Throughput vs Injection Rate
        plot_metric_vs_injection_rate(
            df, topology, 'batch_throughput',
            'Batch Class Throughput',
            'Batch Class Throughput vs Injection Rate',
            f'{topology}_batch_throughput_vs_injection.png'
        )
        
        # 9. DVFS throttling heatmaps per policy
        print("\nDVFS throttling heatmaps:")
        for policy in ['uniform', 'hw_reactive', 'queue_pid', 'perf_target']:
            plot_dvfs_throttling_heatmap(df, topology, policy)
        
        # 10. Per-router frequency plots
        print("\nPer-router frequency plots:")
        for policy in ['uniform', 'hw_reactive', 'queue_pid', 'perf_target']:
            plot_per_router_frequency(topology, policy)
        
        # 11. Combined comparison plot
        print("\nComparison plot:")
        plot_comparison_all_policies(df, topology)
        
        # 12. Detailed latency breakdown by injection rate
        print("\nDetailed latency by injection rate:")
        plot_latency_vs_power_cap_by_injection(df, topology, 'control')
        plot_latency_vs_power_cap_by_injection(df, topology, 'batch')
    
    print(f"\n=== All plots saved to {OUTPUT_DIR} ===")
    print(f"Total files: {len(list(OUTPUT_DIR.glob('*.png')))}")


if __name__ == "__main__":
    main()
