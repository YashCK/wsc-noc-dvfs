#!/usr/bin/env python3
"""
Plot Power Cap Sweep Results

Generates publication-quality figures showing DVFS policy behavior
across different power caps.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import argparse
from pathlib import Path
import seaborn as sns

# Style configuration
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

POLICY_COLORS = {
    'static': '#1f77b4',      # Blue
    'uniform': '#ff7f0e',     # Orange  
    'hw_reactive': '#2ca02c', # Green
    'queue_pid': '#d62728',   # Red
    'perf_target': '#9467bd', # Purple
}

POLICY_MARKERS = {
    'static': 'o',
    'uniform': 's',
    'hw_reactive': '^',
    'queue_pid': 'D',
    'perf_target': 'v',
}

POLICY_LABELS = {
    'static': 'Static (No DVFS)',
    'uniform': 'Uniform Throttle',
    'hw_reactive': 'HW-Reactive',
    'queue_pid': 'Queue-PID',
    'perf_target': 'Perf-Target',
}


def load_data(csv_path):
    """Load and preprocess sweep results."""
    df = pd.read_csv(csv_path)
    
    # Filter out failed experiments
    df = df[df['converged'] == True].copy()
    
    # Handle missing values
    df['latency_avg'] = pd.to_numeric(df['latency_avg'], errors='coerce')
    df['power_avg'] = pd.to_numeric(df['power_avg'], errors='coerce')
    df['freq_avg'] = pd.to_numeric(df['freq_avg'], errors='coerce')
    
    return df


def plot_latency_vs_cap(df, workload, output_dir):
    """Plot latency vs power cap for a specific workload."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    wdf = df[df['workload'] == workload].copy()
    
    for policy in wdf['policy'].unique():
        pdf = wdf[wdf['policy'] == policy].sort_values('power_cap')
        ax.plot(pdf['power_cap'], pdf['latency_avg'], 
                color=POLICY_COLORS.get(policy, 'gray'),
                marker=POLICY_MARKERS.get(policy, 'o'),
                label=POLICY_LABELS.get(policy, policy),
                linewidth=2, markersize=6)
    
    ax.set_xlabel('Power Cap (W)')
    ax.set_ylabel('Average Latency (cycles)')
    ax.set_title(f'Latency vs Power Cap - {workload}')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'latency_vs_cap_{workload}.png')
    plt.savefig(output_dir / f'latency_vs_cap_{workload}.pdf')
    plt.close()


def plot_power_vs_cap(df, workload, output_dir):
    """Plot actual power vs power cap for a specific workload."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    wdf = df[df['workload'] == workload].copy()
    caps = sorted(wdf['power_cap'].unique())
    
    # Plot 1:1 line (power = cap)
    ax.plot([min(caps), max(caps)], [min(caps), max(caps)], 
            'k--', alpha=0.5, label='Cap = Power')
    
    for policy in wdf['policy'].unique():
        pdf = wdf[wdf['policy'] == policy].sort_values('power_cap')
        ax.plot(pdf['power_cap'], pdf['power_avg'], 
                color=POLICY_COLORS.get(policy, 'gray'),
                marker=POLICY_MARKERS.get(policy, 'o'),
                label=POLICY_LABELS.get(policy, policy),
                linewidth=2, markersize=6)
    
    ax.set_xlabel('Power Cap (W)')
    ax.set_ylabel('Average Power (W)')
    ax.set_title(f'Actual Power vs Power Cap - {workload}')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'power_vs_cap_{workload}.png')
    plt.savefig(output_dir / f'power_vs_cap_{workload}.pdf')
    plt.close()


def plot_freq_vs_cap(df, workload, output_dir):
    """Plot average frequency vs power cap."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    wdf = df[df['workload'] == workload].copy()
    
    for policy in wdf['policy'].unique():
        pdf = wdf[wdf['policy'] == policy].sort_values('power_cap')
        ax.plot(pdf['power_cap'], pdf['freq_avg'], 
                color=POLICY_COLORS.get(policy, 'gray'),
                marker=POLICY_MARKERS.get(policy, 'o'),
                label=POLICY_LABELS.get(policy, policy),
                linewidth=2, markersize=6)
    
    ax.set_xlabel('Power Cap (W)')
    ax.set_ylabel('Average Frequency Scale')
    ax.set_title(f'Frequency Scaling vs Power Cap - {workload}')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.1)
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'freq_vs_cap_{workload}.png')
    plt.savefig(output_dir / f'freq_vs_cap_{workload}.pdf')
    plt.close()


def plot_pareto_frontier(df, workload, output_dir):
    """Plot power-latency Pareto frontier."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    wdf = df[df['workload'] == workload].copy()
    
    for policy in wdf['policy'].unique():
        pdf = wdf[wdf['policy'] == policy].sort_values('power_cap')
        
        # Scatter with cap as color intensity
        caps = pdf['power_cap'].values
        norm = plt.Normalize(caps.min(), caps.max())
        colors = plt.cm.viridis(norm(caps))
        
        for i, (_, row) in enumerate(pdf.iterrows()):
            ax.scatter(row['power_avg'], row['latency_avg'],
                      c=[POLICY_COLORS.get(policy, 'gray')],
                      marker=POLICY_MARKERS.get(policy, 'o'),
                      s=50 + (1 - norm(row['power_cap'])) * 100,  # Larger for tighter caps
                      alpha=0.7,
                      label=POLICY_LABELS.get(policy, policy) if i == 0 else '')
        
        # Connect points with line
        ax.plot(pdf['power_avg'], pdf['latency_avg'],
                color=POLICY_COLORS.get(policy, 'gray'),
                alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Average Power (W)')
    ax.set_ylabel('Average Latency (cycles)')
    ax.set_title(f'Power-Latency Tradeoff - {workload}')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'pareto_{workload}.png')
    plt.savefig(output_dir / f'pareto_{workload}.pdf')
    plt.close()


def plot_cap_compliance_heatmap(df, output_dir):
    """Heatmap showing cap compliance (power/cap ratio) across policies and caps."""
    fig, axes = plt.subplots(1, len(df['workload'].unique()), figsize=(4 * len(df['workload'].unique()), 6))
    if len(df['workload'].unique()) == 1:
        axes = [axes]
    
    for ax, workload in zip(axes, df['workload'].unique()):
        wdf = df[df['workload'] == workload].copy()
        
        # Create pivot table: policy x cap -> power/cap ratio
        wdf['compliance'] = wdf['power_avg'] / wdf['power_cap']
        pivot = wdf.pivot_table(values='compliance', index='policy', columns='power_cap')
        
        # Plot heatmap
        sns.heatmap(pivot, ax=ax, cmap='RdYlGn_r', center=1.0,
                    annot=True, fmt='.2f', cbar_kws={'label': 'Power/Cap Ratio'})
        ax.set_title(f'{workload}')
        ax.set_xlabel('Power Cap (W)')
        ax.set_ylabel('Policy')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cap_compliance_heatmap.png')
    plt.savefig(output_dir / 'cap_compliance_heatmap.pdf')
    plt.close()


def plot_combined_overview(df, output_dir):
    """Create a 2x2 combined overview plot."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Pick first workload for overview
    workload = df['workload'].unique()[0]
    wdf = df[df['workload'] == workload].copy()
    
    # 1. Latency vs Cap
    ax = axes[0, 0]
    for policy in wdf['policy'].unique():
        pdf = wdf[wdf['policy'] == policy].sort_values('power_cap')
        ax.plot(pdf['power_cap'], pdf['latency_avg'], 
                color=POLICY_COLORS.get(policy, 'gray'),
                marker=POLICY_MARKERS.get(policy, 'o'),
                label=POLICY_LABELS.get(policy, policy),
                linewidth=2, markersize=5)
    ax.set_xlabel('Power Cap (W)')
    ax.set_ylabel('Average Latency (cycles)')
    ax.set_title('(a) Latency vs Power Cap')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 2. Power vs Cap
    ax = axes[0, 1]
    caps = sorted(wdf['power_cap'].unique())
    ax.plot([min(caps), max(caps)], [min(caps), max(caps)], 'k--', alpha=0.5, label='Cap = Power')
    for policy in wdf['policy'].unique():
        pdf = wdf[wdf['policy'] == policy].sort_values('power_cap')
        ax.plot(pdf['power_cap'], pdf['power_avg'], 
                color=POLICY_COLORS.get(policy, 'gray'),
                marker=POLICY_MARKERS.get(policy, 'o'),
                label=POLICY_LABELS.get(policy, policy),
                linewidth=2, markersize=5)
    ax.set_xlabel('Power Cap (W)')
    ax.set_ylabel('Average Power (W)')
    ax.set_title('(b) Power vs Cap')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 3. Frequency vs Cap
    ax = axes[1, 0]
    for policy in wdf['policy'].unique():
        pdf = wdf[wdf['policy'] == policy].sort_values('power_cap')
        ax.plot(pdf['power_cap'], pdf['freq_avg'], 
                color=POLICY_COLORS.get(policy, 'gray'),
                marker=POLICY_MARKERS.get(policy, 'o'),
                label=POLICY_LABELS.get(policy, policy),
                linewidth=2, markersize=5)
    ax.set_xlabel('Power Cap (W)')
    ax.set_ylabel('Frequency Scale')
    ax.set_title('(c) Frequency vs Cap')
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 4. Pareto
    ax = axes[1, 1]
    for policy in wdf['policy'].unique():
        pdf = wdf[wdf['policy'] == policy].sort_values('power_cap')
        ax.scatter(pdf['power_avg'], pdf['latency_avg'],
                  c=POLICY_COLORS.get(policy, 'gray'),
                  marker=POLICY_MARKERS.get(policy, 'o'),
                  s=60, alpha=0.7,
                  label=POLICY_LABELS.get(policy, policy))
        ax.plot(pdf['power_avg'], pdf['latency_avg'],
                color=POLICY_COLORS.get(policy, 'gray'),
                alpha=0.3, linewidth=1)
    ax.set_xlabel('Average Power (W)')
    ax.set_ylabel('Average Latency (cycles)')
    ax.set_title('(d) Power-Latency Tradeoff')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'DVFS Policy Comparison - {workload}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'overview_combined.png')
    plt.savefig(output_dir / 'overview_combined.pdf')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot power cap sweep results')
    parser.add_argument('csv_file', type=str, help='Input CSV file from sweep')
    parser.add_argument('--output', type=str, default='sweep_plots',
                        help='Output directory for plots')
    args = parser.parse_args()
    
    # Load data
    df = load_data(args.csv_file)
    print(f"Loaded {len(df)} results from {args.csv_file}")
    print(f"Policies: {df['policy'].unique()}")
    print(f"Workloads: {df['workload'].unique()}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Generate plots for each workload
    for workload in df['workload'].unique():
        print(f"Generating plots for {workload}...")
        plot_latency_vs_cap(df, workload, output_dir)
        plot_power_vs_cap(df, workload, output_dir)
        plot_freq_vs_cap(df, workload, output_dir)
        plot_pareto_frontier(df, workload, output_dir)
    
    # Combined plots
    print("Generating combined plots...")
    plot_cap_compliance_heatmap(df, output_dir)
    plot_combined_overview(df, output_dir)
    
    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()
