#!/usr/bin/env python3
"""
Analyze DVFS policy stress test results and generate plots for research paper.

This script:
1. Collects all stress test results from booksim2/sims/stress_tests/
2. Aggregates summary + epoch data into a combined CSV
3. Generates comparison plots across policies

Usage:
  python3 scripts/analyze_stress_tests.py --outdir plots/stress_analysis
"""

import argparse
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Style settings for publication-quality plots
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (8, 5),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Policy display names and colors
POLICY_CONFIG = {
    'static': {'name': 'Static (Baseline)', 'color': '#7f7f7f', 'marker': 's'},
    'uniform': {'name': 'Uniform Throttle', 'color': '#1f77b4', 'marker': 'o'},
    'hw_reactive': {'name': 'HW-Reactive', 'color': '#2ca02c', 'marker': '^'},
    'queue_pid': {'name': 'Queue-PID', 'color': '#ff7f0e', 'marker': 'D'},
    'perf_target': {'name': 'Perf-Target', 'color': '#d62728', 'marker': 'v'},
}

# Workload display names
WORKLOAD_CONFIG = {
    'uniform_0.6': 'Synthetic Uniform (0.6)',
    'bitcomp_0.8': 'Synthetic Bitcomp (0.8)',
    'hotspot_0.6': 'Synthetic Hotspot (0.6)',
    'netrace_fft': 'FFT Trace',
    'netrace_xapian': 'Xapian Trace',
    'netrace_merged': 'Merged Trace',
}


def parse_run_name(run_name: str) -> Tuple[str, str, Optional[float]]:
    """Parse run name to extract policy, workload, power_cap."""
    parts = run_name.split('_')
    
    # Extract policy
    if run_name.startswith('hw_reactive'):
        policy = 'hw_reactive'
        rest = '_'.join(parts[2:])
    elif run_name.startswith('queue_pid'):
        policy = 'queue_pid'
        rest = '_'.join(parts[2:])
    elif run_name.startswith('perf_target'):
        policy = 'perf_target'
        rest = '_'.join(parts[2:])
    elif run_name.startswith('static'):
        policy = 'static'
        rest = '_'.join(parts[1:])
    elif run_name.startswith('uniform'):
        # Could be static baseline (old naming) or uniform throttle
        if 'netrace' in run_name or len(parts) > 2:
            policy = 'static'  # Old naming before rename
        else:
            policy = 'uniform'
        rest = '_'.join(parts[1:])
    else:
        policy = parts[0]
        rest = '_'.join(parts[1:])
    
    # Extract workload and power cap
    workload = 'unknown'
    power_cap = None
    
    if 'netrace_fft' in rest:
        workload = 'netrace_fft'
    elif 'netrace_xapian' in rest:
        workload = 'netrace_xapian'
    elif 'netrace_merged' in rest:
        workload = 'netrace_merged'
    elif 'uniform' in rest:
        workload = 'uniform_0.6'
    elif 'bitcomp' in rest:
        workload = 'bitcomp_0.8'
    elif 'hotspot' in rest:
        workload = 'hotspot_0.6'
    
    # Extract power cap (e.g., "10W" or "7W")
    for part in rest.split('_'):
        if part.endswith('W'):
            try:
                power_cap = float(part[:-1])
            except ValueError:
                pass
    
    return policy, workload, power_cap


def load_stress_test_results(base_dir: str) -> pd.DataFrame:
    """Load all stress test results from directory."""
    results = []
    
    stress_dir = os.path.join(base_dir, 'booksim2', 'sims', 'stress_tests')
    if not os.path.exists(stress_dir):
        print(f"Warning: stress_tests directory not found at {stress_dir}")
        return pd.DataFrame()
    
    for run_dir in glob.glob(os.path.join(stress_dir, '*')):
        if not os.path.isdir(run_dir):
            continue
        
        run_name = os.path.basename(run_dir)
        policy, workload, power_cap = parse_run_name(run_name)
        
        result = {
            'run_name': run_name,
            'policy': policy,
            'workload': workload,
            'power_cap': power_cap,
        }
        
        # Load summary.csv
        summary_path = os.path.join(run_dir, 'summary.csv')
        if os.path.exists(summary_path):
            try:
                summary_df = pd.read_csv(summary_path)
                if not summary_df.empty:
                    for col in summary_df.columns:
                        if col not in ['policy', 'power_cap']:
                            result[col] = summary_df[col].iloc[0]
            except Exception as e:
                print(f"Warning: Could not read {summary_path}: {e}")
        
        # Load epoch.csv for time-series stats
        epoch_path = os.path.join(run_dir, 'epoch.csv')
        if os.path.exists(epoch_path):
            try:
                epoch_df = pd.read_csv(epoch_path)
                if not epoch_df.empty and 'total_power' in epoch_df.columns:
                    result['power_avg'] = epoch_df['total_power'].mean()
                    result['power_peak'] = epoch_df['total_power'].max()
                    result['power_min'] = epoch_df['total_power'].min()
                    result['power_std'] = epoch_df['total_power'].std()
                    result['num_epochs'] = len(epoch_df)
                    
                    if 'headroom' in epoch_df.columns:
                        result['headroom_avg'] = epoch_df['headroom'].mean()
                        result['headroom_min'] = epoch_df['headroom'].min()
                    
                    if 'class0_p99' in epoch_df.columns:
                        valid_p99 = epoch_df['class0_p99'][epoch_df['class0_p99'] > 0]
                        if len(valid_p99) > 0:
                            result['control_p99_epoch_avg'] = valid_p99.mean()
                            result['control_p99_epoch_max'] = valid_p99.max()
                    
                    if 'class0_throughput' in epoch_df.columns:
                        valid_tput = epoch_df['class0_throughput'][epoch_df['class0_throughput'] > 0]
                        if len(valid_tput) > 0:
                            result['throughput_avg'] = valid_tput.mean()
                            
            except Exception as e:
                print(f"Warning: Could not read {epoch_path}: {e}")
        
        # Load latency.csv
        latency_path = os.path.join(run_dir, 'latency.csv')
        if os.path.exists(latency_path):
            try:
                latency_df = pd.read_csv(latency_path)
                for _, row in latency_df.iterrows():
                    cls = str(row.get('class', ''))
                    if cls == '0':
                        result['control_p50'] = row.get('plat_p50')
                        result['control_p95'] = row.get('plat_p95')
                        result['control_p99'] = row.get('plat_p99')
                    elif cls == '1':
                        result['batch_p50'] = row.get('plat_p50')
                        result['batch_p95'] = row.get('plat_p95')
                        result['batch_p99'] = row.get('plat_p99')
            except Exception as e:
                print(f"Warning: Could not read {latency_path}: {e}")
        
        results.append(result)
    
    return pd.DataFrame(results)


def load_epoch_timeseries(base_dir: str, run_name: str) -> Optional[pd.DataFrame]:
    """Load epoch time-series data for a specific run."""
    epoch_path = os.path.join(base_dir, 'booksim2', 'sims', 'stress_tests', 
                              run_name, 'epoch.csv')
    if os.path.exists(epoch_path):
        try:
            return pd.read_csv(epoch_path)
        except:
            pass
    return None


def plot_policy_comparison_bar(df: pd.DataFrame, outdir: str, metric: str, 
                                ylabel: str, title: str, filename: str):
    """Create bar chart comparing policies across workloads."""
    
    # Filter to valid data
    plot_df = df[df[metric].notna()].copy()
    if plot_df.empty:
        print(f"Warning: No data for {metric}")
        return
    
    # Get unique workloads and policies
    workloads = sorted(plot_df['workload'].unique())
    policies = [p for p in POLICY_CONFIG.keys() if p in plot_df['policy'].unique()]
    
    if not workloads or not policies:
        return
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.arange(len(workloads))
    width = 0.15
    n_policies = len(policies)
    
    for i, policy in enumerate(policies):
        policy_data = plot_df[plot_df['policy'] == policy]
        values = []
        for wl in workloads:
            wl_data = policy_data[policy_data['workload'] == wl]
            if not wl_data.empty:
                values.append(wl_data[metric].iloc[0])
            else:
                values.append(0)
        
        offset = (i - n_policies/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, 
                      label=POLICY_CONFIG[policy]['name'],
                      color=POLICY_CONFIG[policy]['color'],
                      edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([WORKLOAD_CONFIG.get(w, w) for w in workloads], 
                       rotation=30, ha='right')
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, filename))
    plt.close()
    print(f"Saved: {filename}")


def plot_latency_percentiles(df: pd.DataFrame, outdir: str):
    """Plot latency percentile comparison (P50, P95, P99)."""
    
    # Filter to runs with all percentile data
    plot_df = df[df['control_p99'].notna() & df['control_p95'].notna() & 
                 df['control_p50'].notna()].copy()
    
    if plot_df.empty:
        print("Warning: No data for latency percentiles")
        return
    
    # Group by workload
    for workload in plot_df['workload'].unique():
        wl_df = plot_df[plot_df['workload'] == workload]
        policies = [p for p in POLICY_CONFIG.keys() if p in wl_df['policy'].unique()]
        
        if len(policies) < 2:
            continue
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        x = np.arange(len(policies))
        width = 0.25
        
        p50_vals = [wl_df[wl_df['policy'] == p]['control_p50'].iloc[0] if p in wl_df['policy'].values else 0 for p in policies]
        p95_vals = [wl_df[wl_df['policy'] == p]['control_p95'].iloc[0] if p in wl_df['policy'].values else 0 for p in policies]
        p99_vals = [wl_df[wl_df['policy'] == p]['control_p99'].iloc[0] if p in wl_df['policy'].values else 0 for p in policies]
        
        ax.bar(x - width, p50_vals, width, label='P50', color='#4daf4a')
        ax.bar(x, p95_vals, width, label='P95', color='#ff7f00')
        ax.bar(x + width, p99_vals, width, label='P99', color='#e41a1c')
        
        ax.set_ylabel('Latency (cycles)')
        ax.set_title(f'Control Class Latency Percentiles - {WORKLOAD_CONFIG.get(workload, workload)}')
        ax.set_xticks(x)
        ax.set_xticklabels([POLICY_CONFIG[p]['name'] for p in policies], rotation=15, ha='right')
        ax.legend()
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        
        plt.tight_layout()
        filename = f'latency_percentiles_{workload}.png'
        plt.savefig(os.path.join(outdir, filename))
        plt.close()
        print(f"Saved: {filename}")


def plot_power_efficiency(df: pd.DataFrame, outdir: str):
    """Plot power vs latency efficiency scatter."""
    
    plot_df = df[df['power_avg'].notna() & df['control_p99'].notna()].copy()
    
    if plot_df.empty:
        print("Warning: No data for power efficiency plot")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for policy in POLICY_CONFIG.keys():
        policy_df = plot_df[plot_df['policy'] == policy]
        if policy_df.empty:
            continue
        
        ax.scatter(policy_df['power_avg'], policy_df['control_p99'],
                   label=POLICY_CONFIG[policy]['name'],
                   color=POLICY_CONFIG[policy]['color'],
                   marker=POLICY_CONFIG[policy]['marker'],
                   s=100, edgecolors='black', linewidth=0.5)
        
        # Annotate with workload
        for _, row in policy_df.iterrows():
            wl_short = row['workload'].replace('netrace_', '').replace('_0.', '@0.')
            ax.annotate(wl_short, (row['power_avg'], row['control_p99']),
                        textcoords='offset points', xytext=(5, 5),
                        fontsize=8, alpha=0.7)
    
    ax.set_xlabel('Average Power (W)')
    ax.set_ylabel('Control Class P99 Latency (cycles)')
    ax.set_title('Power-Latency Efficiency Comparison')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'power_latency_efficiency.png'))
    plt.close()
    print("Saved: power_latency_efficiency.png")


def plot_power_timeseries(base_dir: str, df: pd.DataFrame, outdir: str, workload: str):
    """Plot power over time for all policies on a specific workload."""
    
    wl_df = df[df['workload'] == workload]
    if wl_df.empty:
        return
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for _, row in wl_df.iterrows():
        policy = row['policy']
        run_name = row['run_name']
        
        epoch_df = load_epoch_timeseries(base_dir, run_name)
        if epoch_df is None or 'total_power' not in epoch_df.columns:
            continue
        
        ax.plot(epoch_df['time'] / 1000, epoch_df['total_power'],
                label=POLICY_CONFIG.get(policy, {}).get('name', policy),
                color=POLICY_CONFIG.get(policy, {}).get('color', 'gray'),
                linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Time (k cycles)')
    ax.set_ylabel('Total Power (W)')
    ax.set_title(f'Power Over Time - {WORKLOAD_CONFIG.get(workload, workload)}')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    filename = f'power_timeseries_{workload}.png'
    plt.savefig(os.path.join(outdir, filename))
    plt.close()
    print(f"Saved: {filename}")


def plot_headroom_comparison(df: pd.DataFrame, outdir: str):
    """Plot headroom (power budget remaining) comparison."""
    
    plot_df = df[df['headroom_avg'].notna()].copy()
    if plot_df.empty:
        print("Warning: No headroom data available")
        return
    
    plot_policy_comparison_bar(
        plot_df, outdir, 
        metric='headroom_avg',
        ylabel='Average Headroom (W)',
        title='Power Budget Headroom by Policy',
        filename='headroom_comparison.png'
    )


def plot_summary_table(df: pd.DataFrame, outdir: str):
    """Create a summary table as an image."""
    
    # Select key metrics
    cols = ['policy', 'workload', 'control_p99', 'power_avg', 'headroom_avg']
    table_df = df[cols].dropna(subset=['control_p99']).copy()
    
    if table_df.empty:
        return
    
    # Format numbers
    table_df['control_p99'] = table_df['control_p99'].apply(lambda x: f"{x:.1f}")
    table_df['power_avg'] = table_df['power_avg'].apply(lambda x: f"{x:.3f}")
    table_df['headroom_avg'] = table_df['headroom_avg'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    
    # Rename columns for display
    table_df.columns = ['Policy', 'Workload', 'P99 Latency', 'Avg Power (W)', 'Headroom (W)']
    
    # Create table figure
    fig, ax = plt.subplots(figsize=(12, len(table_df) * 0.4 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#f0f0f0'] * len(table_df.columns)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.title('DVFS Policy Performance Summary', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'summary_table.png'), bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: summary_table.png")


def generate_all_plots(base_dir: str, outdir: str):
    """Generate all analysis plots."""
    
    os.makedirs(outdir, exist_ok=True)
    
    print("Loading stress test results...")
    df = load_stress_test_results(base_dir)
    
    if df.empty:
        print("No results found!")
        return
    
    # Save aggregated CSV
    csv_path = os.path.join(outdir, 'stress_test_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved aggregated results to: {csv_path}")
    print(f"Found {len(df)} test runs")
    print(f"Policies: {df['policy'].unique().tolist()}")
    print(f"Workloads: {df['workload'].unique().tolist()}")
    
    print("\nGenerating plots...")
    
    # 1. P99 Latency comparison bar chart
    plot_policy_comparison_bar(
        df, outdir,
        metric='control_p99',
        ylabel='P99 Latency (cycles)',
        title='Control Class P99 Latency by Policy',
        filename='p99_latency_comparison.png'
    )
    
    # 2. Average power comparison
    plot_policy_comparison_bar(
        df, outdir,
        metric='power_avg',
        ylabel='Average Power (W)',
        title='Average Power Consumption by Policy',
        filename='power_avg_comparison.png'
    )
    
    # 3. Latency percentile breakdown per workload
    plot_latency_percentiles(df, outdir)
    
    # 4. Power-latency efficiency scatter
    plot_power_efficiency(df, outdir)
    
    # 5. Headroom comparison
    plot_headroom_comparison(df, outdir)
    
    # 6. Power time-series for each workload
    for workload in df['workload'].unique():
        if 'netrace' in workload:  # Most interesting for DVFS behavior
            plot_power_timeseries(base_dir, df, outdir, workload)
    
    # 7. Summary table
    plot_summary_table(df, outdir)
    
    print(f"\nAll plots saved to: {outdir}")


def main():
    parser = argparse.ArgumentParser(description='Analyze DVFS stress test results')
    parser.add_argument('--basedir', default='/Users/yash/Projects/wsc-noc-dvfs',
                        help='Base directory of the project')
    parser.add_argument('--outdir', default='plots/stress_analysis',
                        help='Output directory for plots')
    args = parser.parse_args()
    
    generate_all_plots(args.basedir, args.outdir)


if __name__ == '__main__':
    main()
