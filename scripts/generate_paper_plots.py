#!/usr/bin/env python3
"""
Comprehensive DVFS policy analysis plots for research paper.

Generates publication-quality figures comparing:
- Policy performance under different workloads
- Power-latency tradeoffs (Pareto analysis)
- Frequency scaling behavior over time
- SLO compliance analysis
- Fairness metrics

Usage:
  python3 scripts/generate_paper_plots.py --outdir plots/paper_figures
"""

import argparse
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (6, 4),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

# Policy configuration
POLICY_CONFIG = {
    'static': {'name': 'Static (No DVFS)', 'color': '#7f7f7f', 'marker': 's', 'linestyle': '-'},
    'uniform': {'name': 'Uniform Throttle', 'color': '#1f77b4', 'marker': 'o', 'linestyle': '--'},
    'hw_reactive': {'name': 'HW-Reactive', 'color': '#2ca02c', 'marker': '^', 'linestyle': '-.'},
    'queue_pid': {'name': 'Queue-PID', 'color': '#ff7f0e', 'marker': 'D', 'linestyle': ':'},
    'perf_target': {'name': 'Perf-Target', 'color': '#d62728', 'marker': 'v', 'linestyle': '-'},
}

WORKLOAD_CONFIG = {
    'uniform_0.6': {'name': 'Uniform (0.6)', 'short': 'Uniform'},
    'bitcomp_0.8': {'name': 'Bitcomp (0.8)', 'short': 'Bitcomp'},
    'hotspot_0.6': {'name': 'Hotspot (0.6)', 'short': 'Hotspot'},
    'netrace_fft': {'name': 'FFT Trace', 'short': 'FFT'},
    'netrace_xapian': {'name': 'Xapian Trace', 'short': 'Xapian'},
    'netrace_merged': {'name': 'Merged Trace', 'short': 'Merged'},
}


def parse_run_name(run_name: str) -> Tuple[str, str, Optional[float]]:
    """Parse run name to extract policy, workload, power_cap."""
    parts = run_name.split('_')
    
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
        if 'netrace' in run_name:
            policy = 'static'
        else:
            policy = 'uniform'
        rest = '_'.join(parts[1:])
    else:
        policy = parts[0]
        rest = '_'.join(parts[1:])
    
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
    
    for part in rest.split('_'):
        if part.endswith('W'):
            try:
                power_cap = float(part[:-1])
            except ValueError:
                pass
    
    return policy, workload, power_cap


def load_all_results(base_dir: str) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Load all stress test results and epoch time-series data."""
    results = []
    epoch_data = {}
    
    stress_dir = os.path.join(base_dir, 'booksim2', 'sims', 'stress_tests')
    if not os.path.exists(stress_dir):
        return pd.DataFrame(), {}
    
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
        
        # Load summary
        summary_path = os.path.join(run_dir, 'summary.csv')
        if os.path.exists(summary_path):
            try:
                summary_df = pd.read_csv(summary_path)
                if not summary_df.empty:
                    for col in summary_df.columns:
                        if col not in ['policy', 'power_cap']:
                            result[col] = summary_df[col].iloc[0]
            except:
                pass
        
        # Load epoch data
        epoch_path = os.path.join(run_dir, 'epoch.csv')
        if os.path.exists(epoch_path):
            try:
                epoch_df = pd.read_csv(epoch_path)
                if not epoch_df.empty:
                    epoch_data[run_name] = epoch_df
                    
                    if 'total_power' in epoch_df.columns:
                        result['power_avg'] = epoch_df['total_power'].mean()
                        result['power_peak'] = epoch_df['total_power'].max()
                        result['power_std'] = epoch_df['total_power'].std()
                        result['num_epochs'] = len(epoch_df)
                    
                    if 'headroom' in epoch_df.columns:
                        result['headroom_avg'] = epoch_df['headroom'].mean()
                        result['headroom_min'] = epoch_df['headroom'].min()
                    
                    if 'class0_p99' in epoch_df.columns:
                        valid_p99 = epoch_df['class0_p99'][epoch_df['class0_p99'] > 0]
                        if len(valid_p99) > 0:
                            result['control_p99_avg'] = valid_p99.mean()
                            result['control_p99_max'] = valid_p99.max()
                    
                    if 'class0_throughput' in epoch_df.columns:
                        valid_tput = epoch_df['class0_throughput'][epoch_df['class0_throughput'] > 0]
                        if len(valid_tput) > 0:
                            result['throughput_avg'] = valid_tput.mean()
            except:
                pass
        
        # Load latency data
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
            except:
                pass
        
        results.append(result)
    
    return pd.DataFrame(results), epoch_data


# =============================================================================
# Figure 1: Policy Comparison - Latency & Power (grouped bar chart)
# =============================================================================
def fig_policy_comparison(df: pd.DataFrame, outdir: str):
    """Two-panel figure comparing latency and power across policies."""
    
    # Filter to workloads with good data
    plot_df = df[df['control_p99'].notna() & df['power_avg'].notna()].copy()
    if plot_df.empty:
        return
    
    workloads = sorted(plot_df['workload'].unique())
    policies = [p for p in POLICY_CONFIG.keys() if p in plot_df['policy'].unique()]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    x = np.arange(len(workloads))
    width = 0.18
    
    # Panel A: P99 Latency
    ax = axes[0]
    for i, policy in enumerate(policies):
        policy_data = plot_df[plot_df['policy'] == policy]
        values = []
        for wl in workloads:
            wl_data = policy_data[policy_data['workload'] == wl]
            values.append(wl_data['control_p99'].iloc[0] if not wl_data.empty else 0)
        
        offset = (i - len(policies)/2 + 0.5) * width
        ax.bar(x + offset, values, width, 
               label=POLICY_CONFIG[policy]['name'],
               color=POLICY_CONFIG[policy]['color'],
               edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('P99 Latency (cycles)')
    ax.set_title('(a) Control Class Tail Latency')
    ax.set_xticks(x)
    ax.set_xticklabels([WORKLOAD_CONFIG.get(w, {}).get('short', w) for w in workloads])
    ax.legend(loc='upper right', ncol=2, fontsize=8)
    
    # Panel B: Average Power
    ax = axes[1]
    for i, policy in enumerate(policies):
        policy_data = plot_df[plot_df['policy'] == policy]
        values = []
        for wl in workloads:
            wl_data = policy_data[policy_data['workload'] == wl]
            values.append(wl_data['power_avg'].iloc[0] if not wl_data.empty else 0)
        
        offset = (i - len(policies)/2 + 0.5) * width
        ax.bar(x + offset, values, width,
               color=POLICY_CONFIG[policy]['color'],
               edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Average Power (W)')
    ax.set_title('(b) Network Power Consumption')
    ax.set_xticks(x)
    ax.set_xticklabels([WORKLOAD_CONFIG.get(w, {}).get('short', w) for w in workloads])
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'fig1_policy_comparison.pdf'))
    plt.savefig(os.path.join(outdir, 'fig1_policy_comparison.png'))
    plt.close()
    print("Saved: fig1_policy_comparison.pdf/png")


# =============================================================================
# Figure 2: Power-Latency Tradeoff (Pareto frontier)
# =============================================================================
def fig_power_latency_tradeoff(df: pd.DataFrame, outdir: str):
    """Scatter plot showing power vs latency tradeoff with Pareto frontier."""
    
    plot_df = df[df['power_avg'].notna() & df['control_p99'].notna()].copy()
    if plot_df.empty:
        return
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Plot each policy
    for policy in POLICY_CONFIG.keys():
        policy_df = plot_df[plot_df['policy'] == policy]
        if policy_df.empty:
            continue
        
        ax.scatter(policy_df['power_avg'], policy_df['control_p99'],
                   label=POLICY_CONFIG[policy]['name'],
                   color=POLICY_CONFIG[policy]['color'],
                   marker=POLICY_CONFIG[policy]['marker'],
                   s=120, edgecolors='black', linewidth=0.5, alpha=0.8)
        
        # Annotate with workload
        for _, row in policy_df.iterrows():
            wl_short = WORKLOAD_CONFIG.get(row['workload'], {}).get('short', row['workload'])
            ax.annotate(wl_short, (row['power_avg'], row['control_p99']),
                        textcoords='offset points', xytext=(5, 3),
                        fontsize=7, alpha=0.7)
    
    # Find and highlight Pareto-optimal points (lower power AND lower latency is better)
    pareto_points = []
    for _, row in plot_df.iterrows():
        is_dominated = False
        for _, other in plot_df.iterrows():
            if (other['power_avg'] <= row['power_avg'] and 
                other['control_p99'] <= row['control_p99'] and
                (other['power_avg'] < row['power_avg'] or other['control_p99'] < row['control_p99'])):
                is_dominated = True
                break
        if not is_dominated:
            pareto_points.append((row['power_avg'], row['control_p99']))
    
    if pareto_points:
        pareto_points.sort(key=lambda p: p[0])
        pareto_x, pareto_y = zip(*pareto_points)
        ax.plot(pareto_x, pareto_y, 'k--', alpha=0.5, linewidth=1.5, label='Pareto Frontier')
    
    ax.set_xlabel('Average Power (W)')
    ax.set_ylabel('Control Class P99 Latency (cycles)')
    ax.set_title('Power-Latency Efficiency Tradeoff')
    ax.legend(loc='upper right')
    
    # Add efficiency region annotation
    ax.annotate('Better\n(Lower is better)', xy=(0.15, 0.15), xycoords='axes fraction',
                fontsize=10, ha='center', alpha=0.5,
                bbox=dict(boxstyle='round', facecolor='green', alpha=0.1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'fig2_power_latency_tradeoff.pdf'))
    plt.savefig(os.path.join(outdir, 'fig2_power_latency_tradeoff.png'))
    plt.close()
    print("Saved: fig2_power_latency_tradeoff.pdf/png")


# =============================================================================
# Figure 3: Power Time-Series Comparison
# =============================================================================
def fig_power_timeseries(epoch_data: Dict[str, pd.DataFrame], df: pd.DataFrame, 
                         outdir: str, workload: str = 'netrace_xapian'):
    """Time-series plot of power for different policies on same workload."""
    
    wl_df = df[df['workload'] == workload]
    if wl_df.empty:
        return
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    for _, row in wl_df.iterrows():
        policy = row['policy']
        run_name = row['run_name']
        
        if run_name not in epoch_data:
            continue
        
        epoch_df = epoch_data[run_name]
        if 'total_power' not in epoch_df.columns:
            continue
        
        cfg = POLICY_CONFIG.get(policy, {'color': 'gray', 'linestyle': '-'})
        ax.plot(epoch_df['time'] / 1000, epoch_df['total_power'],
                label=cfg.get('name', policy),
                color=cfg['color'],
                linestyle=cfg['linestyle'],
                linewidth=1.5, alpha=0.9)
    
    # Add power cap line if available
    if not wl_df.empty and pd.notna(wl_df['power_cap'].iloc[0]):
        cap = wl_df['power_cap'].iloc[0]
        ax.axhline(y=cap, color='red', linestyle=':', linewidth=2, alpha=0.7, 
                   label=f'Power Cap ({cap}W)')
    
    ax.set_xlabel('Time (k cycles)')
    ax.set_ylabel('Total Power (W)')
    ax.set_title(f'Power Consumption Over Time - {WORKLOAD_CONFIG.get(workload, {}).get("name", workload)}')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    filename = f'fig3_power_timeseries_{workload}'
    plt.savefig(os.path.join(outdir, f'{filename}.pdf'))
    plt.savefig(os.path.join(outdir, f'{filename}.png'))
    plt.close()
    print(f"Saved: {filename}.pdf/png")


# =============================================================================
# Figure 4: Latency Percentile Breakdown
# =============================================================================
def fig_latency_breakdown(df: pd.DataFrame, outdir: str, workload: str = 'uniform_0.6'):
    """Stacked bar chart showing P50, P95, P99 latency breakdown."""
    
    wl_df = df[(df['workload'] == workload) & 
               df['control_p50'].notna() & 
               df['control_p95'].notna() & 
               df['control_p99'].notna()].copy()
    
    if wl_df.empty:
        return
    
    policies = [p for p in POLICY_CONFIG.keys() if p in wl_df['policy'].unique()]
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    x = np.arange(len(policies))
    width = 0.6
    
    # Get values
    p50_vals = []
    p95_delta = []  # P95 - P50
    p99_delta = []  # P99 - P95
    
    for policy in policies:
        row = wl_df[wl_df['policy'] == policy].iloc[0]
        p50 = row['control_p50']
        p95 = row['control_p95']
        p99 = row['control_p99']
        
        p50_vals.append(p50)
        p95_delta.append(max(0, p95 - p50))
        p99_delta.append(max(0, p99 - p95))
    
    # Stacked bar
    ax.bar(x, p50_vals, width, label='P50', color='#4daf4a')
    ax.bar(x, p95_delta, width, bottom=p50_vals, label='P50-P95', color='#ff7f00')
    bottom2 = [p50 + p95d for p50, p95d in zip(p50_vals, p95_delta)]
    ax.bar(x, p99_delta, width, bottom=bottom2, label='P95-P99 (Tail)', color='#e41a1c')
    
    ax.set_ylabel('Latency (cycles)')
    ax.set_title(f'Latency Distribution - {WORKLOAD_CONFIG.get(workload, {}).get("name", workload)}')
    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_CONFIG[p]['name'] for p in policies], rotation=15, ha='right')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    filename = f'fig4_latency_breakdown_{workload}'
    plt.savefig(os.path.join(outdir, f'{filename}.pdf'))
    plt.savefig(os.path.join(outdir, f'{filename}.png'))
    plt.close()
    print(f"Saved: {filename}.pdf/png")


# =============================================================================
# Figure 5: Headroom (Power Budget) Analysis
# =============================================================================
def fig_headroom_analysis(df: pd.DataFrame, outdir: str):
    """Bar chart showing average power headroom by policy."""
    
    plot_df = df[df['headroom_avg'].notna()].copy()
    if plot_df.empty:
        return
    
    # Group by policy and average across workloads
    policy_headroom = plot_df.groupby('policy').agg({
        'headroom_avg': 'mean',
        'headroom_min': 'min',
    }).reset_index()
    
    policies = [p for p in POLICY_CONFIG.keys() if p in policy_headroom['policy'].values]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    x = np.arange(len(policies))
    width = 0.5
    
    avg_vals = [policy_headroom[policy_headroom['policy'] == p]['headroom_avg'].iloc[0] 
                for p in policies]
    colors = [POLICY_CONFIG[p]['color'] for p in policies]
    
    bars = ax.bar(x, avg_vals, width, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Average Headroom (W)')
    ax.set_title('Power Budget Headroom by Policy\n(Higher = More Room Under Cap)')
    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_CONFIG[p]['name'] for p in policies], rotation=15, ha='right')
    
    # Add value labels
    for bar, val in zip(bars, avg_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'fig5_headroom_analysis.pdf'))
    plt.savefig(os.path.join(outdir, 'fig5_headroom_analysis.png'))
    plt.close()
    print("Saved: fig5_headroom_analysis.pdf/png")


# =============================================================================
# Figure 6: Summary Heatmap
# =============================================================================
def fig_summary_heatmap(df: pd.DataFrame, outdir: str, metric: str = 'control_p99'):
    """Heatmap showing metric across policies and workloads."""
    
    plot_df = df[df[metric].notna()].copy()
    if plot_df.empty:
        return
    
    # Pivot to create matrix
    pivot = plot_df.pivot_table(index='policy', columns='workload', 
                                 values=metric, aggfunc='first')
    
    # Reorder
    policy_order = [p for p in POLICY_CONFIG.keys() if p in pivot.index]
    pivot = pivot.reindex(policy_order)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
    
    # Labels
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels([WORKLOAD_CONFIG.get(w, {}).get('short', w) for w in pivot.columns])
    ax.set_yticklabels([POLICY_CONFIG.get(p, {}).get('name', p) for p in pivot.index])
    
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    
    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                text = ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                               color='white' if val > pivot.values[~np.isnan(pivot.values)].mean() else 'black',
                               fontsize=9)
    
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(f'{metric.replace("_", " ").title()}', rotation=-90, va='bottom')
    
    ax.set_title(f'{metric.replace("_", " ").title()} by Policy and Workload')
    
    plt.tight_layout()
    filename = f'fig6_heatmap_{metric}'
    plt.savefig(os.path.join(outdir, f'{filename}.pdf'))
    plt.savefig(os.path.join(outdir, f'{filename}.png'))
    plt.close()
    print(f"Saved: {filename}.pdf/png")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Generate paper-quality DVFS analysis plots')
    parser.add_argument('--basedir', default='/Users/yash/Projects/wsc-noc-dvfs',
                        help='Base directory of the project')
    parser.add_argument('--outdir', default='plots/paper_figures',
                        help='Output directory for plots')
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    print("Loading results...")
    df, epoch_data = load_all_results(args.basedir)
    
    if df.empty:
        print("No results found!")
        return
    
    # Save aggregated data
    csv_path = os.path.join(args.outdir, 'aggregated_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved: aggregated_results.csv")
    print(f"Found {len(df)} runs across {df['policy'].nunique()} policies")
    
    print("\nGenerating figures...")
    
    # Figure 1: Policy comparison bars
    fig_policy_comparison(df, args.outdir)
    
    # Figure 2: Power-latency tradeoff
    fig_power_latency_tradeoff(df, args.outdir)
    
    # Figure 3: Power time-series (for each netrace workload)
    for wl in ['netrace_xapian', 'netrace_fft']:
        if wl in df['workload'].values:
            fig_power_timeseries(epoch_data, df, args.outdir, wl)
    
    # Figure 4: Latency breakdown (for synthetic workload)
    for wl in ['uniform_0.6', 'netrace_xapian']:
        if wl in df['workload'].values:
            fig_latency_breakdown(df, args.outdir, wl)
    
    # Figure 5: Headroom analysis
    fig_headroom_analysis(df, args.outdir)
    
    # Figure 6: Summary heatmaps
    fig_summary_heatmap(df, args.outdir, 'control_p99')
    fig_summary_heatmap(df, args.outdir, 'power_avg')
    
    print(f"\nAll figures saved to: {args.outdir}")
    print("\nRecommended additional plots for research paper:")
    print("  - Run more tests with tighter power caps (0.5-0.8W) to show throttling behavior")
    print("  - Sweep injection rates to show saturation behavior")
    print("  - Add CDF plots for latency distribution")


if __name__ == '__main__':
    main()
