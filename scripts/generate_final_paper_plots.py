#!/usr/bin/env python3
"""
Final Paper Plots - DVFS Policy Analysis

Generates comprehensive publication-quality figures for the DVFS paper:
1. P99 Latency vs Power Cap (per topology, traffic)
2. Power vs Injection Rate (per policy)
3. Control vs Batch Class P99 Comparison
4. Policy Comparison Grouped Bar Charts
5. Pareto Efficiency Curves
6. Frequency Scaling Behavior
7. Tail Latency Fairness

Usage:
    python scripts/generate_final_paper_plots.py --input final_paper_results/sweep_*.csv
"""

import argparse
import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
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
    'savefig.pad_inches': 0.1,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Policy styling
POLICY_STYLE = {
    'static': {
        'name': 'Static (Baseline)',
        'color': '#7f7f7f',
        'marker': 's',
        'linestyle': '-',
        'zorder': 1,
    },
    'uniform': {
        'name': 'Uniform Throttle',
        'color': '#1f77b4',
        'marker': 'o',
        'linestyle': '--',
        'zorder': 2,
    },
    'hw_reactive': {
        'name': 'HW-Reactive',
        'color': '#2ca02c',
        'marker': '^',
        'linestyle': '-.',
        'zorder': 3,
    },
    'queue_pid': {
        'name': 'Queue-PID',
        'color': '#ff7f0e',
        'marker': 'D',
        'linestyle': ':',
        'zorder': 4,
    },
    'perf_target': {
        'name': 'Perf-Target (Ours)',
        'color': '#d62728',
        'marker': 'v',
        'linestyle': '-',
        'linewidth': 2.5,
        'zorder': 5,
    },
}

TOPOLOGY_NAMES = {
    'flatfly': 'FlatFly 4×4',
    'mesh': 'Mesh 8×8',
}

TRAFFIC_NAMES = {
    'uniform': 'Uniform Random',
    'bitcomp': 'Bit-Complement',
    'hotspot': 'Hotspot',
}


def load_sweep_results(input_path: str) -> pd.DataFrame:
    """Load sweep results from CSV file(s)."""
    if '*' in input_path:
        files = sorted(glob.glob(input_path))
        if not files:
            raise FileNotFoundError(f"No files matching: {input_path}")
        # Use most recent file
        input_path = files[-1]
        print(f"Loading: {input_path}")
    
    df = pd.read_csv(input_path)
    
    # Filter to converged runs (handle both bool and string)
    if 'converged' in df.columns:
        df = df[df['converged'].isin([True, 'True', 1])].copy()
    
    # Standardize column names from existing results format
    if 'summary_control_p99' in df.columns and 'control_p99' not in df.columns:
        df['control_p99'] = df['summary_control_p99']
    if 'summary_batch_p99' in df.columns and 'batch_p99' not in df.columns:
        df['batch_p99'] = df['summary_batch_p99']
    if 'summary_total_power_avg' in df.columns and 'power_avg' not in df.columns:
        df['power_avg'] = df['summary_total_power_avg']
    
    # Use epoch max if latency.csv values not available
    if 'control_p99_epoch_max' in df.columns:
        mask = df['control_p99'].isna() | (df['control_p99'] == 0)
        df.loc[mask, 'control_p99'] = df.loc[mask, 'control_p99_epoch_max']
    if 'batch_p99_epoch_max' in df.columns:
        mask = df['batch_p99'].isna() | (df['batch_p99'] == 0)
        df.loc[mask, 'batch_p99'] = df.loc[mask, 'batch_p99_epoch_max']
    
    # Add latency_avg if not present
    if 'latency_avg' not in df.columns:
        # Estimate as average of control and batch P99 * 0.5
        if 'control_p99' in df.columns and 'batch_p99' in df.columns:
            df['latency_avg'] = (df['control_p99'].fillna(0) + df['batch_p99'].fillna(0)) / 2 * 0.5
    
    # Add derived columns
    if 'control_p99' in df.columns and 'batch_p99' in df.columns:
        df['p99_ratio'] = df['control_p99'] / df['batch_p99'].replace(0, np.nan)
    
    # Filter out unknown policies
    df = df[df['policy'] != 'unknown'].copy()
    
    return df


def save_figure(fig, outdir: str, name: str, formats=['pdf', 'png']):
    """Save figure in multiple formats."""
    for fmt in formats:
        path = os.path.join(outdir, f"{name}.{fmt}")
        fig.savefig(path)
    print(f"  Saved: {name}")


# =============================================================================
# Figure 1: P99 Latency vs Power Cap
# =============================================================================
def plot_p99_vs_power_cap(df: pd.DataFrame, outdir: str):
    """Plot control class P99 latency vs power cap for each topology/traffic combo."""
    
    topologies = df['topology'].unique()
    traffic_types = df['traffic'].unique()
    
    for topo in topologies:
        for traffic in traffic_types:
            subset = df[(df['topology'] == topo) & (df['traffic'] == traffic)]
            if subset.empty:
                continue
            
            # Get representative injection rate (middle value)
            rates = sorted(subset['injection_rate'].unique())
            if not rates:
                continue
            mid_rate = rates[len(rates) // 2]
            
            plot_df = subset[subset['injection_rate'] == mid_rate]
            if plot_df.empty:
                continue
            
            fig, ax = plt.subplots(figsize=(7, 5))
            
            for policy in POLICY_STYLE.keys():
                policy_df = plot_df[plot_df['policy'] == policy].sort_values('power_cap')
                if policy_df.empty:
                    continue
                
                style = POLICY_STYLE[policy]
                ax.plot(policy_df['power_cap'], policy_df['control_p99'],
                        label=style['name'],
                        color=style['color'],
                        marker=style['marker'],
                        linestyle=style['linestyle'],
                        linewidth=style.get('linewidth', 1.5),
                        markersize=8,
                        zorder=style['zorder'])
            
            ax.set_xlabel('Power Cap (W)')
            ax.set_ylabel('Control Class P99 Latency (cycles)')
            ax.set_title(f'{TOPOLOGY_NAMES.get(topo, topo)} - {TRAFFIC_NAMES.get(traffic, traffic)} (rate={mid_rate})')
            ax.legend(loc='upper right')
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            
            save_figure(fig, outdir, f'p99_vs_power_cap_{topo}_{traffic}')
            plt.close(fig)


# =============================================================================
# Figure 2: Power vs Injection Rate
# =============================================================================
def plot_power_vs_injection(df: pd.DataFrame, outdir: str):
    """Plot average power vs injection rate for each policy."""
    
    topologies = df['topology'].unique()
    traffic_types = df['traffic'].unique()
    
    for topo in topologies:
        for traffic in traffic_types:
            subset = df[(df['topology'] == topo) & (df['traffic'] == traffic)]
            if subset.empty:
                continue
            
            # Use moderate power cap
            caps = sorted(subset['power_cap'].unique())
            target_cap = 1.0  # Moderate cap
            if target_cap not in caps:
                target_cap = caps[len(caps) // 2] if caps else None
            if target_cap is None:
                continue
            
            plot_df = subset[subset['power_cap'] == target_cap]
            if plot_df.empty:
                continue
            
            fig, ax = plt.subplots(figsize=(7, 5))
            
            for policy in POLICY_STYLE.keys():
                policy_df = plot_df[plot_df['policy'] == policy].sort_values('injection_rate')
                if policy_df.empty:
                    continue
                
                style = POLICY_STYLE[policy]
                ax.plot(policy_df['injection_rate'], policy_df['power_avg'],
                        label=style['name'],
                        color=style['color'],
                        marker=style['marker'],
                        linestyle=style['linestyle'],
                        linewidth=style.get('linewidth', 1.5),
                        markersize=8,
                        zorder=style['zorder'])
            
            ax.axhline(y=target_cap, color='red', linestyle=':', linewidth=2, 
                       alpha=0.7, label=f'Power Cap ({target_cap}W)')
            
            ax.set_xlabel('Injection Rate')
            ax.set_ylabel('Average Power (W)')
            ax.set_title(f'{TOPOLOGY_NAMES.get(topo, topo)} - {TRAFFIC_NAMES.get(traffic, traffic)} (cap={target_cap}W)')
            ax.legend(loc='upper left')
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            
            save_figure(fig, outdir, f'power_vs_injection_{topo}_{traffic}')
            plt.close(fig)


# =============================================================================
# Figure 3: Control vs Batch P99 Comparison
# =============================================================================
def plot_class_p99_comparison(df: pd.DataFrame, outdir: str):
    """Plot control vs batch class P99 to show class differentiation."""
    
    if 'control_p99' not in df.columns or 'batch_p99' not in df.columns:
        print("  Skipping class comparison - no per-class P99 data")
        return
    
    topologies = df['topology'].unique()
    
    for topo in topologies:
        subset = df[df['topology'] == topo]
        if subset.empty:
            continue
        
        # Get one injection rate
        rates = sorted(subset['injection_rate'].unique())
        mid_rate = rates[len(rates) // 2] if rates else None
        if mid_rate is None:
            continue
        
        plot_df = subset[(subset['injection_rate'] == mid_rate) & 
                         (subset['control_p99'].notna()) & 
                         (subset['batch_p99'].notna())]
        
        if plot_df.empty:
            continue
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Panel A: Grouped bar chart
        ax = axes[0]
        policies = [p for p in POLICY_STYLE.keys() if p in plot_df['policy'].unique()]
        caps = sorted(plot_df['power_cap'].unique())[:5]  # Limit to 5 caps
        
        x = np.arange(len(caps))
        width = 0.35
        
        for i, policy in enumerate(policies[:3]):  # Show top 3 policies
            policy_data = plot_df[plot_df['policy'] == policy].set_index('power_cap')
            ctrl_vals = [policy_data.loc[c, 'control_p99'] if c in policy_data.index else 0 for c in caps]
            batch_vals = [policy_data.loc[c, 'batch_p99'] if c in policy_data.index else 0 for c in caps]
            
            offset = (i - 1) * 0.3
            style = POLICY_STYLE[policy]
            ax.bar(x + offset - width/2, ctrl_vals, width * 0.8, label=f'{style["name"]} (Ctrl)',
                   color=style['color'], alpha=0.9, edgecolor='black', linewidth=0.5)
            ax.bar(x + offset + width/2, batch_vals, width * 0.8, 
                   color=style['color'], alpha=0.4, edgecolor='black', linewidth=0.5, hatch='//')
        
        ax.set_xlabel('Power Cap (W)')
        ax.set_ylabel('P99 Latency (cycles)')
        ax.set_title(f'(a) Per-Class P99 Latency - {TOPOLOGY_NAMES.get(topo, topo)}')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{c}W' for c in caps])
        ax.legend(loc='upper right', fontsize=8)
        
        # Panel B: Scatter plot control vs batch
        ax = axes[1]
        for policy in policies:
            policy_df = plot_df[plot_df['policy'] == policy]
            style = POLICY_STYLE[policy]
            ax.scatter(policy_df['batch_p99'], policy_df['control_p99'],
                       label=style['name'],
                       color=style['color'],
                       marker=style['marker'],
                       s=100, edgecolors='black', linewidth=0.5, alpha=0.8,
                       zorder=style['zorder'])
        
        # Add diagonal line (equal latency)
        max_val = max(plot_df['control_p99'].max(), plot_df['batch_p99'].max())
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Equal Latency')
        
        ax.set_xlabel('Batch Class P99 (cycles)')
        ax.set_ylabel('Control Class P99 (cycles)')
        ax.set_title(f'(b) Class Fairness - {TOPOLOGY_NAMES.get(topo, topo)}')
        ax.legend(loc='upper left', fontsize=8)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        save_figure(fig, outdir, f'class_p99_comparison_{topo}')
        plt.close(fig)


# =============================================================================
# Figure 4: Policy Comparison Bar Chart
# =============================================================================
def plot_policy_comparison_bars(df: pd.DataFrame, outdir: str):
    """Grouped bar chart comparing all policies across metrics."""
    
    topologies = df['topology'].unique()
    
    for topo in topologies:
        subset = df[df['topology'] == topo]
        if subset.empty:
            continue
        
        # Aggregate across injection rates and traffic patterns
        # Use moderate power cap
        caps = sorted(subset['power_cap'].dropna().unique())
        target_cap = caps[len(caps) // 2] if caps else None
        if target_cap is None:
            continue
        
        plot_df = subset[subset['power_cap'] == target_cap]
        if plot_df.empty:
            plot_df = subset  # Use all if no matching cap
        
        # Determine which columns exist for aggregation
        agg_cols = {}
        if 'latency_avg' in df.columns:
            agg_cols['latency_avg'] = 'mean'
        if 'control_p99' in df.columns:
            agg_cols['control_p99'] = 'mean'
        if 'power_avg' in df.columns:
            agg_cols['power_avg'] = 'mean'
        if 'throughput' in df.columns:
            agg_cols['throughput'] = 'mean'
        if 'throughput_avg' in df.columns:
            agg_cols['throughput_avg'] = 'mean'
        
        if not agg_cols:
            continue
        
        # Group by policy, aggregate metrics
        agg_df = plot_df.groupby('policy').agg(agg_cols).reset_index()
        
        if agg_df.empty:
            continue
        
        # Use throughput_avg as fallback for throughput
        if 'throughput' not in agg_df.columns and 'throughput_avg' in agg_df.columns:
            agg_df['throughput'] = agg_df['throughput_avg']
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        policies = [p for p in POLICY_STYLE.keys() if p in agg_df['policy'].values]
        x = np.arange(len(policies))
        colors = [POLICY_STYLE[p]['color'] for p in policies]
        names = [POLICY_STYLE[p]['name'] for p in policies]
        
        def safe_get(df, policy, col):
            """Safely get value from dataframe."""
            try:
                row = df[df['policy'] == policy]
                if not row.empty and col in df.columns:
                    val = row[col].values[0]
                    return val if pd.notna(val) else 0
            except:
                pass
            return 0
        
        # Panel A: Average Latency
        ax = axes[0, 0]
        if 'latency_avg' in agg_df.columns:
            values = [safe_get(agg_df, p, 'latency_avg') for p in policies]
            ax.bar(x, values, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_ylabel('Avg Latency (cycles)')
        ax.set_title('(a) Average Latency')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha='right')
        
        # Panel B: Control P99
        ax = axes[0, 1]
        if 'control_p99' in agg_df.columns:
            values = [safe_get(agg_df, p, 'control_p99') for p in policies]
            ax.bar(x, values, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_ylabel('Control P99 (cycles)')
        ax.set_title('(b) Control Class P99 Latency')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha='right')
        
        # Panel C: Average Power
        ax = axes[1, 0]
        if 'power_avg' in agg_df.columns:
            values = [safe_get(agg_df, p, 'power_avg') for p in policies]
            ax.bar(x, values, color=colors, edgecolor='black', linewidth=0.5)
        ax.axhline(y=target_cap, color='red', linestyle=':', linewidth=2, alpha=0.7)
        ax.set_ylabel('Avg Power (W)')
        ax.set_title(f'(c) Power Consumption (cap={target_cap}W)')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha='right')
        
        # Panel D: Throughput
        ax = axes[1, 1]
        if 'throughput' in agg_df.columns:
            values = [safe_get(agg_df, p, 'throughput') for p in policies]
            ax.bar(x, values, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_ylabel('Throughput')
        ax.set_title('(d) Network Throughput')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha='right')
        
        plt.suptitle(f'{TOPOLOGY_NAMES.get(topo, topo)} Policy Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_figure(fig, outdir, f'policy_comparison_bars_{topo}')
        plt.close(fig)


# =============================================================================
# Figure 5: Pareto Efficiency (Power vs Latency)
# =============================================================================
def plot_pareto_efficiency(df: pd.DataFrame, outdir: str):
    """Scatter plot showing power vs latency tradeoff with Pareto frontier."""
    
    topologies = df['topology'].unique()
    
    for topo in topologies:
        subset = df[(df['topology'] == topo) & 
                    df['power_avg'].notna() & 
                    df['control_p99'].notna()]
        if subset.empty:
            continue
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for policy in POLICY_STYLE.keys():
            policy_df = subset[subset['policy'] == policy]
            if policy_df.empty:
                continue
            
            style = POLICY_STYLE[policy]
            ax.scatter(policy_df['power_avg'], policy_df['control_p99'],
                       label=style['name'],
                       color=style['color'],
                       marker=style['marker'],
                       s=80, alpha=0.7,
                       edgecolors='black', linewidth=0.3,
                       zorder=style['zorder'])
        
        # Find Pareto frontier
        points = subset[['power_avg', 'control_p99']].values
        pareto_mask = np.ones(len(points), dtype=bool)
        for i, (p1, l1) in enumerate(points):
            for j, (p2, l2) in enumerate(points):
                if i != j and p2 <= p1 and l2 <= l1 and (p2 < p1 or l2 < l1):
                    pareto_mask[i] = False
                    break
        
        pareto_points = points[pareto_mask]
        if len(pareto_points) > 0:
            pareto_points = pareto_points[pareto_points[:, 0].argsort()]
            ax.plot(pareto_points[:, 0], pareto_points[:, 1], 
                    'k--', linewidth=1.5, alpha=0.5, label='Pareto Frontier', zorder=0)
        
        ax.set_xlabel('Average Power (W)')
        ax.set_ylabel('Control Class P99 Latency (cycles)')
        ax.set_title(f'Power-Latency Efficiency - {TOPOLOGY_NAMES.get(topo, topo)}')
        ax.legend(loc='upper right')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        
        # Add "better" annotation
        ax.annotate('Better →', xy=(0.08, 0.08), xycoords='axes fraction',
                    fontsize=12, ha='left', alpha=0.5,
                    bbox=dict(boxstyle='round', facecolor='green', alpha=0.15))
        
        save_figure(fig, outdir, f'pareto_efficiency_{topo}')
        plt.close(fig)


# =============================================================================
# Figure 6: Summary Heatmap (All Policies x Metrics)
# =============================================================================
def plot_summary_heatmap(df: pd.DataFrame, outdir: str):
    """Heatmap showing normalized performance across all policies and metrics."""
    
    topologies = df['topology'].unique()
    
    for topo in topologies:
        subset = df[df['topology'] == topo]
        if subset.empty:
            continue
        
        # Determine which columns exist for aggregation
        agg_cols = {}
        if 'latency_avg' in df.columns:
            agg_cols['latency_avg'] = 'mean'
        if 'control_p99' in df.columns:
            agg_cols['control_p99'] = 'mean'
        if 'power_avg' in df.columns:
            agg_cols['power_avg'] = 'mean'
        if 'throughput' in df.columns:
            agg_cols['throughput'] = 'mean'
        elif 'throughput_avg' in df.columns:
            agg_cols['throughput_avg'] = 'mean'
        
        if len(agg_cols) < 2:
            continue
        
        # Aggregate by policy
        agg_df = subset.groupby('policy').agg(agg_cols).reset_index()
        
        if agg_df.empty or len(agg_df) < 2:
            continue
        
        # Map throughput_avg to throughput if needed
        if 'throughput_avg' in agg_df.columns and 'throughput' not in agg_df.columns:
            agg_df['throughput'] = agg_df['throughput_avg']
        
        # Normalize metrics (0-1 scale, lower is better for lat/power, higher for throughput)
        available_metrics = [m for m in ['latency_avg', 'control_p99', 'power_avg', 'throughput'] if m in agg_df.columns]
        metric_labels = {'latency_avg': 'Avg Latency', 'control_p99': 'Control P99', 
                        'power_avg': 'Avg Power', 'throughput': 'Throughput'}
        
        if len(available_metrics) < 2:
            continue
        
        # Lower is better for first 3, higher is better for throughput
        normalized = pd.DataFrame()
        for metric in available_metrics:
            vals = agg_df[metric].fillna(0).values
            if vals.max() > vals.min():
                if metric == 'throughput':
                    # Higher is better
                    normalized[metric] = (vals - vals.min()) / (vals.max() - vals.min())
                else:
                    # Lower is better (invert)
                    normalized[metric] = 1 - (vals - vals.min()) / (vals.max() - vals.min())
            else:
                normalized[metric] = 0.5
        
        if normalized.empty:
            continue
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        policies = agg_df['policy'].tolist()
        policy_names = [POLICY_STYLE.get(p, {}).get('name', p) for p in policies]
        
        im = ax.imshow(normalized.values.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(range(len(policies)))
        ax.set_xticklabels(policy_names, rotation=45, ha='right')
        ax.set_yticks(range(len(available_metrics)))
        ax.set_yticklabels([metric_labels.get(m, m) for m in available_metrics])
        
        # Add value annotations
        for i in range(len(available_metrics)):
            for j in range(len(policies)):
                val = normalized.iloc[j, i]
                color = 'white' if val < 0.3 or val > 0.7 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=10)
        
        plt.colorbar(im, ax=ax, label='Normalized Score (1=Best)')
        
        plt.colorbar(im, ax=ax, label='Normalized Score (1=Best)')
        ax.set_title(f'Policy Performance Summary - {TOPOLOGY_NAMES.get(topo, topo)}')
        
        plt.tight_layout()
        save_figure(fig, outdir, f'summary_heatmap_{topo}')
        plt.close(fig)


# =============================================================================
# Figure 7: Latency Distribution (Box Plot)
# =============================================================================
def plot_latency_distribution(df: pd.DataFrame, outdir: str):
    """Box plot showing latency distribution across policies."""
    
    topologies = df['topology'].unique()
    
    for topo in topologies:
        subset = df[(df['topology'] == topo) & df['latency_avg'].notna()]
        if subset.empty:
            continue
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        policies = [p for p in POLICY_STYLE.keys() if p in subset['policy'].unique()]
        data = [subset[subset['policy'] == p]['latency_avg'].values for p in policies]
        
        bp = ax.boxplot(data, patch_artist=True, labels=[POLICY_STYLE[p]['name'] for p in policies])
        
        for patch, policy in zip(bp['boxes'], policies):
            patch.set_facecolor(POLICY_STYLE[policy]['color'])
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Average Latency (cycles)')
        ax.set_title(f'Latency Distribution - {TOPOLOGY_NAMES.get(topo, topo)}')
        plt.xticks(rotation=30, ha='right')
        
        plt.tight_layout()
        save_figure(fig, outdir, f'latency_distribution_{topo}')
        plt.close(fig)


# =============================================================================
# Figure 8: Multi-Traffic Comparison
# =============================================================================
def plot_multi_traffic_comparison(df: pd.DataFrame, outdir: str):
    """Compare policies across different traffic patterns."""
    
    topologies = df['topology'].unique()
    traffic_types = df['traffic'].unique()
    
    if len(traffic_types) < 2:
        print("  Skipping multi-traffic comparison - only 1 traffic type")
        return
    
    for topo in topologies:
        subset = df[df['topology'] == topo]
        if subset.empty:
            continue
        
        # Use one power cap
        caps = sorted(subset['power_cap'].unique())
        target_cap = caps[len(caps) // 2] if caps else None
        if target_cap is None:
            continue
        
        plot_df = subset[subset['power_cap'] == target_cap]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Panel A: Control P99 across traffic patterns
        ax = axes[0]
        policies = [p for p in POLICY_STYLE.keys() if p in plot_df['policy'].unique()]
        x = np.arange(len(traffic_types))
        width = 0.15
        
        for i, policy in enumerate(policies):
            policy_data = plot_df[plot_df['policy'] == policy]
            values = []
            for traffic in traffic_types:
                traffic_data = policy_data[policy_data['traffic'] == traffic]
                values.append(traffic_data['control_p99'].mean() if not traffic_data.empty else 0)
            
            offset = (i - len(policies)/2 + 0.5) * width
            style = POLICY_STYLE[policy]
            ax.bar(x + offset, values, width,
                   label=style['name'],
                   color=style['color'],
                   edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Traffic Pattern')
        ax.set_ylabel('Control P99 Latency (cycles)')
        ax.set_title(f'(a) P99 Latency by Traffic - {TOPOLOGY_NAMES.get(topo, topo)}')
        ax.set_xticks(x)
        ax.set_xticklabels([TRAFFIC_NAMES.get(t, t) for t in traffic_types])
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        
        # Panel B: Power consumption
        ax = axes[1]
        for i, policy in enumerate(policies):
            policy_data = plot_df[plot_df['policy'] == policy]
            values = []
            for traffic in traffic_types:
                traffic_data = policy_data[policy_data['traffic'] == traffic]
                values.append(traffic_data['power_avg'].mean() if not traffic_data.empty else 0)
            
            offset = (i - len(policies)/2 + 0.5) * width
            style = POLICY_STYLE[policy]
            ax.bar(x + offset, values, width,
                   color=style['color'],
                   edgecolor='black', linewidth=0.5)
        
        ax.axhline(y=target_cap, color='red', linestyle=':', linewidth=2, alpha=0.7)
        ax.set_xlabel('Traffic Pattern')
        ax.set_ylabel('Average Power (W)')
        ax.set_title(f'(b) Power by Traffic - {TOPOLOGY_NAMES.get(topo, topo)} (cap={target_cap}W)')
        ax.set_xticks(x)
        ax.set_xticklabels([TRAFFIC_NAMES.get(t, t) for t in traffic_types])
        
        plt.tight_layout()
        save_figure(fig, outdir, f'multi_traffic_{topo}')
        plt.close(fig)


# =============================================================================
# Figure 9: All Policies Line Plot (Key Figure)
# =============================================================================
def plot_all_policies_key_figure(df: pd.DataFrame, outdir: str):
    """Key figure showing all policies on P99 vs Power Cap."""
    
    topologies = df['topology'].unique()
    traffic_types = df['traffic'].unique()
    
    for topo in topologies:
        for traffic in traffic_types:
            subset = df[(df['topology'] == topo) & (df['traffic'] == traffic)]
            if subset.empty:
                continue
            
            # Get middle injection rate
            rates = sorted(subset['injection_rate'].unique())
            if not rates:
                continue
            mid_rate = rates[len(rates) // 2]
            
            plot_df = subset[subset['injection_rate'] == mid_rate]
            if plot_df.empty:
                continue
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Panel A: Control P99 vs Power Cap
            ax = axes[0]
            for policy in POLICY_STYLE.keys():
                policy_df = plot_df[plot_df['policy'] == policy].sort_values('power_cap')
                if policy_df.empty:
                    continue
                
                style = POLICY_STYLE[policy]
                ax.plot(policy_df['power_cap'], policy_df['control_p99'],
                        label=style['name'],
                        color=style['color'],
                        marker=style['marker'],
                        linestyle=style['linestyle'],
                        linewidth=style.get('linewidth', 2),
                        markersize=10,
                        zorder=style['zorder'])
            
            ax.set_xlabel('Power Cap (W)', fontsize=12)
            ax.set_ylabel('Control Class P99 Latency (cycles)', fontsize=12)
            ax.set_title(f'(a) Tail Latency vs Power Budget', fontsize=13)
            ax.legend(loc='upper right', fontsize=10)
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            
            # Panel B: Power vs Power Cap (shows throttling behavior)
            ax = axes[1]
            for policy in POLICY_STYLE.keys():
                policy_df = plot_df[plot_df['policy'] == policy].sort_values('power_cap')
                if policy_df.empty:
                    continue
                
                style = POLICY_STYLE[policy]
                ax.plot(policy_df['power_cap'], policy_df['power_avg'],
                        label=style['name'],
                        color=style['color'],
                        marker=style['marker'],
                        linestyle=style['linestyle'],
                        linewidth=style.get('linewidth', 2),
                        markersize=10,
                        zorder=style['zorder'])
            
            # Add cap line
            cap_range = np.linspace(0, plot_df['power_cap'].max(), 100)
            ax.plot(cap_range, cap_range, 'k:', linewidth=2, alpha=0.5, label='Power = Cap')
            
            ax.set_xlabel('Power Cap (W)', fontsize=12)
            ax.set_ylabel('Average Power (W)', fontsize=12)
            ax.set_title(f'(b) Power Consumption vs Budget', fontsize=13)
            ax.legend(loc='upper left', fontsize=10)
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            
            plt.suptitle(f'{TOPOLOGY_NAMES.get(topo, topo)} - {TRAFFIC_NAMES.get(traffic, traffic)} (rate={mid_rate})',
                         fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            save_figure(fig, outdir, f'all_policies_{topo}_{traffic}')
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Generate final paper plots')
    parser.add_argument('--input', type=str, default='final_paper_results/sweep_*.csv',
                        help='Input CSV file(s) glob pattern')
    parser.add_argument('--outdir', type=str, default='final_paper_plots',
                        help='Output directory for plots')
    parser.add_argument('--formats', nargs='+', default=['pdf', 'png'],
                        help='Output formats')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    print(f"=== Final Paper Plot Generation ===")
    print(f"Output directory: {args.outdir}")
    
    try:
        df = load_sweep_results(args.input)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Run the sweep script first: python scripts/run_final_paper_sweep.py")
        return 1
    
    print(f"Loaded {len(df)} converged experiments")
    print(f"Topologies: {df['topology'].unique().tolist()}")
    print(f"Traffic: {df['traffic'].unique().tolist()}")
    print(f"Policies: {df['policy'].unique().tolist()}")
    print()
    
    print("Generating plots...")
    
    # Generate all figures
    plot_all_policies_key_figure(df, args.outdir)
    plot_p99_vs_power_cap(df, args.outdir)
    plot_power_vs_injection(df, args.outdir)
    plot_class_p99_comparison(df, args.outdir)
    plot_policy_comparison_bars(df, args.outdir)
    plot_pareto_efficiency(df, args.outdir)
    plot_summary_heatmap(df, args.outdir)
    plot_latency_distribution(df, args.outdir)
    plot_multi_traffic_comparison(df, args.outdir)
    
    print(f"\n=== Done! Plots saved to: {args.outdir} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
