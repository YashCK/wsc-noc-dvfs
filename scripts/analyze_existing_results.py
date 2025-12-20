#!/usr/bin/env python3
"""
Analyze Existing Stress Test Results

Parses existing booksim simulation outputs to create a unified CSV for plotting.
Works with the existing stress_tests folder structure.
"""

import os
import sys
import csv
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
SIMS_DIR = PROJECT_DIR / "booksim2" / "sims"
OUTPUT_DIR = PROJECT_DIR / "final_paper_results"


def parse_run_name(run_name: str) -> Dict[str, any]:
    """Parse run name to extract policy, workload, traffic, power cap."""
    result = {
        'run_name': run_name,
        'policy': 'unknown',
        'traffic': 'unknown',
        'workload': run_name,
        'power_cap': None,
        'injection_rate': None,
    }
    
    # Extract power cap (e.g., _10W or _0.7W)
    cap_match = re.search(r'_([\d.]+)W$', run_name)
    if cap_match:
        result['power_cap'] = float(cap_match.group(1))
    
    # Extract policy
    if run_name.startswith('hw_reactive'):
        result['policy'] = 'hw_reactive'
        rest = run_name[len('hw_reactive_'):]
    elif run_name.startswith('queue_pid'):
        result['policy'] = 'queue_pid'
        rest = run_name[len('queue_pid_'):]
    elif run_name.startswith('perf_target'):
        result['policy'] = 'perf_target'
        rest = run_name[len('perf_target_'):]
    elif run_name.startswith('static'):
        result['policy'] = 'static'
        rest = run_name[len('static_'):]
    elif run_name.startswith('uniform'):
        result['policy'] = 'uniform'
        rest = run_name[len('uniform_'):]
    else:
        rest = run_name
    
    # Extract traffic pattern
    if 'netrace_fft' in rest:
        result['traffic'] = 'netrace_fft'
    elif 'netrace_xapian' in rest:
        result['traffic'] = 'netrace_xapian'
    elif 'netrace_merged' in rest:
        result['traffic'] = 'netrace_merged'
    elif 'uniform' in rest:
        result['traffic'] = 'uniform'
        # Extract injection rate
        rate_match = re.search(r'uniform_([\d.]+)', rest)
        if rate_match:
            result['injection_rate'] = float(rate_match.group(1))
    elif 'bitcomp' in rest:
        result['traffic'] = 'bitcomp'
        rate_match = re.search(r'bitcomp_([\d.]+)', rest)
        if rate_match:
            result['injection_rate'] = float(rate_match.group(1))
    elif 'hotspot' in rest:
        result['traffic'] = 'hotspot'
        rate_match = re.search(r'hotspot_([\d.]+)', rest)
        if rate_match:
            result['injection_rate'] = float(rate_match.group(1))
    
    return result


def load_run_results(run_dir: Path) -> Dict[str, any]:
    """Load results from a single run directory."""
    result = {}
    
    # Load summary.csv if exists
    summary_path = run_dir / 'summary.csv'
    if summary_path.exists():
        try:
            df = pd.read_csv(summary_path)
            if not df.empty:
                for col in df.columns:
                    result[f'summary_{col}'] = df[col].iloc[0]
        except Exception as e:
            print(f"  Warning: Failed to load {summary_path}: {e}")
    
    # Load latency.csv for per-class P99
    latency_path = run_dir / 'latency.csv'
    if latency_path.exists():
        try:
            df = pd.read_csv(latency_path)
            for _, row in df.iterrows():
                cls = str(row.get('class', ''))
                if cls == '0':
                    result['control_p50'] = row.get('plat_p50') or row.get('nlat_p50')
                    result['control_p95'] = row.get('plat_p95') or row.get('nlat_p95')
                    result['control_p99'] = row.get('plat_p99') or row.get('nlat_p99')
                elif cls == '1':
                    result['batch_p50'] = row.get('plat_p50') or row.get('nlat_p50')
                    result['batch_p95'] = row.get('plat_p95') or row.get('nlat_p95')
                    result['batch_p99'] = row.get('plat_p99') or row.get('nlat_p99')
        except Exception as e:
            print(f"  Warning: Failed to load {latency_path}: {e}")
    
    # Load epoch.csv for time-series stats
    epoch_path = run_dir / 'epoch.csv'
    if epoch_path.exists():
        try:
            df = pd.read_csv(epoch_path)
            if not df.empty:
                if 'total_power' in df.columns:
                    result['power_avg'] = df['total_power'].mean()
                    result['power_peak'] = df['total_power'].max()
                    result['power_std'] = df['total_power'].std()
                
                if 'headroom' in df.columns:
                    result['headroom_avg'] = df['headroom'].mean()
                    result['headroom_min'] = df['headroom'].min()
                
                if 'class0_p99' in df.columns:
                    valid = df['class0_p99'][df['class0_p99'] > 0]
                    if len(valid) > 0:
                        result['control_p99_epoch_avg'] = valid.mean()
                        result['control_p99_epoch_max'] = valid.max()
                
                if 'class1_p99' in df.columns:
                    valid = df['class1_p99'][df['class1_p99'] > 0]
                    if len(valid) > 0:
                        result['batch_p99_epoch_avg'] = valid.mean()
                        result['batch_p99_epoch_max'] = valid.max()
                
                if 'class0_throughput' in df.columns:
                    valid = df['class0_throughput'][df['class0_throughput'] > 0]
                    if len(valid) > 0:
                        result['throughput_avg'] = valid.mean()
                
                result['num_epochs'] = len(df)
        except Exception as e:
            print(f"  Warning: Failed to load {epoch_path}: {e}")
    
    # Load throughput.csv
    throughput_path = run_dir / 'throughput.csv'
    if throughput_path.exists():
        try:
            df = pd.read_csv(throughput_path)
            if not df.empty and 'accepted_rate' in df.columns:
                result['throughput'] = df['accepted_rate'].iloc[-1]
        except:
            pass
    
    return result


def analyze_stress_tests(base_dir: Path, topology: str = 'flatfly') -> pd.DataFrame:
    """Analyze all stress test results in a directory."""
    results = []
    
    for run_dir in sorted(base_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        
        run_name = run_dir.name
        print(f"  Processing: {run_name}")
        
        # Parse run name for metadata
        metadata = parse_run_name(run_name)
        metadata['topology'] = topology
        
        # Load results
        run_results = load_run_results(run_dir)
        metadata.update(run_results)
        
        # Mark as converged if we have data
        metadata['converged'] = bool(run_results)
        
        results.append(metadata)
    
    return pd.DataFrame(results)


def main():
    """Main entry point."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    all_results = []
    
    # Analyze stress_tests folder (flatfly)
    stress_tests_dir = SIMS_DIR / 'stress_tests'
    if stress_tests_dir.exists():
        print(f"\nAnalyzing: {stress_tests_dir}")
        df = analyze_stress_tests(stress_tests_dir, topology='flatfly')
        all_results.append(df)
        print(f"  Found {len(df)} runs")
    
    # Analyze flatfly_stress_tests folder
    flatfly_stress_dir = SIMS_DIR / 'flatfly_stress_tests'
    if flatfly_stress_dir.exists():
        print(f"\nAnalyzing: {flatfly_stress_dir}")
        df = analyze_stress_tests(flatfly_stress_dir, topology='flatfly')
        all_results.append(df)
        print(f"  Found {len(df)} runs")
    
    # Analyze mesh_stress_tests folder
    mesh_stress_dir = SIMS_DIR / 'mesh_stress_tests'
    if mesh_stress_dir.exists():
        print(f"\nAnalyzing: {mesh_stress_dir}")
        df = analyze_stress_tests(mesh_stress_dir, topology='mesh')
        all_results.append(df)
        print(f"  Found {len(df)} runs")
    
    # Analyze class_test folder
    class_test_dir = SIMS_DIR / 'class_test'
    if class_test_dir.exists():
        print(f"\nAnalyzing: {class_test_dir}")
        df = analyze_stress_tests(class_test_dir, topology='flatfly')
        all_results.append(df)
        print(f"  Found {len(df)} runs")
    
    if not all_results:
        print("No results found!")
        return 1
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Clean up and standardize
    # Use control_p99 from latency.csv or epoch_max as fallback
    if 'control_p99' not in combined_df.columns:
        combined_df['control_p99'] = np.nan
    
    # Fill missing control_p99 from epoch data or summary
    mask = combined_df['control_p99'].isna()
    if 'control_p99_epoch_max' in combined_df.columns:
        combined_df.loc[mask, 'control_p99'] = combined_df.loc[mask, 'control_p99_epoch_max']
    if 'summary_control_p99' in combined_df.columns:
        mask = combined_df['control_p99'].isna()
        combined_df.loc[mask, 'control_p99'] = combined_df.loc[mask, 'summary_control_p99']
    
    # Same for batch_p99
    if 'batch_p99' not in combined_df.columns:
        combined_df['batch_p99'] = np.nan
    
    mask = combined_df['batch_p99'].isna()
    if 'batch_p99_epoch_max' in combined_df.columns:
        combined_df.loc[mask, 'batch_p99'] = combined_df.loc[mask, 'batch_p99_epoch_max']
    if 'summary_batch_p99' in combined_df.columns:
        mask = combined_df['batch_p99'].isna()
        combined_df.loc[mask, 'batch_p99'] = combined_df.loc[mask, 'summary_batch_p99']
    
    # Save results
    output_file = OUTPUT_DIR / 'existing_results.csv'
    combined_df.to_csv(output_file, index=False)
    print(f"\n=== Saved {len(combined_df)} runs to: {output_file} ===")
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Topologies: {combined_df['topology'].unique().tolist()}")
    print(f"Policies: {combined_df['policy'].unique().tolist()}")
    print(f"Traffic patterns: {combined_df['traffic'].unique().tolist()}")
    
    print("\n=== Per-Policy Summary ===")
    for policy in sorted(combined_df['policy'].unique()):
        policy_df = combined_df[combined_df['policy'] == policy]
        ctrl_p99 = policy_df['control_p99'].dropna()
        power = policy_df['power_avg'].dropna()
        print(f"  {policy:15} runs={len(policy_df):2d}  "
              f"ctrl_p99={ctrl_p99.mean():7.1f} (n={len(ctrl_p99)})  "
              f"power={power.mean():.3f}W (n={len(power)})")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
