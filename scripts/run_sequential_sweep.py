#!/usr/bin/env python3
"""
Sequential DVFS Sweep - More Reliable Version

Runs experiments sequentially for better reliability.
Generates results directly into the stress_tests folders.
"""

import subprocess
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
BOOKSIM_PATH = PROJECT_DIR / "booksim2" / "src" / "booksim"
STRESS_CONFIGS_DIR = PROJECT_DIR / "booksim2" / "flatfly_stress_configs"
MESH_CONFIGS_DIR = PROJECT_DIR / "booksim2" / "mesh_stress_configs"

def run_config(config_path: Path, timeout: int = 300) -> bool:
    """Run a single booksim config file."""
    print(f"  Running: {config_path.name}...")
    
    try:
        result = subprocess.run(
            [str(BOOKSIM_PATH), str(config_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(PROJECT_DIR / "booksim2" / "src"),
        )
        
        if result.returncode == 0:
            print(f"    ✓ Success")
            return True
        else:
            print(f"    ✗ Failed (exit={result.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"    ✗ Timeout after {timeout}s")
        return False
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False


def run_all_configs(config_dir: Path, timeout: int = 300):
    """Run all config files in a stress configs directory."""
    print(f"\n=== Running configs from: {config_dir.name} ===")
    
    results = {'success': 0, 'failed': 0}
    
    for policy_dir in sorted(config_dir.iterdir()):
        if not policy_dir.is_dir():
            continue
        
        print(f"\n--- Policy: {policy_dir.name} ---")
        
        for config_file in sorted(policy_dir.glob("*.cfg")):
            success = run_config(config_file, timeout)
            if success:
                results['success'] += 1
            else:
                results['failed'] += 1
    
    return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Sequential DVFS sweep runner')
    parser.add_argument('--topology', choices=['flatfly', 'mesh', 'both'], 
                        default='flatfly', help='Topology to run')
    parser.add_argument('--timeout', type=int, default=300,
                        help='Timeout per experiment (seconds)')
    parser.add_argument('--policy', type=str, default=None,
                        help='Only run specific policy (e.g., hw_reactive)')
    args = parser.parse_args()
    
    start_time = time.time()
    total_results = {'success': 0, 'failed': 0}
    
    if args.topology in ['flatfly', 'both']:
        if STRESS_CONFIGS_DIR.exists():
            if args.policy:
                # Run specific policy only
                policy_dir = STRESS_CONFIGS_DIR / args.policy
                if policy_dir.exists():
                    print(f"\n=== Running {args.policy} configs (flatfly) ===")
                    for config_file in sorted(policy_dir.glob("*.cfg")):
                        success = run_config(config_file, args.timeout)
                        if success:
                            total_results['success'] += 1
                        else:
                            total_results['failed'] += 1
            else:
                results = run_all_configs(STRESS_CONFIGS_DIR, args.timeout)
                total_results['success'] += results['success']
                total_results['failed'] += results['failed']
    
    if args.topology in ['mesh', 'both']:
        if MESH_CONFIGS_DIR.exists():
            if args.policy:
                policy_dir = MESH_CONFIGS_DIR / args.policy
                if policy_dir.exists():
                    print(f"\n=== Running {args.policy} configs (mesh) ===")
                    for config_file in sorted(policy_dir.glob("*.cfg")):
                        success = run_config(config_file, args.timeout)
                        if success:
                            total_results['success'] += 1
                        else:
                            total_results['failed'] += 1
            else:
                results = run_all_configs(MESH_CONFIGS_DIR, args.timeout)
                total_results['success'] += results['success']
                total_results['failed'] += results['failed']
    
    elapsed = time.time() - start_time
    print(f"\n=== Complete ===")
    print(f"Success: {total_results['success']}")
    print(f"Failed: {total_results['failed']}")
    print(f"Time: {elapsed:.1f}s")
    
    return 0 if total_results['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
