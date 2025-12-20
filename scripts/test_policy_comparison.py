#!/usr/bin/env python3
"""
Test script to compare DVFS policies for priority-aware NoC control.

Evaluates: uniform throttling < hw_reactive < queue_pid < perf_target (optimal)
for P99 latency of control class packets under a power cap.
"""

import subprocess
import re
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional
import json

# Path configuration
BOOKSIM_PATH = os.path.join(os.path.dirname(__file__), "..", "booksim2", "src", "booksim")
RUNFILES_PATH = os.path.join(os.path.dirname(__file__), "..", "booksim2", "runfiles")

@dataclass
class SimResult:
    """Results from a booksim simulation"""
    policy: str
    traffic: str
    power_cap: float
    avg_latency: float
    p99_latency_class0: float  # Control class
    p99_latency_class1: float  # Batch class
    avg_power: float
    throughput: float
    converged: bool
    raw_output: str = ""


def run_booksim(config_file: str, overrides: Dict[str, str] = None) -> Optional[SimResult]:
    """Run booksim with given config and return parsed results."""
    
    cmd = [BOOKSIM_PATH, config_file]
    if overrides:
        for key, val in overrides.items():
            cmd.append(f"{key}={val}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=os.path.dirname(BOOKSIM_PATH)
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: {config_file}")
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None
    
    # Parse results from output
    return parse_booksim_output(output, overrides)


def parse_booksim_output(output: str, overrides: Dict = None) -> Optional[SimResult]:
    """Parse booksim output to extract key metrics."""
    
    # Check for convergence
    converged = "Simulation unstable" not in output and "converging" not in output.lower()
    
    # Extract policy type
    policy_match = re.search(r'dvfs_policy\s*=\s*(\w+)', output)
    policy = policy_match.group(1) if policy_match else (overrides.get("dvfs_policy", "unknown") if overrides else "unknown")
    
    # Extract traffic type
    traffic_match = re.search(r'traffic\s*=\s*(\w+)', output)
    traffic = traffic_match.group(1) if traffic_match else "unknown"
    
    # Extract power cap
    power_cap = float(overrides.get("power_cap", "0")) if overrides else 0.0
    
    # Extract average latency (overall)
    avg_lat_match = re.search(r'Overall average latency\s*=\s*([\d.]+)', output)
    avg_latency = float(avg_lat_match.group(1)) if avg_lat_match else 0.0
    
    # Extract per-class P99 latencies
    # Look for class 0 (control) P99
    p99_class0 = 0.0
    p99_class1 = 0.0
    
    # Try to find per-class stats
    class0_p99_match = re.search(r'Class 0.*?P99[:\s]*([\d.]+)', output, re.DOTALL)
    if class0_p99_match:
        p99_class0 = float(class0_p99_match.group(1))
    
    class1_p99_match = re.search(r'Class 1.*?P99[:\s]*([\d.]+)', output, re.DOTALL)
    if class1_p99_match:
        p99_class1 = float(class1_p99_match.group(1))
    
    # Alternative: look for packet latency stats
    if p99_class0 == 0.0:
        # Look for overall stats format
        overall_p99 = re.search(r'Packet latency.*?99th percentile\s*=\s*([\d.]+)', output, re.DOTALL)
        if overall_p99:
            p99_class0 = float(overall_p99.group(1))
    
    # Look for "Class 0 Packet latency P99" or similar
    class_patterns = [
        r'Class\s*0\s*Packet\s+latency.*?P99[:\s=]*([\d.]+)',
        r'class\[0\].*?p99[:\s=]*([\d.]+)',
        r'Class 0.*?99th percentile\s*=\s*([\d.]+)',
        r'Packet latency P99:\s*([\d.]+)',
        r'network latency p99.*?class\s*0.*?:\s*([\d.]+)'
    ]
    for pattern in class_patterns:
        match = re.search(pattern, output, re.IGNORECASE | re.DOTALL)
        if match:
            p99_class0 = float(match.group(1))
            break
    
    # Extract average power
    power_match = re.search(r'Total Power:\s*([\d.]+)|total_power\s*=\s*([\d.]+)', output, re.IGNORECASE)
    avg_power = float(power_match.group(1) or power_match.group(2)) if power_match else 0.0
    
    # Extract throughput
    throughput_match = re.search(r'Overall.*?throughput\s*=\s*([\d.]+)', output, re.IGNORECASE)
    throughput = float(throughput_match.group(1)) if throughput_match else 0.0
    
    # Look for DVFS policy output lines for class latency
    dvfs_matches = re.findall(r'(?:PERF_TARGET|QUEUE_PID|HW_REACTIVE).*?ctrl_lat[:\s=]*([\d.]+)', output)
    if dvfs_matches and p99_class0 == 0.0:
        p99_class0 = float(dvfs_matches[-1])  # Use last reported value
    
    return SimResult(
        policy=policy,
        traffic=traffic,
        power_cap=power_cap,
        avg_latency=avg_latency,
        p99_latency_class0=p99_class0,
        p99_latency_class1=p99_class1,
        avg_power=avg_power,
        throughput=throughput,
        converged=converged,
        raw_output=output
    )


def create_test_config() -> str:
    """Create a temporary test config with 2 classes."""
    config = """
// Test config for policy comparison
// 2-class traffic: class 0 = control (high priority), class 1 = batch (low priority)

sim_type = latency;
topology = flatfly;
routing_function = ran_min;
c = 4;
k = 4;
n = 2;
x = 4;
y = 4;
xr = 2;
yr = 2;
subnets = 1;
num_vcs = 16;
vc_buf_size = 8;
vc_allocator = max_size;
sw_allocator = max_size;
alloc_iters = 2;
credit_delay = 2;
routing_delay = 0;
vc_alloc_delay = 1;
sw_alloc_delay = 1;
st_final_delay = 1;
input_speedup = 1;
output_speedup = 1;
internal_speedup = 1.0;
warmup_periods = 3;
sim_count = 1;
sample_period = 10000;
max_samples = 10;
traffic = uniform;
packet_size = 1;
injection_rate = 0.5;
latency_thres = 5000;

// 2 class configuration - control (0) and batch (1)
classes = 2;
class_priority = {1, 0};
class_slo = {50, -1};
class_priority_boost = {1.0, 1.0};
priority_policy = static_class;

// Control class gets ~30% of traffic, batch ~70%
injection_rate = {0.15, 0.35};

use_orion = 1;
Vdd = 1.0;
Orion_tr = 0.2;
Orion_Freq = 1e9;
Orion_inport = 5;
Orion_outport = 5;
Orion_bitwidth = 128;
Orion_vc_class = 1;
Orion_IsSharedBuffIn = 0;
Orion_IsSharedBuffOut = 0;
Orion_crossbar_model = 0;
Orion_crsbar_degree = 4;
Orion_Cxbar_Cxpoint = 0;
Orion_trans_type = 0;
Orion_IsInBuff = 1;
Orion_out_buf_size = 0;
Orion_IsOutBuff = 0;
Orion_buff_type = 0;
Orion_in_arb_model = 1;
Orion_out_arb_model = 1;
Orion_allocator_model = 1;
Orion_in_vc_arb_model = 1;
Orion_out_vc_arb_model = 1;
wire_length = 1.0;

// Default DVFS - will be overridden
dvfs_policy = static;
dvfs_epoch = 10000;
dvfs_log_interval = 10000;
power_cap = 0.6;
dvfs_min_scale = 0.5;
dvfs_max_scale = 1.0;
router_domains = 0;

// Control class settings
control_class_id = 0;
control_slo_cycles = 50;

// HW-reactive settings (when used)
hw_reactive_signal = queue;
hw_reactive_high_thresh = 0.05;
hw_reactive_low_thresh = 0.01;
hw_reactive_high_scale = 1.0;
hw_reactive_low_scale = 0.5;
hw_reactive_hysteresis_epochs = 2;
hw_reactive_per_router = 0;
hw_reactive_headroom_margin = 0.0;

// Queue-PID settings (when used)
queue_pid_target = 0.02;
queue_pid_kp = 5.0;
queue_pid_ki = 0.0;
queue_pid_kd = 0.0;
queue_pid_per_router = 0;
queue_pid_headroom_margin = 0.0;

// Perf-target settings (when used)
perf_target_metric = latency;
perf_target_value = 50;
perf_target_class = 0;
perf_target_kp = 0.1;
perf_target_per_router = 0;
perf_target_headroom_margin = 0.0;
"""
    
    config_path = os.path.join(RUNFILES_PATH, "policy_test_2class.cfg")
    with open(config_path, 'w') as f:
        f.write(config)
    return config_path


def run_policy_comparison(power_cap: float = 0.6):
    """Run all 4 policies and compare results."""
    
    print(f"\n{'='*70}")
    print(f"DVFS Policy Comparison Test - Power Cap = {power_cap}")
    print(f"{'='*70}\n")
    
    # Create test config
    config_path = create_test_config()
    print(f"Using config: {config_path}\n")
    
    policies = [
        ("uniform", {}),
        ("hw_reactive", {}),
        ("queue_pid", {}),
        ("perf_target", {}),
    ]
    
    results: List[SimResult] = []
    
    for policy_name, extra_overrides in policies:
        print(f"\n--- Testing {policy_name.upper()} policy ---")
        
        overrides = {
            "dvfs_policy": policy_name,
            "power_cap": str(power_cap),
        }
        overrides.update(extra_overrides)
        
        result = run_booksim(config_path, overrides)
        
        if result:
            results.append(result)
            print(f"  Avg Latency: {result.avg_latency:.2f}")
            print(f"  P99 Class 0 (control): {result.p99_latency_class0:.2f}")
            print(f"  P99 Class 1 (batch): {result.p99_latency_class1:.2f}")
            print(f"  Avg Power: {result.avg_power:.4f}")
            print(f"  Converged: {result.converged}")
        else:
            print("  FAILED to get results")
    
    # Summary comparison
    print(f"\n{'='*70}")
    print("SUMMARY: Control Class (Class 0) P99 Latency Comparison")
    print(f"{'='*70}")
    print(f"{'Policy':<15} {'P99 Control':<15} {'P99 Batch':<15} {'Avg Power':<12}")
    print("-" * 60)
    
    for r in results:
        print(f"{r.policy:<15} {r.p99_latency_class0:<15.2f} {r.p99_latency_class1:<15.2f} {r.avg_power:<12.4f}")
    
    # Check expected ordering
    print(f"\n{'='*70}")
    print("Expected ordering: uniform > hw_reactive > queue_pid > perf_target")
    print("(Lower P99 is better for control class)")
    print(f"{'='*70}")
    
    policy_order = ["uniform", "hw_reactive", "queue_pid", "perf_target"]
    p99_values = {r.policy: r.p99_latency_class0 for r in results}
    
    ordering_correct = True
    for i in range(len(policy_order) - 1):
        p1, p2 = policy_order[i], policy_order[i+1]
        if p1 in p99_values and p2 in p99_values:
            if p99_values[p1] < p99_values[p2]:
                print(f"  WARNING: {p1} ({p99_values[p1]:.2f}) < {p2} ({p99_values[p2]:.2f}) - unexpected!")
                ordering_correct = False
            else:
                print(f"  OK: {p1} ({p99_values[p1]:.2f}) >= {p2} ({p99_values[p2]:.2f})")
    
    if ordering_correct:
        print("\n✓ Policy ordering is as expected!")
    else:
        print("\n✗ Policy ordering needs investigation")
    
    return results


def run_with_existing_configs(power_cap: float = 0.6):
    """Run tests using existing stress configs."""
    
    print(f"\n{'='*70}")
    print(f"Running with existing stress configs - Power Cap = {power_cap}")
    print(f"{'='*70}\n")
    
    stress_configs_base = os.path.join(os.path.dirname(__file__), "..", "booksim2", "stress_configs")
    
    # Find matching configs across policies
    test_configs = [
        # (policy_folder, config_name)
        ("uniform_throttle", "uniform_uniform_0.6_0.7W.cfg"),
        ("hw_reactive", "hw_reactive_uniform_0.6_10W.cfg"),
        ("queue_pid", "queue_pid_uniform_0.6_10W.cfg"),
        ("optimal_control", "perf_target_uniform_0.6_10W.cfg"),
    ]
    
    results = []
    
    for policy_folder, config_name in test_configs:
        config_path = os.path.join(stress_configs_base, policy_folder, config_name)
        
        if not os.path.exists(config_path):
            print(f"  Config not found: {config_path}")
            continue
        
        print(f"\n--- Running {policy_folder}: {config_name} ---")
        
        # Override power cap to match our test
        overrides = {"power_cap": str(power_cap), "max_samples": "8"}
        
        result = run_booksim(config_path, overrides)
        
        if result:
            results.append(result)
            print(f"  Avg Latency: {result.avg_latency:.2f}")
            print(f"  P99 Control: {result.p99_latency_class0:.2f}")
            print(f"  Converged: {result.converged}")
    
    return results


if __name__ == "__main__":
    power_cap = 0.6
    if len(sys.argv) > 1:
        power_cap = float(sys.argv[1])
    
    print("Testing DVFS Policy Comparison")
    print("=" * 70)
    
    # Run with generated config
    results = run_policy_comparison(power_cap)
    
    # Save results
    results_file = os.path.join(os.path.dirname(__file__), "..", "policy_comparison_results.json")
    with open(results_file, 'w') as f:
        json.dump([{
            "policy": r.policy,
            "traffic": r.traffic,
            "power_cap": r.power_cap,
            "avg_latency": r.avg_latency,
            "p99_latency_class0": r.p99_latency_class0,
            "p99_latency_class1": r.p99_latency_class1,
            "avg_power": r.avg_power,
            "throughput": r.throughput,
            "converged": r.converged,
        } for r in results], f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
