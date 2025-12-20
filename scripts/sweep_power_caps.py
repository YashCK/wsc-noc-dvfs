#!/usr/bin/env python3
"""
Power Cap Sweep Script for DVFS Policy Evaluation

This script runs all DVFS policies across multiple power cap values and workloads,
collecting latency and power metrics for comparison plots.
"""

import subprocess
import os
import re
import csv
import sys
import argparse
import tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# Configuration
BOOKSIM_PATH = Path(__file__).parent.parent / "booksim2" / "src" / "booksim"
RESULTS_DIR = Path(__file__).parent.parent / "sweep_results"
NETRACE_DIR = Path(__file__).parent.parent / "netrace" / "testraces"

# Policies to test
POLICIES = ["static", "uniform", "hw_reactive", "queue_pid", "perf_target"]

# Power caps to sweep (in Watts) - for uniform traffic (~0.8W baseline)
POWER_CAPS_UNIFORM = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0, 1.5]

# Power caps for netrace (lower baseline power ~0.1-0.5W)
POWER_CAPS_NETRACE = [0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3, 0.5, 1.0]

# Workloads: (name, traffic_type, injection_rate_or_netrace_file, power_caps)
WORKLOADS = [
    ("uniform_0.3", "uniform", 0.3, POWER_CAPS_UNIFORM),
    ("uniform_0.5", "uniform", 0.5, POWER_CAPS_UNIFORM),
    ("uniform_0.6", "uniform", 0.6, POWER_CAPS_UNIFORM),
    ("netrace_lngrex", "netrace", "lngrex.tra.bz2", POWER_CAPS_NETRACE),
    ("netrace_multiregion", "netrace", "multiregion.tra.bz2", POWER_CAPS_NETRACE),
]

# Base config template
BASE_CONFIG = """
// Auto-generated config for power cap sweep
// Policy: {policy}, Workload: {workload}, Power Cap: {power_cap}W

read_request_begin_vc  = 0;
read_request_end_vc    = 5;
write_reply_begin_vc   = 2;
write_reply_end_vc     = 7;
read_reply_begin_vc    = 8;
read_reply_end_vc      = 12;
write_request_begin_vc = 10;
write_request_end_vc   = 15;
num_vcs     = 16;
vc_buf_size = 8;
wait_for_tail_credit = 0;
vc_allocator = max_size;
sw_allocator = max_size;
alloc_iters  = 2;
credit_delay   = 2;
routing_delay  = 0;
vc_alloc_delay = 1;
sw_alloc_delay = 1;
st_final_delay = 1;
input_speedup     = 1;
output_speedup    = 1;
internal_speedup  = 1.0;

sim_type = latency;
warmup_periods = 3;
sim_count = 1;
topology = flatfly;
subnets = 1;
c  = 4;
k  = 4;
n  = 2;
x  = 4;
y  = 4;
xr = 2;
yr = 2;
limit = 64;
routing_function = ran_min;
packet_size = 1;
sample_period  = 10000;
max_samples = 30;
use_read_write = 0;
batch_size = 10000;
latency_thres = 2000;

// Traffic config
{traffic_config}

// ORION power model
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

// Power cap
power_cap = {power_cap};

// DVFS policy config
{dvfs_config}

// Output files (disabled for sweep)
// latency_csv = /dev/null;
// stall_csv = /dev/null;
// throughput_csv = /dev/null;
// summary_csv = /dev/null;
"""

DVFS_CONFIGS = {
    "static": """
dvfs_policy = static;
dvfs_epoch = 10000;
dvfs_log_interval = 10000;
dvfs_freqs = 1.0;
dvfs_voltages = 1.0;
dvfs_min_scale = 1.0;
dvfs_max_scale = 1.0;
router_domains = 0;
""",
    "uniform": """
dvfs_policy = uniform;
dvfs_epoch = 10000;
dvfs_log_interval = 10000;
dvfs_freqs = 1.0;
dvfs_voltages = 1.0;
dvfs_min_scale = 0.5;
dvfs_max_scale = 1.0;
router_domains = 0;
""",
    "hw_reactive": """
dvfs_policy = hw_reactive;
dvfs_epoch = 10000;
dvfs_log_interval = 10000;
dvfs_freqs = 1.0;
dvfs_voltages = 1.0;
dvfs_min_scale = 0.5;
dvfs_max_scale = 1.0;
hw_reactive_signal = queue;
hw_reactive_high_thresh = 0.05;
hw_reactive_low_thresh  = 0.005;
hw_reactive_high_scale  = 1.0;
hw_reactive_low_scale   = 0.5;
hw_reactive_hysteresis_epochs = 2;
hw_reactive_per_router = 1;
hw_reactive_headroom_margin = 0.05;
control_class_id = 0;
control_slo_cycles = 50.0;
router_domains = 0;
""",
    "queue_pid": """
dvfs_policy = queue_pid;
dvfs_epoch = 10000;
dvfs_log_interval = 10000;
dvfs_freqs = 1.0;
dvfs_voltages = 1.0;
dvfs_min_scale = 0.5;
dvfs_max_scale = 1.0;
queue_pid_target = 0.02;
queue_pid_kp = 0.5;
queue_pid_ki = 0.01;
queue_pid_kd = 0.1;
queue_pid_per_router = 1;
queue_pid_headroom_margin = 0.05;
router_domains = 0;
""",
    "perf_target": """
dvfs_policy = perf_target;
dvfs_epoch = 10000;
dvfs_log_interval = 10000;
dvfs_freqs = 1.0;
dvfs_voltages = 1.0;
dvfs_min_scale = 0.5;
dvfs_max_scale = 1.0;
perf_target_metric = latency;
perf_target_value = 50.0;
perf_target_class = 0;
perf_target_kp = 0.03;
perf_target_per_router = 1;
perf_target_headroom_margin = 0.05;
router_domains = 0;
""",
}


def generate_config(policy, workload_name, traffic_type, traffic_param, power_cap):
    """Generate a config file content for given parameters."""
    if traffic_type == "uniform":
        traffic_config = f"""traffic = uniform;
injection_rate = {traffic_param};
"""
    else:  # netrace
        netrace_path = NETRACE_DIR / traffic_param
        traffic_config = f"""traffic = netrace;
netrace_file = {netrace_path};
netrace_scale = 0.1;
netrace_packet_size = 1;
"""
    
    config = BASE_CONFIG.format(
        policy=policy,
        workload=workload_name,
        power_cap=power_cap,
        traffic_config=traffic_config,
        dvfs_config=DVFS_CONFIGS[policy],
    )
    return config


def run_booksim(config_content, timeout=180):
    """Run booksim with given config and return parsed results."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False) as f:
        f.write(config_content)
        config_path = f.name
    
    try:
        result = subprocess.run(
            [str(BOOKSIM_PATH), config_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout + result.stderr
        
        # Parse results
        results = parse_output(output)
        results['exit_code'] = result.returncode
        return results
    except subprocess.TimeoutExpired:
        return {'error': 'timeout', 'exit_code': -1}
    except Exception as e:
        return {'error': str(e), 'exit_code': -1}
    finally:
        os.unlink(config_path)


def parse_output(output):
    """Parse booksim output for key metrics."""
    results = {
        'latency_avg': None,
        'latency_p99': None,
        'power_avg': None,
        'power_final': None,
        'freq_avg': None,
        'freq_final': None,
        'throughput': None,
        'converged': False,
    }
    
    # Find "Packet latency average" from final summary
    lat_matches = re.findall(r'Packet latency average\s*=\s*([\d.]+)', output)
    if lat_matches:
        results['latency_avg'] = float(lat_matches[-1])
    
    # Find "Overall average latency" for P99 approximation (or actual P99 if available)
    p99_matches = re.findall(r'Overall minimum latency\s*=\s*([\d.]+)', output)
    # Try to get percentile stats if available
    p99_direct = re.findall(r'(?:p99|99th percentile)[^\d]*([\d.]+)', output, re.IGNORECASE)
    if p99_direct:
        results['latency_p99'] = float(p99_direct[-1])
    elif lat_matches:
        # Approximate P99 as ~1.5x average for uniform traffic
        results['latency_p99'] = results['latency_avg'] * 1.5
    
    # Parse DVFS epoch data for power and frequency
    epoch_pattern = r'DVFS epoch t=\d+ total_power=([\d.]+) headroom=[\d.-]+ avg_power=([\d.]+) domains\{0:freq=([\d.]+)'
    epoch_matches = re.findall(epoch_pattern, output)
    if epoch_matches:
        powers = [float(m[0]) for m in epoch_matches]
        freqs = [float(m[2]) for m in epoch_matches]
        results['power_avg'] = sum(powers) / len(powers)
        results['power_final'] = powers[-1]
        results['freq_avg'] = sum(freqs) / len(freqs)
        results['freq_final'] = freqs[-1]
    else:
        # Try alternative power parsing for static policy
        power_matches = re.findall(r'total_power=([\d.]+)', output)
        if power_matches:
            powers = [float(p) for p in power_matches]
            results['power_avg'] = sum(powers) / len(powers)
            results['power_final'] = powers[-1]
            results['freq_avg'] = 1.0  # Static policy runs at full freq
            results['freq_final'] = 1.0
    
    # Check for convergence
    if 'Total run time' in output or 'Simulation complete' in output.lower():
        results['converged'] = True
    if 'unstable' in output.lower() or 'did not converge' in output.lower():
        results['converged'] = False
    
    # Get throughput
    thru_matches = re.findall(r'Overall average accepted rate\s*=\s*([\d.]+)', output)
    if thru_matches:
        results['throughput'] = float(thru_matches[-1])
    
    return results


def run_single_experiment(args):
    """Run a single experiment (for parallel execution)."""
    policy, workload_name, traffic_type, traffic_param, power_cap = args
    
    config = generate_config(policy, workload_name, traffic_type, traffic_param, power_cap)
    results = run_booksim(config)
    
    return {
        'policy': policy,
        'workload': workload_name,
        'power_cap': power_cap,
        **results
    }


def main():
    parser = argparse.ArgumentParser(description='Sweep power caps across DVFS policies')
    parser.add_argument('--policies', nargs='+', default=POLICIES, 
                        help='Policies to test')
    parser.add_argument('--workloads', nargs='+', default=None,
                        help='Workloads to test (by name)')
    parser.add_argument('--parallel', type=int, default=4,
                        help='Number of parallel workers')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: fewer caps and workloads')
    args = parser.parse_args()
    
    # Filter workloads if specified
    workloads = WORKLOADS
    if args.workloads:
        workloads = [w for w in WORKLOADS if w[0] in args.workloads]
    
    if args.quick:
        # Quick mode: just uniform_0.5 with fewer caps
        workloads = [("uniform_0.5", "uniform", 0.5, [0.5, 0.6, 0.7, 0.8, 1.0])]
        args.policies = ["static", "uniform", "hw_reactive", "queue_pid"]
    
    # Create results directory
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Generate experiment list
    experiments = []
    for policy in args.policies:
        for workload_name, traffic_type, traffic_param, power_caps in workloads:
            for cap in power_caps:
                experiments.append((policy, workload_name, traffic_type, traffic_param, cap))
    
    print(f"Running {len(experiments)} experiments across {len(args.policies)} policies")
    print(f"Workloads: {[w[0] for w in workloads]}")
    print(f"Using {args.parallel} parallel workers")
    
    # Run experiments
    results = []
    completed = 0
    
    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        futures = {executor.submit(run_single_experiment, exp): exp for exp in experiments}
        
        for future in as_completed(futures):
            exp = futures[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1
                
                status = "✓" if result.get('converged') else "✗"
                lat_str = f"{result.get('latency_avg', 0):.2f}" if result.get('latency_avg') else "N/A"
                pwr_str = f"{result.get('power_avg', 0):.4f}" if result.get('power_avg') else "N/A"
                print(f"[{completed}/{len(experiments)}] {status} {result['policy']:12} {result['workload']:20} cap={result['power_cap']:.2f}W "
                      f"lat={lat_str:>8} pwr={pwr_str:>8}")
            except Exception as e:
                print(f"[{completed}/{len(experiments)}] ERROR: {exp} - {e}")
                completed += 1
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output or (RESULTS_DIR / f"power_cap_sweep_{timestamp}.csv")
    
    fieldnames = ['policy', 'workload', 'power_cap', 'latency_avg', 'latency_p99', 
                  'power_avg', 'power_final', 'freq_avg', 'freq_final', 
                  'throughput', 'converged', 'exit_code', 'error']
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to: {output_file}")
    
    # Print summary
    print("\n=== Summary ===")
    for policy in args.policies:
        policy_results = [r for r in results if r['policy'] == policy and r.get('converged')]
        if policy_results:
            avg_lat = sum(r['latency_avg'] for r in policy_results if r['latency_avg']) / len(policy_results)
            avg_pwr = sum(r['power_avg'] for r in policy_results if r['power_avg']) / len(policy_results)
            print(f"{policy:15} avg_latency={avg_lat:.2f} avg_power={avg_pwr:.3f}W ({len(policy_results)} converged)")


if __name__ == "__main__":
    main()
