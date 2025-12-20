#!/usr/bin/env python3
"""
Quick test for differentiated DVFS control.
Runs one simulation per policy and compares results.
"""

import subprocess
import os
import re
import sys
import tempfile
from pathlib import Path

BOOKSIM_PATH = Path(__file__).parent.parent / "booksim2" / "src" / "booksim"

# Common config
BASE_CONFIG = """
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
max_samples = 15;
use_read_write = 0;
batch_size = 10000;
latency_thres = 2000;

traffic = {traffic_pattern};
injection_rate = 0.3;
{traffic_extra}

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

power_cap = {power_cap};
router_domains = 0;
{dvfs_config}
"""

POLICIES = {
    "static": """
dvfs_policy = static;
dvfs_epoch = 10000;
dvfs_log_interval = 10000;
dvfs_freqs = 1.0;
dvfs_voltages = 1.0;
dvfs_min_scale = 1.0;
dvfs_max_scale = 1.0;
""",
    "uniform": """
dvfs_policy = uniform;
dvfs_epoch = 10000;
dvfs_log_interval = 10000;
dvfs_freqs = 1.0;
dvfs_voltages = 1.0;
dvfs_min_scale = 0.5;
dvfs_max_scale = 1.0;
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
""",
}


def run_test(policy, power_cap=0.7, timeout=60, traffic="uniform", traffic_extra=""):
    """Run booksim with given policy and return metrics."""
    config = BASE_CONFIG.format(
        power_cap=power_cap, 
        dvfs_config=POLICIES[policy],
        traffic_pattern=traffic,
        traffic_extra=traffic_extra
    )
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False) as f:
        f.write(config)
        config_path = f.name
    
    try:
        result = subprocess.run(
            [str(BOOKSIM_PATH), config_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        output = result.stdout + result.stderr
        
        # Parse metrics
        latency = None
        power = None
        
        for line in output.split('\n'):
            if 'Packet latency average' in line:
                m = re.search(r'= ([\d.]+)', line)
                if m:
                    latency = float(m.group(1))
            if 'Total Power:' in line or 'Average Power:' in line:
                m = re.search(r'([\d.]+)\s*W', line)
                if m:
                    power = float(m.group(1))
            # Also try to get from summary
            if 'avg_lat' in line.lower() or 'latency' in line.lower():
                m = re.search(r'[\d.]+', line)
                if m and latency is None:
                    latency = float(m.group())
        
        # Try extracting from DVFS log lines
        if latency is None:
            for line in output.split('\n'):
                if 'HW_REACTIVE_DIFF' in line or 'QUEUE_PID_DIFF' in line or 'PERF_TARGET_DIFF' in line:
                    pass  # Just log lines for debugging
        
        return {
            "policy": policy,
            "power_cap": power_cap,
            "latency": latency,
            "power": power,
            "converged": "unstable" not in output.lower() and latency is not None,
            "output_snippet": output[-2000:] if len(output) > 2000 else output
        }
    except subprocess.TimeoutExpired:
        return {"policy": policy, "power_cap": power_cap, "latency": None, "power": None, 
                "converged": False, "output_snippet": "TIMEOUT"}
    finally:
        os.unlink(config_path)


def main():
    power_cap = 0.7
    if len(sys.argv) > 1:
        power_cap = float(sys.argv[1])
    
    policies_to_test = ["static", "uniform", "hw_reactive", "queue_pid", "perf_target"]
    if len(sys.argv) > 2:
        policies_to_test = sys.argv[2:]
    
    # Test both uniform and hotspot traffic patterns
    traffic_patterns = [
        ("uniform", "uniform", ""),
        ("hotspot", "hotspot", "hotspot_pct = 0.5;"),
        ("neighbor", "neighbor", ""),
    ]
    
    for traffic_name, traffic_type, traffic_extra in traffic_patterns:
        print(f"\n{'='*60}")
        print(f"Traffic: {traffic_name.upper()} - Power Cap: {power_cap}W")
        print(f"{'='*60}\n")
        
        results = []
        for policy in policies_to_test:
            print(f"Testing {policy}...", end=" ", flush=True)
            result = run_test(policy, power_cap, timeout=90, traffic=traffic_type, traffic_extra=traffic_extra)
            results.append(result)
            
            if result["converged"]:
                print(f"✓ Latency: {result['latency']:.2f} cycles")
            else:
                print(f"✗ FAILED (see output)")
                if "output_snippet" in result:
                    lines = result["output_snippet"].split('\n')
                    relevant = [l for l in lines if any(x in l for x in ['HW_REACTIVE', 'QUEUE_PID', 'PERF_TARGET', 'UNIFORM', 'STATIC', 'converged', 'unstable'])]
                    for l in relevant[-5:]:
                        print(f"   {l}")
        
        print(f"\n{'='*60}")
        print(f"RESULTS - {traffic_name.upper()}")
        print(f"{'='*60}")
        print(f"{'Policy':<15} {'Latency':>12} {'Power':>12} {'Status':>10}")
        print(f"{'-'*50}")
        
        for r in results:
            lat = f"{r['latency']:.2f}" if r['latency'] else "N/A"
            pwr = f"{r['power']:.3f}W" if r['power'] else "N/A"
            status = "OK" if r['converged'] else "FAILED"
            print(f"{r['policy']:<15} {lat:>12} {pwr:>12} {status:>10}")
        
        # Check if smart policies beat uniform
        print(f"\nAnalysis:")
        uniform_result = next((r for r in results if r['policy'] == 'uniform'), None)
        if uniform_result and uniform_result['latency']:
            for r in results:
                if r['policy'] in ['hw_reactive', 'queue_pid', 'perf_target'] and r['latency']:
                    improvement = (uniform_result['latency'] - r['latency']) / uniform_result['latency'] * 100
                    if improvement > 0:
                        print(f"  {r['policy']}: {improvement:.1f}% BETTER than uniform ✓")
                    else:
                        print(f"  {r['policy']}: {-improvement:.1f}% WORSE than uniform ✗")


if __name__ == "__main__":
    main()
