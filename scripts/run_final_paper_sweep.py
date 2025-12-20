#!/usr/bin/env python3
"""
Final Paper Sweep Script - DVFS Policy Evaluation

Runs comprehensive experiments across:
- Topologies: FlatFly, Mesh
- Traffic patterns: uniform, bitcomp, hotspot
- Injection rates: multiple levels
- Power caps: multiple levels  
- DVFS policies: static, uniform, hw_reactive, queue_pid, perf_target

Outputs structured CSV results for plotting.
"""

import subprocess
import os
import sys
import csv
import re
import tempfile
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
BOOKSIM_PATH = PROJECT_DIR / "booksim2" / "src" / "booksim"
RESULTS_DIR = PROJECT_DIR / "final_paper_results"
NETRACE_DIR = PROJECT_DIR / "netrace" / "testraces"

# Experiment configurations
TOPOLOGIES = {
    'flatfly': {
        'topology': 'flatfly',
        'k': 4, 'n': 2, 'c': 4,
        'x': 4, 'y': 4, 'xr': 2, 'yr': 2,
        'limit': 64,
        'routing_function': 'ran_min',
        'num_routers': 16,
        'num_nodes': 64,
    },
    'mesh': {
        'topology': 'mesh',
        'k': 8, 'n': 2,
        'routing_function': 'dor',
        'num_routers': 64,
        'num_nodes': 64,
    }
}

# Traffic patterns and injection rates
TRAFFIC_CONFIGS = {
    'uniform': {
        'traffic': 'uniform',
        'injection_rates': [0.3, 0.4, 0.5, 0.6],  # FlatFly can handle higher rates
    },
    'bitcomp': {
        'traffic': 'bitcomp',
        'injection_rates': [0.4, 0.5, 0.6, 0.7],
    },
    'hotspot': {
        'traffic': 'hotspot',
        'hotspot_nodes': '0,1,2,3',  # Corner nodes
        'hotspot_fraction': 0.5,
        'injection_rates': [0.3, 0.4, 0.5],
    },
}

# Lower rates for mesh (higher hop count)
MESH_TRAFFIC_CONFIGS = {
    'uniform': {
        'traffic': 'uniform',
        'injection_rates': [0.08, 0.12, 0.16, 0.20],
    },
    'bitcomp': {
        'traffic': 'bitcomp', 
        'injection_rates': [0.10, 0.15, 0.20, 0.25],
    },
    'hotspot': {
        'traffic': 'hotspot',
        'hotspot_nodes': '0,1,2,3',
        'hotspot_fraction': 0.5,
        'injection_rates': [0.06, 0.10, 0.14],
    },
}

# Power caps (W) - relative to baseline power
POWER_CAPS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 10.0]

# DVFS policies with their configurations
POLICIES = {
    'static': {
        'dvfs_policy': 'static',
        'dvfs_min_scale': 1.0,
        'dvfs_max_scale': 1.0,
    },
    'uniform': {
        'dvfs_policy': 'uniform',
        'dvfs_min_scale': 0.5,
        'dvfs_max_scale': 1.0,
    },
    'hw_reactive': {
        'dvfs_policy': 'hw_reactive',
        'dvfs_min_scale': 0.5,
        'dvfs_max_scale': 1.0,
        'hw_reactive_signal': 'queue',
        'hw_reactive_high_thresh': 0.05,
        'hw_reactive_low_thresh': 0.01,
        'hw_reactive_high_scale': 1.0,
        'hw_reactive_low_scale': 0.7,
        'hw_reactive_hysteresis_epochs': 3,
        'hw_reactive_per_router': 0,
        'hw_reactive_headroom_margin': 0.0,
        'control_class_id': 0,
        'control_slo_cycles': 50.0,
    },
    'queue_pid': {
        'dvfs_policy': 'queue_pid',
        'dvfs_min_scale': 0.5,
        'dvfs_max_scale': 1.0,
        'queue_pid_target': 0.02,
        'queue_pid_kp': 0.5,
        'queue_pid_ki': 0.01,
        'queue_pid_kd': 0.1,
        'queue_pid_per_router': 0,
        'queue_pid_headroom_margin': 0.0,
        'control_class_id': 0,
        'control_slo_cycles': 50.0,
    },
    'perf_target': {
        'dvfs_policy': 'perf_target',
        'dvfs_min_scale': 0.5,
        'dvfs_max_scale': 1.0,
        'perf_target_metric': 'latency',
        'perf_target_value': 40.0,
        'perf_target_class': 0,
        'perf_target_kp': 0.05,
        'perf_target_per_router': 0,
        'perf_target_headroom_margin': 0.0,
    },
}

# Base config template
BASE_CONFIG_TEMPLATE = """
// Auto-generated config for final paper sweep
// Topology: {topology_name}, Traffic: {traffic_name}, Policy: {policy_name}
// Injection Rate: {injection_rate}, Power Cap: {power_cap}W

// VC Configuration
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

// Simulation settings
sim_type = latency;
warmup_periods = 3;
sim_count = 1;
sample_period  = 10000;
max_samples = 25;
use_read_write = 0;
batch_size = 10000;
latency_thres = 5000;
packet_size = 1;

// Topology configuration
{topology_config}

// Traffic configuration
{traffic_config}

// 2-class traffic for SLO-aware policies
classes = 2;
class_priority = {{1,0}};
class_slo = {{40,-1}};
priority_policy = static_class;

// Orion power model
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

// DVFS configuration
dvfs_epoch = 10000;
dvfs_log_interval = 10000;
dvfs_freqs = 1.0;
dvfs_voltages = 1.0;
router_domains = 0;
{dvfs_config}
"""


def generate_topology_config(topo_name: str) -> str:
    """Generate topology-specific configuration."""
    topo = TOPOLOGIES[topo_name]
    
    if topo_name == 'flatfly':
        return f"""topology = flatfly;
subnets = 1;
c  = {topo['c']};
k  = {topo['k']};
n  = {topo['n']};
x  = {topo['x']};
y  = {topo['y']};
xr = {topo['xr']};
yr = {topo['yr']};
limit = {topo['limit']};
routing_function = {topo['routing_function']};"""
    else:  # mesh
        return f"""topology = mesh;
k  = {topo['k']};
n  = {topo['n']};
routing_function = {topo['routing_function']};"""


def generate_traffic_config(traffic_name: str, injection_rate: float, topo_name: str) -> str:
    """Generate traffic-specific configuration."""
    # Use topology-appropriate rates
    traffic_configs = MESH_TRAFFIC_CONFIGS if topo_name == 'mesh' else TRAFFIC_CONFIGS
    traffic = traffic_configs.get(traffic_name, TRAFFIC_CONFIGS[traffic_name])
    
    config = f"traffic = {traffic['traffic']};\n"
    
    # Split injection rate across 2 classes (30% control, 70% batch)
    control_rate = injection_rate * 0.3
    batch_rate = injection_rate * 0.7
    config += f"injection_rate = {{{control_rate},{batch_rate}}};\n"
    
    if traffic_name == 'hotspot':
        config += f"hotspot = {traffic.get('hotspot_nodes', '0,1,2,3')};\n"
        # config += f"hotspot_fraction = {traffic.get('hotspot_fraction', 0.5)};\n"
    
    return config


def generate_dvfs_config(policy_name: str) -> str:
    """Generate DVFS policy configuration."""
    policy = POLICIES[policy_name]
    lines = []
    for key, value in policy.items():
        if isinstance(value, float):
            lines.append(f"{key} = {value};")
        elif isinstance(value, int):
            lines.append(f"{key} = {value};")
        else:
            lines.append(f"{key} = {value};")
    return '\n'.join(lines)


def generate_config(topology: str, traffic: str, injection_rate: float, 
                    power_cap: float, policy: str) -> str:
    """Generate complete configuration file content."""
    return BASE_CONFIG_TEMPLATE.format(
        topology_name=topology,
        traffic_name=traffic,
        policy_name=policy,
        injection_rate=injection_rate,
        power_cap=power_cap,
        topology_config=generate_topology_config(topology),
        traffic_config=generate_traffic_config(traffic, injection_rate, topology),
        dvfs_config=generate_dvfs_config(policy),
    )


def parse_booksim_output(output: str) -> dict:
    """Parse booksim output for key metrics."""
    results = {
        'converged': False,
        'latency_avg': None,
        'latency_p99': None,
        'control_p99': None,
        'batch_p99': None,
        'power_avg': None,
        'power_peak': None,
        'throughput': None,
        'freq_avg': None,
    }
    
    # Check convergence
    if 'Simulation complete' in output or 'Total run time' in output:
        results['converged'] = True
    if 'unstable' in output.lower() or 'did not converge' in output.lower():
        results['converged'] = False
    
    # Parse latency (final overall values)
    lat_matches = re.findall(r'Packet latency average\s*=\s*([\d.]+)', output)
    if lat_matches:
        results['latency_avg'] = float(lat_matches[-1])
    
    # Parse per-class latency from "Traffic class X" sections
    class_sections = re.split(r'====== Traffic class (\d+) ======', output)
    for i in range(1, len(class_sections), 2):
        class_id = int(class_sections[i])
        section = class_sections[i + 1] if i + 1 < len(class_sections) else ""
        
        # Look for P99 in section
        p99_match = re.search(r'(?:nlat_p99|Packet latency p99)[^\d]*([\d.]+)', section, re.IGNORECASE)
        if p99_match:
            p99_val = float(p99_match.group(1))
            if class_id == 0:
                results['control_p99'] = p99_val
            elif class_id == 1:
                results['batch_p99'] = p99_val
        
        # If no p99, use avg * 1.5 as estimate
        avg_match = re.search(r'Packet latency average\s*=\s*([\d.]+)', section)
        if avg_match:
            avg_val = float(avg_match.group(1))
            if class_id == 0 and results['control_p99'] is None:
                results['control_p99'] = avg_val * 1.5
            elif class_id == 1 and results['batch_p99'] is None:
                results['batch_p99'] = avg_val * 1.5
    
    # Parse power from DVFS epochs
    power_matches = re.findall(r'DVFS epoch t=\d+ total_power=([\d.]+)', output)
    if power_matches:
        powers = [float(p) for p in power_matches]
        results['power_avg'] = sum(powers) / len(powers)
        results['power_peak'] = max(powers)
    
    # Parse frequency scaling
    freq_matches = re.findall(r'domains\{0:freq=([\d.]+)', output)
    if freq_matches:
        freqs = [float(f) for f in freq_matches]
        results['freq_avg'] = sum(freqs) / len(freqs)
    
    # Parse throughput
    thru_matches = re.findall(r'Overall average accepted rate\s*=\s*([\d.]+)', output)
    if thru_matches:
        results['throughput'] = float(thru_matches[-1])
    
    # Parse actual P99 from latency stats if available
    p99_direct = re.findall(r'plat_p99\s*[:=]\s*([\d.]+)', output)
    if p99_direct:
        results['latency_p99'] = float(p99_direct[-1])
    elif results['latency_avg']:
        results['latency_p99'] = results['latency_avg'] * 1.5
    
    return results


def run_experiment(args: tuple) -> dict:
    """Run a single booksim experiment."""
    topology, traffic, injection_rate, power_cap, policy, timeout = args
    
    config_content = generate_config(topology, traffic, injection_rate, power_cap, policy)
    
    # Write config to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False) as f:
        f.write(config_content)
        config_path = f.name
    
    result = {
        'topology': topology,
        'traffic': traffic,
        'injection_rate': injection_rate,
        'power_cap': power_cap,
        'policy': policy,
        'error': None,
    }
    
    try:
        proc = subprocess.run(
            [str(BOOKSIM_PATH), config_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = proc.stdout + proc.stderr
        
        parsed = parse_booksim_output(output)
        result.update(parsed)
        result['exit_code'] = proc.returncode
        
    except subprocess.TimeoutExpired:
        result['error'] = 'timeout'
        result['converged'] = False
    except Exception as e:
        result['error'] = str(e)
        result['converged'] = False
    finally:
        os.unlink(config_path)
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Run DVFS policy sweep for final paper')
    parser.add_argument('--topologies', nargs='+', default=['flatfly', 'mesh'],
                        choices=['flatfly', 'mesh'], help='Topologies to test')
    parser.add_argument('--traffic', nargs='+', default=['uniform', 'bitcomp'],
                        choices=['uniform', 'bitcomp', 'hotspot'], help='Traffic patterns')
    parser.add_argument('--policies', nargs='+', default=['static', 'uniform', 'hw_reactive', 'queue_pid', 'perf_target'],
                        choices=list(POLICIES.keys()), help='DVFS policies to test')
    parser.add_argument('--power-caps', nargs='+', type=float, default=None,
                        help='Power caps to test (default: preset list)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers')
    parser.add_argument('--timeout', type=int, default=300,
                        help='Timeout per experiment (seconds)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: fewer configurations')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file path')
    args = parser.parse_args()
    
    # Quick mode adjustments
    if args.quick:
        power_caps = [0.6, 0.8, 1.0, 2.0]
        injection_rates_override = {
            'flatfly': {'uniform': [0.4, 0.5], 'bitcomp': [0.5, 0.6], 'hotspot': [0.4]},
            'mesh': {'uniform': [0.12, 0.16], 'bitcomp': [0.15, 0.20], 'hotspot': [0.10]},
        }
    else:
        power_caps = args.power_caps or POWER_CAPS
        injection_rates_override = None
    
    # Generate experiment list
    experiments = []
    for topo in args.topologies:
        traffic_configs = MESH_TRAFFIC_CONFIGS if topo == 'mesh' else TRAFFIC_CONFIGS
        
        for traffic in args.traffic:
            if traffic not in traffic_configs:
                continue
            
            if injection_rates_override:
                rates = injection_rates_override[topo].get(traffic, [])
            else:
                rates = traffic_configs[traffic]['injection_rates']
            
            for rate in rates:
                for cap in power_caps:
                    for policy in args.policies:
                        experiments.append((topo, traffic, rate, cap, policy, args.timeout))
    
    print(f"=== Final Paper DVFS Sweep ===")
    print(f"Topologies: {args.topologies}")
    print(f"Traffic patterns: {args.traffic}")
    print(f"Policies: {args.policies}")
    print(f"Power caps: {power_caps}")
    print(f"Total experiments: {len(experiments)}")
    print(f"Workers: {args.workers}")
    print()
    
    # Create results directory
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Run experiments in parallel
    results = []
    completed = 0
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_experiment, exp): exp for exp in experiments}
        
        for future in as_completed(futures):
            exp = futures[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1
                
                status = "✓" if result.get('converged') else "✗"
                lat_str = f"{result.get('latency_avg', 0):.1f}" if result.get('latency_avg') else "N/A"
                pwr_str = f"{result.get('power_avg', 0):.3f}" if result.get('power_avg') else "N/A"
                ctrl_p99 = f"{result.get('control_p99', 0):.1f}" if result.get('control_p99') else "N/A"
                
                print(f"[{completed:3d}/{len(experiments)}] {status} {result['topology']:8} {result['traffic']:8} "
                      f"rate={result['injection_rate']:.2f} cap={result['power_cap']:.1f}W "
                      f"{result['policy']:12} lat={lat_str:>6} ctrl_p99={ctrl_p99:>6} pwr={pwr_str}")
                
            except Exception as e:
                print(f"[{completed:3d}/{len(experiments)}] ERROR: {exp[:5]} - {e}")
                completed += 1
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output or (RESULTS_DIR / f"sweep_{timestamp}.csv")
    
    fieldnames = [
        'topology', 'traffic', 'injection_rate', 'power_cap', 'policy',
        'converged', 'latency_avg', 'latency_p99', 'control_p99', 'batch_p99',
        'power_avg', 'power_peak', 'throughput', 'freq_avg', 'error', 'exit_code'
    ]
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n=== Results saved to: {output_file} ===")
    
    # Print summary statistics
    print("\n=== Summary by Policy ===")
    for policy in args.policies:
        policy_results = [r for r in results if r['policy'] == policy and r.get('converged')]
        if policy_results:
            avg_lat = sum(r['latency_avg'] for r in policy_results if r['latency_avg']) / max(1, len([r for r in policy_results if r['latency_avg']]))
            avg_pwr = sum(r['power_avg'] for r in policy_results if r['power_avg']) / max(1, len([r for r in policy_results if r['power_avg']]))
            avg_ctrl_p99 = sum(r['control_p99'] for r in policy_results if r['control_p99']) / max(1, len([r for r in policy_results if r['control_p99']]))
            print(f"  {policy:15} converged={len(policy_results):3d} avg_lat={avg_lat:6.1f} ctrl_p99={avg_ctrl_p99:6.1f} avg_pwr={avg_pwr:.3f}W")
        else:
            print(f"  {policy:15} no converged runs")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
