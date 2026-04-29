import os
import sys
import pickle
import glob
import re
import argparse
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

from dsns.logging import MessageData
from dsns.message import DirectMessage


@dataclass
class RunMetrics:
    avg_delay: float
    avg_hops: float
    delivery_ratio: float
    avg_jitter: float
    throughput_mbps: float

T_TABLE_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    15: 2.131, 20: 2.086, 30: 2.042, 50: 2.009, 100: 1.984
}

def get_t_value(dof):
    if dof < 1: return 1.96
    if dof in T_TABLE_95: return T_TABLE_95[dof]
    keys = sorted(T_TABLE_95.keys())
    if dof > keys[-1]: return 1.96
    closest = min(keys, key=lambda x: abs(x - dof))
    return T_TABLE_95[closest]

def process_file(filepath):
    filename = os.path.basename(filepath)
    
    pattern = r"^(?P<scenario>failure|congestion)_(?P<percentile>\d+)_(?P<pairs>\w+)_(?P<protocol>[A-Za-z]+)_(?P<duration>[\d.]+)s_(?P<iteration>\d+)_iter(?P<bw>bw|no_bw)\.pickle$"
    match = re.match(pattern, filename)
    
    if not match:
        return None, {}
        
    metadata = match.groupdict()
    metadata['percentile'] = int(metadata['percentile'])
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        if not isinstance(data, tuple) or len(data) < 3:
            return metadata, {}
            
        direct_messages = data[0]
        if not direct_messages:
            return metadata, {}

        duration = float(metadata['duration'])
        
        has_bw_stats = (metadata['bw'] == 'bw') or (len(data) >= 4 and data[3] is not None)
        metadata['has_bw_stats'] = has_bw_stats

        flows = defaultdict(list)
        for m in direct_messages.values():
            if m.destination is not None:
                flows[(m.source, m.destination)].append(m)

        results = {}

        for flow_key, msgs in flows.items():
            total_msgs = len(msgs)
            if total_msgs == 0: continue
                
            delivered_msgs = [m for m in msgs if m.delivered]
            num_delivered = len(delivered_msgs)
            
            delivery_ratio = (num_delivered / total_msgs) * 100.0
            
            if num_delivered == 0:
                results[flow_key] = RunMetrics(0.0, 0.0, delivery_ratio, 0.0, 0.0)
                continue

            delays = [(m.end_time - m.start_time) for m in delivered_msgs if m.end_time is not None]
            avg_delay = np.mean(delays) if delays else 0.0
            
            hops_vals = [m.hops for m in delivered_msgs if m.hops is not None]
            avg_hops = np.mean(hops_vals) if hops_vals else 0.0
            
            sorted_msgs = sorted(delivered_msgs, key=lambda x: x.start_time)
            avg_jitter = 0.0
            if len(sorted_msgs) > 1:
                flow_delays = np.array([(m.end_time - m.start_time) for m in sorted_msgs if m.end_time is not None])
                if len(flow_delays) > 1:
                    avg_jitter = np.mean(np.abs(np.diff(flow_delays)))
            
            total_bytes = sum(m.message.size for m in delivered_msgs)
            throughput_bps = (total_bytes * 8) / duration
            throughput_mbps = throughput_bps / 1e6
            
            results[flow_key] = RunMetrics(avg_delay, avg_hops, delivery_ratio, avg_jitter, throughput_mbps)

        return metadata, results

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None, {}

def compute_stats(values):
    if not values: return 0.0, 0.0
    arr = np.array(values)
    mean = np.mean(arr)
    if len(arr) > 1:
        std_err = np.std(arr, ddof=1) / np.sqrt(len(arr))
        ci = std_err * get_t_value(len(arr)-1)
    else:
        ci = 0.0
    return mean, ci

def main():
    parser = argparse.ArgumentParser(description="Generate Plots from LEO Routing Simulation Logs")
    parser.add_argument("log_dir", type=str, help="Directory containing .pickle logs")
    parser.add_argument("--output-dir", type=str, default="plots", help="Directory to save plots")
    parser.add_argument("--scenario", type=str, choices=["failure", "congestion", "all"], default="all", help="Which scenario to process")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    pickle_files = glob.glob(os.path.join(args.log_dir, "*.pickle"))
    if not pickle_files:
        print(f"No pickle files found in {args.log_dir}")
        sys.exit(0)
        
    print(f"Found {len(pickle_files)} log files. Processing...")

    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))))
    
    scenario_has_bw = defaultdict(bool)
    
    with ProcessPoolExecutor() as executor:
        future_to_file = {executor.submit(process_file, f): f for f in pickle_files}
        
        for future in as_completed(future_to_file):
            metadata, results = future.result()
            if metadata is None: continue
            
            scenario = metadata['scenario']
            if args.scenario != 'all' and scenario != args.scenario:
                continue

            percentile = metadata['percentile']
            protocol = metadata['protocol']
            has_bw = metadata['has_bw_stats']
            
            if has_bw:
                scenario_has_bw[scenario] = True
            
            for flow, metrics in results.items():
                data[scenario][flow]['delay'][protocol][percentile].append(metrics.avg_delay)
                data[scenario][flow]['hops'][protocol][percentile].append(metrics.avg_hops)
                data[scenario][flow]['pdr'][protocol][percentile].append(metrics.delivery_ratio)
                data[scenario][flow]['jitter'][protocol][percentile].append(metrics.avg_jitter)
                data[scenario][flow]['throughput'][protocol][percentile].append(metrics.throughput_mbps)

    print("Processing complete. Generating plots...")

    base_metrics = [
        ('delay', 'Average Delay (s)', 'Delay'),
        ('hops', 'Average Hops', 'Hops'),
        ('pdr', 'Packet Delivery Ratio (%)', 'PDR'),
        ('jitter', 'Average Jitter (s)', 'Jitter'),
    ]
    bw_metrics = [
        ('throughput', 'Throughput (Mbps)', 'Throughput')
    ]
    
    protocol_colors = {
        'GlobalRouting': '#1f77b4',      # Blue
        'SourceRouting': '#ff7f0e',      # Orange
        'ResilientRouting': '#2ca02c'    # Green
    }
    
    for scenario, pairs_data in data.items():
        current_metrics = list(base_metrics)
        if scenario_has_bw[scenario]:
            current_metrics.extend(bw_metrics)
            
        for flow, flow_data in pairs_data.items():
            pair_name = f"{flow[0]}_{flow[1]}"
            pair_label = f"Src {flow[0]} -> Dst {flow[1]}"
            
            for metric_key, y_label, title_suffix in current_metrics:
                plt.figure(figsize=(10, 6))
                
                protocols = sorted(flow_data[metric_key].keys())
                
                for protocol in protocols:
                    percentiles_dict = flow_data[metric_key][protocol]
                    del percentiles_dict[90]
                    percentiles_sorted = sorted(percentiles_dict.keys())
                    
                    if not percentiles_sorted: continue
                    
                    means = []
                    cis = []
                    
                    for p in percentiles_sorted:
                        val_list = percentiles_dict[p]
                        m, ci = compute_stats(val_list)
                        means.append(m)
                        cis.append(ci)
                        
                    plt.errorbar(
                        percentiles_sorted, 
                        means, 
                        yerr=cis, 
                        label=protocol,
                        capsize=4,
                        fmt='-o',
                        markersize=4,
                        linewidth=2,
                        color=protocol_colors.get(protocol, None)
                    )
                
                scenario_title = "Link Failure" if scenario == "failure" else "Congestion"
                xlabel = "Failure Rate (Percentile)" if scenario == "failure" else "Congestion Level (Percentile)"
                
                plt.title(f"{scenario_title}: {title_suffix} for {pair_label}")
                plt.xlabel(xlabel)
                plt.ylabel(y_label)
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.5)
                
                filename = f"{scenario}_{pair_name}_{metric_key}.png"
                out_path = os.path.join(args.output_dir, filename)
                plt.savefig(out_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved {out_path}")

if __name__ == '__main__':
    main()