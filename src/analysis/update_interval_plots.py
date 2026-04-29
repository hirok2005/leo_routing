import os
import glob
import pickle
import re
import argparse
import sys
import numpy as np
from collections import defaultdict
import concurrent.futures
import time
import matplotlib.pyplot as plt
import dsns.logging

BIN_SIZE_SECONDS = 0.5

def analyze_file(filepath):
    with open(filepath, "rb") as f:
        try:
            data = pickle.load(f)
            if len(data) == 3:
                direct_messages, broadcast_messages, other_events = data
            else:
                return None
        except Exception:
            return None

    file_stats = {}

    for msg, msg_data in direct_messages.items():
        if hasattr(msg, "data") and isinstance(msg.data, str) and "-" in msg.data:
            flow_name = msg.data.rpartition("-")[0]
        else:
            flow_name = "Unknown"
            
        time_val = msg_data.start_time
        if time_val is None:
            continue
            
        bin_time = float(int(time_val // BIN_SIZE_SECONDS) * BIN_SIZE_SECONDS)
        
        if flow_name not in file_stats:
            file_stats[flow_name] = {}
        if bin_time not in file_stats[flow_name]:
            file_stats[flow_name][bin_time] = {'total': 0, 'dropped': 0}
            
        file_stats[flow_name][bin_time]['total'] += 1
        if msg_data.dropped:
            file_stats[flow_name][bin_time]['dropped'] += 1

    return file_stats

def main(results_dir, output_dir):
    if not os.path.exists(results_dir):
        alt_dir = "results" if results_dir == "results_dir/" else "results_dir"
        if os.path.exists(alt_dir):
            print(f"Directory '{results_dir}' not found. Using '{alt_dir}' instead.")
            results_dir = alt_dir
        else:
            print(f"Error: Directory '{results_dir}' not found.")
            return

    os.makedirs(output_dir, exist_ok=True)
    pattern = re.compile(r"update_inteval_([\d\.]+)_")
    files = glob.glob(os.path.join(results_dir, "*.pickle"))

    if not files:
        print(f"No pickle files found in {results_dir}")
        return

    aggregate = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'total': 0, 'dropped': 0})))

    print(f"Analyzing {len(files)} log files using {os.cpu_count()} workers...")
    
    tasks = []
    task_intervals = []
    for filepath in files:
        filename = os.path.basename(filepath)
        match = pattern.search(filename)
        if match:
            tasks.append(filepath)
            task_intervals.append(float(match.group(1)))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(analyze_file, tasks)

    processed_count = 0
    for update_interval, file_stats in zip(task_intervals, results):
        if file_stats:
            processed_count += 1
            for flow_name, time_bins in file_stats.items():
                for bin_time, stats in time_bins.items():
                    aggregate[flow_name][update_interval][bin_time]['total'] += stats['total']
                    aggregate[flow_name][update_interval][bin_time]['dropped'] += stats['dropped']

    if not aggregate:
        print("No valid data found in files.")
        return

    print(f"Successfully processed {processed_count} files.")
    print(f"Generating plots in '{output_dir}/'...")

    plt.rcParams.update({'font.size': 12, 'lines.linewidth': 2})

    for flow_name in sorted(aggregate.keys()):
        plt.figure(figsize=(12, 6))
        
        interval_data = aggregate[flow_name]
        
        for update_interval in sorted(interval_data.keys()):
            times_dict = interval_data[update_interval]
            
            sorted_times = sorted(times_dict.keys())
            x_times = list(sorted_times)
            y_drop_rates = []
            
            for t in sorted_times:
                total = times_dict[t]['total']
                dropped = times_dict[t]['dropped']
                rate = (dropped / total * 100.0) if total > 0 else 0.0
                y_drop_rates.append(rate)
            
            plt.plot(x_times, y_drop_rates, marker='.', label=f"Interval: {update_interval}s")
            
        plt.title(f"Packet Drop Rate Over Time - {flow_name}")
        plt.xlabel("Simulation Time (seconds)")
        plt.ylabel("Packet Drop Rate (%)")
        plt.ylim(0, 105) 
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper right')
        
        out_filename = os.path.join(output_dir, f"{flow_name}_drop_rate_timelines.png")
        plt.savefig(out_filename, bbox_inches='tight')
        plt.close()
        print(f"Saved {out_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot instantaneous packet drop rates over time.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results_dir/",
        help="Directory containing pickle log files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results_plots",
        help="Directory to save the generated plots",
    )
    args = parser.parse_args()

    t = time.time()
    main(args.results_dir, args.output_dir)
    print(f"Finished in: {time.time() - t:.2f} seconds")