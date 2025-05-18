import pandas as pd
import numpy as np
import json
import sys
import os
from pathlib import Path

def analyze_results(results_dir, tps):
    # Load locust stats
    stats = pd.read_csv(f"{results_dir}/load_{tps}tps_stats.csv")
    
    # Calculate metrics
    metrics = {
        "tps": tps,
        "avg_response_time": stats[stats['Name'] == 'Aggregated']['Average Response Time'].values[0],
        "max_response_time": stats[stats['Name'] == 'Aggregated']['Max Response Time'].values[0],
        "p99_response_time": np.percentile(
            pd.read_csv(f"{results_dir}/load_{tps}tps_response_times.csv")['Response Time'],
            99
        ),
        "failure_rate": stats[stats['Name'] == 'Aggregated']['Failure Rate'].values[0],
    }
    
    # Save metrics
    with open(f"{results_dir}/metrics_{tps}tps.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Analysis complete for {tps} TPS test")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python analyze_results.py <results_dir> <tps>")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    tps = int(sys.argv[2])
    analyze_results(results_dir, tps)
