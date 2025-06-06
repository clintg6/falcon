# Author: Clint Greene
# Description: Script that demonstrates how to use the saved benchmarking results from TorchProfiler to accelerate hipBLASLt offline tuning
# Date: 2025-04-17

import os

log_file = 'hipblaslt_log.txt'
tuning_file = 'tuning_ready.txt'

os.environ['HIPBLASLT_LOG_MASK'] = '32'
os.environ['HIPBLASLT_LOG_FILE'] = log_file
os.environ['HIPBLASLT_TUNING_FILE'] = 'tuned.txt'

import torch
import subprocess
import numpy as np
import pandas as pd
from falcon import create_profiler

# Setup profiler
torch_profiler = create_profiler(backend='torch', level='layer', verbose=False)

# Load benchmarking data
bench_data = pd.read_csv('bench_results.csv')

# Cumulative GEMM runtime
gemm_time = bench_data['total_time'].cumsum()

# Pruning threshold
thresh = 1

# Calculate percentage change of cumulative sum
percent_change = np.diff(gemm_time) / gemm_time[:-1] * 100

# Find the index where the % change in the cumulative runtime is less than the specified threshold
filter_index = np.where(percent_change < thresh)[0][0]

# Filter the GEMMs
filtered_gemms = bench_data.iloc[:filter_index + 1]

print(f'Reduced number of GEMMs to tune from {len(bench_data)} to {len(filtered_gemms)}')

# Begin GEMM tuning
for index, row in filtered_gemms.iterrows():
    key = row['original_key']
    
    # Get layer information
    layer_name, input_shape, input_dtype, kwargs = torch_profiler.parse_key(key)

    # Run layer
    _ = torch_profiler.benchmark_layer(layer_name, input_shape, input_dtype, kwargs, compile=compile)

# Prepare the tuning file
preprocess_commands = [
    # First command: sort, count unique lines, sort numerically, strip counts, and write to tuning_unique.txt
    f"sort {log_file} | uniq -c | sort -bgr | sed 's/^ *[0-9]* *//' | tee tuning_unique.txt",
    # Second command: modify tuning_unique.txt and write to tuning_ready.txt
    f"sed 's/ --algo_method index --solution_index [0-9]*//' tuning_unique.txt | sed 's/$/ --initialization trig_float --cold_iters 50 --iters 50 --flush --rotating 512 --algo_method all/' > {tuning_file}"
]

# Execute preprocessing commands
# Execute preprocessing commands
for cmd in preprocess_commands:
    print(f"Running preprocessing: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error in preprocessing: {e}")
    except Exception as e:
        print(f"Unexpected error in preprocessing: {e}")
        
# Open and read the tuning file to begin tuning
with open(tuning_file, "r") as file:
    # Read each line (command) from the file
    for line in file:
        # Remove leading/trailing whitespace
        command = line.strip()
        if command:  # Skip empty lines
            print(f"Running: {command}")
            try:
                # Execute the command in the shell
                subprocess.run(command, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running command: {e}")
            except Exception as e:
                print(f"Unexpected error: {e}")

print("Finished tuning.")
