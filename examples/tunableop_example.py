import os

os.environ['PYTORCH_TUNABLEOP_ENABLED'] = '1'

import torch
import numpy as np
import pandas as pd
from falcon import create_profiler

# Setup profiler
torch_profiler = create_profiler('torch', verbose=False)

# Load benchmarking data
bench_data = pd.read_csv('bench_results.csv')

gemm_time = bench_data['total_time'].cumsum()

# Calculate the first derivative (slope of the curve)
first_derivative = np.diff(gemm_time)

# Find the index where the first derivative is less than 1
kink_index = np.where(first_derivative < 1)[0][0]

# Filter the GEMMs to include only rows where the index is < kink_index
filtered_gemms = bench_data.iloc[:kink_index + 1]

print(f'Reduced number of GEMMs to tune from {len(bench_data)} to {len(filtered_gemms)}')

# Create CUDA events to record timing
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# Record the start event
start_event.record()

# Begin GEMM tuning
for index, row in filtered_gemms.iterrows():
    key = row['original_key']
    
    # Get layer information
    layer_name, input_shape, input_dtype, kwargs = torch_profiler.parse_key(key)

    # Run layer
    _ = torch_profiler.benchmark_layer(layer_name, input_shape, input_dtype, kwargs, compile=compile)

# Record the end event
end_event.record()

# Wait for the events to complete
torch.cuda.synchronize()

# Calculate and print the elapsed time in seconds
elapsed_time = start_event.elapsed_time(end_event) / 1000
print(f"GEMM tuning time: {elapsed_time} ms")

