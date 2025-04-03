import torch
import torch.nn as nn
from diffusers import FluxPipeline
from falcon import create_profiler

# Create profiler for PyTorch
torch_profiler = create_profiler('torch', verbose=False)

# Load Flux model
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.to('cuda')

# Enable logging for specified modules
torch_profiler.enable_logging(modules=[nn.Conv2d, nn.Linear, nn.LayerNorm])

# Generate an image
image = pipe(prompt="ghibli style, a fantasy landscape with castles",
            num_inference_steps=50,
            num_images_per_prompt=1,
            height=1024,
            width=1024
        ).images[0]

# Disable logging
torch_profiler.disable_logging()

# Benchmark logged modules
results = torch_profiler.benchmark_modules()

# Sort and print the top 10 benchmarked modules by the total amount of time taken
results.sort_values(by="total_time", ascending=False, inplace=True)
print(results.head(10))

# Save benchmark results for GEMM tuning
results.to_csv('GEMM_bench_results.csv')