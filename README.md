# Falcon

Falcon is a versatile profiling tool designed for analyzing the performance of deep learning models in JAX and PyTorch. It provides detailed insights into module-level and operator-level operations, helping AI developers optimize neural network performance. Falcon includes three profilers: `AtenProfiler` for low-level PyTorch ATen operators, `TorchProfiler` for PyTorch modules, and `JAXProfiler` for JAX modules (Flax). Each profiler logs execution times, input/output shapes, data types, and operation counts, with options to benchmark modules or export statistics to CSV.

## Features

- **Operator-Level Profiling**: `AtenProfiler` monitors low-level PyTorch ATen operations (e.g., `aten::addmm`, `aten::matmul`) using `TorchDispatchMode`.
- **Module-Level Profiling**: `JAXProfiler` and `TorchProfiler` track high-level module calls (e.g., `nnx.Linear`, `nn.Linear`) with input/output details and execution times.
- **Flexible Output**: Summarize operation counts, print detailed statistics, or export to CSV for further analysis.
- **Benchmarking**: Measure module performance with customizable runs and compilation options (JAX and PyTorch).
- **Cross-Framework Support**: Seamless profiling for both JAX (Flax `nnx` and `linen`) and PyTorch.

## Installation

### Prerequisites

- Python 3.9+
- JAX (`jax`, `jaxlib`)
- PyTorch (`torch`)
- Flax (`flax`)
- Pandas (`pandas`)

### Install Falcon

Clone the repository and install dependencies:

```bash
git clone https://github.com/clintg6/falcon.git
cd falcon
pip install -e . pandas
```

## Usage

### Falcon AtenProfiler: Profiling PyTorch ATen Operators

Profile low-level ATen operations with detailed statistics and optional CSV export.

```python
import torch
import torch.nn as nn
from falcon import create_profiler

# Create a model
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 2)
).to('cuda').to(torch.float16)

# Initialize profiler
profiler = create_profiler(backend='torch', level='aten', verbose=False)
profiler.enable_logging()

# Run model with profiling:
x = torch.randn(2, 10, dtype=torch.float16).to('cuda')
output = model(x)

# Print and save statistics
profiler.print_statistics(top_k=5, save_csv='operator_stats.csv')
```

**Example CSV Output** (`operator_stats.csv`):

```
operator,count,total_time_ms,avg_time_ms,median_time_ms,percent_time
aten.addmm.default,2,0.17942997961305082,0.08971498980652541,0.08971498980652541,75.84232175032176
aten._to_copy.default,1,0.03381899073254317,0.03381899073254317,0.03381899073254317,14.294772712676219
aten.relu.default,1,0.00935798957478255,0.00935798957478255,0.00935798957478255,3.9554797798973427
aten.randn.default,1,0.007312989602796733,0.007312989602796733,0.007312989602796733,3.0910894133085343
aten.t.default,2,0.003399980803951621,0.0016999904019758105,0.0016999904019758105,1.4371201436588743
```

### Falcon TorchProfiler: Profiling PyTorch Modules

Profile PyTorch modules with detailed logging and benchmarking.

```python
import torch
import torch.nn as nn
from falcon import create_profiler

# Create a model
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 2)
).to('cuda').to(torch.float16)

# Initialize profiler
profiler = create_profiler(backend='torch', level='layer', verbose=False)

# Specify layers to profile
profiler.enable_logging(modules=[nn.Linear])

# Run model
x = torch.randn(2, 10, dtype=torch.float16).to('cuda')
output = model(x)

# Benchmark modules and print results
results = profiler.benchmark_modules()
print(results)
```

### Falcon JAXProfiler: Profiling JAX Modules

Profile Flax modules (`nnx` or `linen`) with input/output details and benchmark performance.

```python
import jax
import jax.numpy as jnp
from flax import nnx
from falcon import create_profiler

# Create a model
model = nnx.Linear(10, 5, rngs=nnx.Rngs(0))
x = jax.random.normal(jax.random.PRNGKey(0), (2, 10), dtype=jnp.float16)

# Initialize profiler
profiler = create_profiler(backend='jax', verbose=False)

# Specify modules to profile
profiler.enable_logging(modules=[nnx.Linear])

# Run model
output = model(x)

# Benchmark modules and print results
results = profiler.benchmark_modules()
print(results)
```

### GEMM Tuning

Browse the examples folder to see how to use Falcon to accelerate GEMM tuning by 2x+.

## How It Works

- **AtenProfiler**: Uses PyTorch’s `TorchDispatchMode` to intercept all ATen operator calls (e.g., `aten::addmm`), tracking counts, total time, and individual durations. Outputs detailed statistics and CSV exports.
- **TorchProfiler**: Patches PyTorch module `forward` methods (e.g., `nn.Linear`, `nn.Conv2d`) to capture similar details. Offers module-level benchmarking with customizable runs.
- **JAXProfiler**: Patches Flax module `__call__` methods (e.g., `nnx.Linear`, `nn.Dense`) to log execution details like input/output shapes, dtypes, and times. Supports benchmarking with JAX’s JIT compilation.

## Contributing

Contributions are welcome! Please submit issues or pull requests to the GitHub repository. Ensure code follows PEP 8 and includes tests.

## License

MIT License. See LICENSE for details.

## Contact

For questions or support, open an issue.
