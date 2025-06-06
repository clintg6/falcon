# Author: Clint Greene
# Description: Script that demonstrates how to profile a simple CNN in JAX built from flax.nnx modules
# Date: 2025-04-17

import jax
import jax.numpy as jnp
from functools import partial
from flax import nnx
from falcon import create_profiler

# Create profilers
jax_profiler = create_profiler(backend='jax', verbose=False)

# Enable logging
jax_profiler.enable_logging()

class CNN(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
        self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
        self.linear1 = nnx.Linear(3136, 256, rngs=rngs)
        self.linear2 = nnx.Linear(256, 10, rngs=rngs)

    def __call__(self, x):
        x = self.avg_pool(nnx.relu(self.conv1(x)))
        x = self.avg_pool(nnx.relu(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)  # flatten
        x = nnx.relu(self.linear1(x))
        x = self.linear2(x)
        return x

model = CNN(rngs=nnx.Rngs(0))
    
# Create some test data
x = jnp.ones((1, 28, 28, 1))
    
# Run the model (this will trigger logging)
print("Running model forward pass...")
output = model(x)

# Disable logging
jax_profiler.disable_logging()

# Benchmark layers
results = jax_profiler.benchmark_modules()

# View benchmarking results
print(results.head())