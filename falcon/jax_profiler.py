# Author: Clint Greene
# Description: Profiles Flax nnx and linen modules
# Date: 2025-04-17

import functools
import re
import ast
import jax
import json
import time
import datetime
from flax import nnx
from flax import linen as nn
import jax.numpy as jnp
import pandas as pd
from collections import defaultdict
from .layer_factory import LayerFactory
from .base_profiler import BaseProfiler
from typing import List, Dict, Any, Optional, Tuple, Type

class JAXProfiler(BaseProfiler):
    """Profiler for Flax-based GenAI applications."""

    def create_patched_call(self, module_class, original_call):
        @functools.wraps(original_call)
        def logged_call(instance, *args, **kwargs):
            try:
                # Log input information
                input_info = self._get_input_info(args)
                
                # Extract module parameters
                module_params = self._get_module_params(instance)
                
                # Create structured log entry
                log_entry = {
                    "module_type": module_class.__name__, #__module__.split('.')[-1] + module_class.__name__,
                    "module_id": id(instance),
                    "input_shape": str(input_info.get("shape", "N/A")),
                    "input_dtype": str(input_info.get("dtype", "N/A")),
                    "module_params": module_params,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Call the original method
                result = original_call(instance, *args, **kwargs)
                
                # Log output information if possible
                if hasattr(result, 'shape'):
                    
                    # Add output info to the log entry
                    log_entry["output_shape"] = str(result.shape)
                    if hasattr(result, 'dtype'):
                        log_entry["output_dtype"] = str(result.dtype)
                        
                # Add to operation log
                self.logged_operations.append(log_entry)
                
                if self.verbose:
                    print(f"Module details: {log_entry}")
                
                return result
                
            except Exception as e:
                # Log the error but let the original call proceed
                print(f"Error in logging for {module_class.__name__}: {str(e)}")
                return original_call(instance, *args, **kwargs)
                
        return logged_call
    
    def enable_logging(self, modules: Optional[List[Type]] = None) -> bool:
        """
        Enable logging for specified modules or use defaults if none provided.
        
        Args:
            modules: List of module classes to patch. If None, uses default set.
            
        Returns:
            True if patching was successful
        """
        # Default modules to patch if none specified
        if modules is None:
            modules = [
                    nnx.Linear,
                    nnx.Conv,
                    nnx.LayerNorm,
                    nnx.GroupNorm,
                    nn.Dense,
                    nn.DenseGeneral
                ]
            
        self.modules = modules
                    
        # Patch each module
        successfully_patched = 0
        for module_class in modules:
            try:
                if module_class in self.patched_modules:
                    continue
                    
                if hasattr(module_class, '__call__'):
                    # Store the original method
                    original_call = module_class.__call__
                    self.original_methods[module_class] = original_call
                    
                    # Patch with our logging version
                    patched_call = self.create_patched_call(module_class, original_call)
                    module_class.__call__ = patched_call
                    
                    # Mark as patched
                    self.patched_modules.add(module_class)
                    successfully_patched += 1
                else:
                    print(f"Could not patch {module_class.__name__}: no __call__ method found")
            except Exception as e:
                print(f"Error patching {module_class.__name__}: {str(e)}")
        
        if self.verbose:
            print(f"Successfully patched {successfully_patched} of {len(modules)} modules")
        return successfully_patched > 0
    
    def disable_logging(self) -> bool:
        """Restore original __call__ methods on patched modules."""
        success = True
        for module_class, original_call in self.original_methods.items():
            try:
                module_class.__call__ = original_call
                self.patched_modules.remove(module_class)
            except Exception as e:
                print(f"Error restoring {module_class.__name__}: {str(e)}")
                success = False
                
        if success:
            print(f"Restored {len(self.original_methods)} original module calls")
            self.original_methods = {}
        
        return success
    
    def _get_input_info(self, args: tuple) -> Dict[str, Any]:
        """Extract shape and dtype information from input arguments."""
        info = {}
        if args:
            arg0 = args[0]
            if hasattr(arg0, 'shape'):
                info['shape'] = arg0.shape
            if hasattr(arg0, 'dtype'):
                info['dtype'] = arg0.dtype
        return info
    
    def _get_module_params(self, module) -> Dict[str, str]:
        """Extract parameter information from a module."""
        params = {}
        try:
            for k in dir(module):
                if not k.startswith("_") and not callable(getattr(module, k, None)):
                    try:
                        value = getattr(module, k)
                        if isinstance(value, nnx.Param):
                            params[k] = f"Param: shape={getattr(value, 'shape', 'N/A')}, dtype={getattr(value, 'dtype', 'N/A')}"
                        elif hasattr(value, 'shape'):
                            params[k] = f"shape={value.shape}, dtype={getattr(value, 'dtype', 'N/A')}"
                        else:
                            params[k] = str(value)
                    except Exception as e:
                        params[k] = f"Error: {str(e)}"
        except Exception as e:
            print(f"Error getting module parameters: {str(e)}")
        return params

    def get_jnp_dtype(self, dtype_str: str) -> Any:
        """Convert string dtype to jnp dtype."""
        dtype_map = {
            'float16': jnp.float16,
            'bfloat16': jnp.bfloat16,
            'float32': jnp.float32,
            'float64': jnp.float64,
            'int8': jnp.int8,
            'int16': jnp.int16,
            'int32': jnp.int32,
            'int64': jnp.int64,
            'uint8': jnp.uint8,
            'uint16': jnp.uint16,
            'uint32': jnp.uint32,
            'uint64': jnp.uint64,
            'bool': jnp.bool_,
        }
        return dtype_map.get(dtype_str, jnp.float32)
    
    def create_layer(self, layer_name: str, kwargs: Dict, input_dtype: Any = jnp.float32) -> Any:
        return LayerFactory.create_jax_layer(layer_name, kwargs, input_dtype)

    def benchmark_layer(self, layer_name: str, input_shape: Tuple, input_dtype: str, kwargs: Dict, compile: bool = False) -> float:
        """Benchmark a single layer with given parameters."""

        jnp_dtype = self.get_jnp_dtype(input_dtype)
        
        # Pass input_dtype to create_layer
        layer = self.create_layer(layer_name, kwargs, input_dtype=jnp_dtype)
            
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, input_shape, dtype=jnp_dtype)

        # Handle flax.linen module initialization
        if isinstance(layer, (nn.Dense, nn.DenseGeneral)):
            params = layer.init(key, x)
            def forward(x):
                return layer.apply(params, x)
        else:  # nnx modules
            def forward(x):
                return layer(x)
        
        # Apply compilation if requested
        if compile:
            forward = jax.jit(forward)
        
        # Warm-up runs
        for _ in range(2):
            result = forward(x)
            result.block_until_ready()
        
        # Actual benchmark (multiple runs for more accuracy)
        total_time = 0.0
        
        for _ in range(self.num_runs):
            start_time = time.time()
            result = forward(x)
            result.block_until_ready()
            end_time = time.time()
            total_time += (end_time - start_time)
        
        return total_time / self.num_runs
    
    def get_logged_operations(self) -> List[Dict[str, Any]]:
        """Return the list of logged operations."""
        return self.logged_operations
    
    def summarize_operations(self):
        """Summarizes logged operations, counting identical calls based on module name, input shape, and parameters."""
        
        operation_summary = defaultdict(int)

        for log_entry in self.logged_operations:
            # Extract details for unique identification
            module_name = log_entry.get("module_type", "unknown")
            input_shape = log_entry.get("input_shape", "unknown")
            input_dtype = log_entry.get("input_dtype", "unknown")
            module_params = log_entry.get("module_params", {})

            # Convert module_params dictionary to a sorted string for consistent key generation
            params_str = json.dumps(module_params, sort_keys=True)

            # Create a unique key for this module call
            key = f"{module_name}|{input_shape}|{input_dtype}|{params_str}"

            # Increment count for identical calls
            operation_summary[key] += 1

        return operation_summary

    def parse_key(self, key: str) -> Tuple[str, Tuple, str, Dict]:
        """Parse a log dictionary key into its components."""
        parts = key.split('|')
        if len(parts) < 4:
            raise ValueError(f"Invalid key format: {key}")
        
        layer_name = parts[0]
        input_shape = ast.literal_eval(parts[1])
        input_dtype = parts[2]
        kwargs_str = parts[3]
        
        # Parse kwargs from string
        kwargs = {}
        if kwargs_str.strip() not in ['{}', '']:
            # Extract key-value pairs using regex
            pattern = r'"([^"]+)":\s*"([^"]+)"'
            matches = re.findall(pattern, kwargs_str)
            for k, v in matches:
                # Convert string values to appropriate types
                try:
                    if v.lower() == 'none':
                        kwargs[k] = None
                    elif v.lower() == 'true':
                        kwargs[k] = True
                    elif v.lower() == 'false':
                        kwargs[k] = False
                    elif '(' in v and ')' in v:  # It's likely a tuple or other structure
                        kwargs[k] = ast.literal_eval(v)
                    elif v.isdigit():
                        kwargs[k] = int(v)
                    elif '.' in v and all(part.isdigit() for part in v.split('.') if part):
                        kwargs[k] = float(v)
                    else:
                        kwargs[k] = v
                except (ValueError, SyntaxError):
                    kwargs[k] = v
        
        return layer_name, input_shape, input_dtype, kwargs
    
    def benchmark_modules(self, compile: bool = False) -> pd.DataFrame:
        """
        Benchmark all supported layers in the module_counts dictionary.
        
        Args:
            module_counts: Dictionary with keys in format "LayerName|input_shape|input_dtype|unknown|kwargs|"
                        and values representing call counts
        
        Returns:
            Pandas DataFrame with benchmark results
        """
        results = []
        module_counts = self.summarize_operations()
        
        for key, call_count in module_counts.items():
            try:
                layer_name, input_shape, input_dtype, kwargs = self.parse_key(key)
                    
                benchmark_time = self.benchmark_layer(layer_name, input_shape, input_dtype, kwargs, compile=compile)
                total_time = benchmark_time * call_count
                
                results.append({
                    'layer_type': layer_name,
                    'input_shape': str(input_shape),
                    'input_dtype': input_dtype,
                    'time_per_call': benchmark_time,
                    'call_count': call_count,
                    'total_time': total_time,
                    'original_key': key,
                })
                if self.verbose:
                    print(f"Benchmarked {layer_name}: {benchmark_time:.6f}s ({call_count} calls)")
            except Exception as e:
                print(f"Error benchmarking {key}: {str(e)}")
        
        # Convert to DataFrame
        return pd.DataFrame(results)