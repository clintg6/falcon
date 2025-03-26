import functools
import re
import ast
import jax
import time
import datetime
from flax import nnx
import jax.numpy as jnp
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
            try:
                modules = [
                    nnx.Linear,
                    nnx.Conv,
                    nnx.LayerNorm,
                    nnx.GroupNorm,
                    # Add other common modules here
                ]
            except AttributeError as e:
                print(f"Error accessing default modules: {e}")
                return False
        
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
    
    def create_layer(self, layer_name: str, kwargs: Dict) -> Any:
        """Create a Flax nnx layer based on name and kwargs."""
        if layer_name == 'Conv':
            # Extract and prepare Conv kwargs
            kernel_size = kwargs.get('kernel_size', (1, 1))
            in_features = int(kwargs.get('in_features', 1))
            out_features = int(kwargs.get('out_features', 1))
            strides = kwargs.get('strides', 1)
            if isinstance(strides, str):
                strides = int(strides)
            use_bias = kwargs.get('use_bias', False)
            padding = kwargs.get('padding', 0)
            feature_group_count = int(kwargs.get('feature_group_count', 1))
            rngs = nnx.Rngs(0)
            
            return nnx.Conv(
                in_features=in_features,
                out_features=out_features,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                feature_group_count=feature_group_count,
                use_bias=use_bias,
                rngs=rngs
            )
        
        elif layer_name == 'Linear':
            in_features = int(kwargs.get('in_features', 1))
            out_features = int(kwargs.get('out_features', 1))
            use_bias = kwargs.get('use_bias', False)
            rngs = nnx.Rngs(0)
            
            return nnx.Linear(
                in_features=in_features,
                out_features=out_features,
                use_bias=use_bias,
                rngs=rngs,
            )
        
        elif layer_name == 'LayerNorm':
            # Fixed: Add num_features parameter
            epsilon = float(kwargs.get('epsilon', 1e-5))
            use_bias = kwargs.get('use_bias', False)
            use_scale = kwargs.get('use_scale', True)
            num_features = kwargs.get('num_features')
            rngs = nnx.Rngs(0)

            if num_features and isinstance(num_features, str) and num_features.isdigit():
                num_features = int(num_features)
            
            return nnx.LayerNorm(
                epsilon=epsilon,
                use_bias=use_bias,
                use_scale=use_scale,
                num_features=num_features,
                rngs=rngs,
            )
        
        elif layer_name == 'GroupNorm':
            # Fixed: Properly map both num_groups and group_size

            num_features = kwargs.get('num_features')
            if num_features and isinstance(num_features, str) and num_features.isdigit():
                num_features = int(num_features)

            num_groups = kwargs.get('num_groups')
            if num_groups and isinstance(num_groups, str) and num_groups.isdigit():
                num_groups = int(num_groups)
            
            group_size = kwargs.get('group_size')
            if group_size and isinstance(group_size, str) and group_size.isdigit():
                group_size = int(group_size)
                
            epsilon = float(kwargs.get('epsilon', 1e-5))
            use_bias = kwargs.get('use_bias', False)
            use_scale = kwargs.get('use_scale', True)
            rngs = nnx.Rngs(0)
            
            return nnx.GroupNorm(
                num_features=num_features,
                num_groups=num_groups,
                group_size=group_size,
                epsilon=epsilon,
                use_bias=use_bias,
                use_scale=use_scale,
                rngs=rngs
            )
        
        else:
            raise ValueError(f"Unsupported layer type: {layer_name}")

    def benchmark_layer(self, layer_name: str, input_shape: Tuple, input_dtype: str, kwargs: Dict) -> float:
        """Benchmark a single layer with given parameters."""
        # Skip unsupported layers
        if layer_name not in ['Conv', 'Linear', 'LayerNorm', 'GroupNorm']:
            return -1.0
        
        # Create the layer
        layer = self.create_layer(layer_name, kwargs)
        
        # Convert dtype string to jnp dtype
        jnp_dtype = self.get_jnp_dtype(input_dtype)
        
        # Create random input data
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, input_shape, dtype=jnp_dtype)
        
        # Compile with jit for more accurate benchmarking
        @jax.jit
        def forward(x):
            return layer(x)
        
        # Warm-up runs
        for _ in range(2):
            result = forward(x)
            result.block_until_ready()
        
        # Actual benchmark (multiple runs for more accuracy)
        num_runs = 10
        total_time = 0.0
        
        for _ in range(num_runs):
            start_time = time.time()
            result = forward(x)
            result.block_until_ready()
            end_time = time.time()
            total_time += (end_time - start_time)
        
        return total_time / num_runs