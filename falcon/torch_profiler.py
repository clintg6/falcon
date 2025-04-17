# Author: Clint Greene
# Description: Profiles torch.nn modules
# Date: 2025-04-17

import re
import ast
import json
import time
import torch
import datetime
import functools
import pandas as pd
import torch.nn as nn
from collections import defaultdict, Counter
from .layer_factory import LayerFactory
from .base_profiler import BaseProfiler
from typing import List, Dict, Any, Optional, Type, Tuple

class TorchProfiler(BaseProfiler):
    """Profiler for PyTorch-based GenAI applications."""

    def create_patched_call(self, module_class, original_forward):
        @functools.wraps(original_forward)
        def logged_forward(instance, *args, **kwargs):
            try:                
                input_info = self._get_input_info(args)
                module_params = self._get_module_params(instance)
                log_entry = {
                    "module_type": module_class.__name__,
                    "module_id": id(instance),
                    "input_shape": str(input_info.get("shape", "N/A")),
                    "input_dtype": str(input_info.get("dtype", "N/A")),
                    "module_params": module_params,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                result = original_forward(instance, *args, **kwargs)
                
                if isinstance(result, torch.Tensor):
                    log_entry["output_shape"] = str(tuple(result.shape))
                    log_entry["output_dtype"] = str(result.dtype)
                
                self.logged_operations.append(log_entry)
                if self.verbose:
                    print(f"Module details: {log_entry}")
                return result
            except Exception as e:
                if self.verbose:
                    print(f"Error in logging for {module_class.__name__}: {str(e)}")
                return original_forward(instance, *args, **kwargs)
        return logged_forward
    
    def enable_logging(self, modules: Optional[List[Type]] = None) -> bool:
        if modules is None:
            modules = [nn.Linear, nn.Conv2d]
        
        self.modules = modules

        successfully_patched = 0
        for module_class in modules:
            try:
                if module_class in self.patched_modules:
                    continue
                if hasattr(module_class, 'forward'):
                    original_forward = module_class.forward
                    self.original_methods[module_class] = original_forward
                    patched_forward = self.create_patched_call(module_class, original_forward)
                    module_class.forward = patched_forward
                    self.patched_modules.add(module_class)
                    successfully_patched += 1
                else:
                    if self.verbose:
                        print(f"Could not patch {module_class.__name__}: no forward method found")
            except Exception as e:
                if self.verbose:
                    print(f"Error patching {module_class.__name__}: {str(e)}")
        if self.verbose:
            print(f"Successfully patched {successfully_patched} of {len(modules)} modules")
        return successfully_patched > 0
    
    def _get_input_info(self, args: tuple) -> Dict[str, Any]:
        info = {}
        if args and isinstance(args[0], torch.Tensor):
            arg0 = args[0]
            info['shape'] = tuple(arg0.shape)
            info['dtype'] = str(arg0.dtype)
        return info
    
    def _get_module_params(self, module) -> Dict[str, str]:
        params = {}
        try:
            for name, param in module.named_parameters():
                params[name] = f"Param: shape={tuple(param.shape)}, dtype={param.dtype}"
            for k in dir(module):
                if not k.startswith("_") and k not in params:
                    try:
                        value = getattr(module, k)
                        if isinstance(value, torch.Tensor):
                            params[k] = f"shape={tuple(value.shape)}, dtype={value.dtype}"
                        elif not callable(value):
                            params[k] = str(value)
                    except Exception:
                        pass
        except Exception as e:
            if self.verbose:
                print(f"Error getting module parameters: {str(e)}")
        return params
    
    def disable_logging(self) -> bool:
        success = True
        for module_class, original_forward in self.original_methods.items():
            try:
                module_class.forward = original_forward

                self.patched_modules.remove(module_class)
            except Exception as e:
                if self.verbose:
                    print(f"Error restoring {module_class.__name__}: {str(e)}")
                success = False
        if success and self.verbose:
            print(f"Restored {len(self.original_methods)} original module calls")
            self.original_methods = {}
        return success
    
    def create_layer(self, layer_name: str, kwargs: Dict) -> Any:
            return LayerFactory.create_torch_layer(layer_name, kwargs)
        
    def get_torch_dtype(self, dtype_str: str) -> Any:
        dtype_str = dtype_str.replace("torch.", "")
        dtype_map = {
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'float32': torch.float32,
            'float64': torch.float64,
            'int8': torch.int8,
            'int16': torch.int16,
            'int32': torch.int32,
            'int64': torch.int64,
            'uint8': torch.uint8,
            'bool': torch.bool,
        }

        return dtype_map.get(dtype_str, torch.float32)

    def benchmark_layer(self, layer_name: str, input_shape: Tuple, input_dtype: str, kwargs: Dict, compile: bool = False) -> float:
        layer = self.create_layer(layer_name, kwargs)

        torch_dtype = self.get_torch_dtype(input_dtype)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        x = torch.randn(*input_shape, dtype=torch_dtype)
        layer = layer.to(device).to(torch_dtype)
        x = x.to(device)

        if compile:
            compiled_layer = torch.compile(layer, mode="reduce-overhead")
            def forward(x):
                return compiled_layer(x)
        else:
            def forward(x):
                return layer(x)
        
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True) 
            end_event = torch.cuda.Event(enable_timing=True)
        
        with torch.no_grad():
            for _ in range(2):  # Warm-up
                result = forward(x)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                
            total_time_sec = 0.0
            for _ in range(self.num_runs):
                if torch.cuda.is_available():
                    start_event.record()
                    result = layer(x)
                    end_event.record()
                    torch.cuda.synchronize()
                    total_time_sec += start_event.elapsed_time(end_event) / 1000  # ms to sec
                else:
                    # Use time.time() for CPU-based timing
                    start_time = time.time()
                    result = layer(x)
                    end_time = time.time()
                    total_time_sec += end_time - start_time  # time in seconds
        
        return total_time_sec / self.num_runs

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

