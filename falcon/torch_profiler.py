import re
import ast
import json
import torch
import datetime
import functools
import pandas as pd
import torch.nn as nn
from collections import defaultdict
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

    def benchmark_layer(self, layer_name: str, input_shape: Tuple, input_dtype: str, kwargs: Dict) -> float:
        layer = self.create_layer(layer_name, kwargs)
        torch_dtype = self.get_torch_dtype(input_dtype)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        x = torch.randn(*input_shape, dtype=torch_dtype)
        layer = layer.to(device)
        x = x.to(device)
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        with torch.no_grad():
            for _ in range(2):  # Warm-up
                result = layer(x)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                
            num_runs = 10
            total_time_sec = 0.0
            for _ in range(num_runs):
                start_event.record()
                result = layer(x)
                end_event.record()
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                total_time_sec += start_event.elapsed_time(end_event) / 1000 # ms to sec
        
        return total_time_sec / num_runs

