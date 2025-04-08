from collections import defaultdict
from typing import Optional, Dict, List
import time
from dataclasses import dataclass
from statistics import mean, median
import torch
from torch.utils._pytorch_dispatcher import TorchDispatchMode
from base_profiler import BaseProfiler

@dataclass
class OpStats:
    count: int = 0
    total_time: float = 0.0
    times: list = None
    
    def __post_init__(self):
        self.times = []

class OperatorTracker(TorchDispatchMode):
    def __init__(self, profiler: 'AtenProfiler'):
        super().__init__()
        self.profiler = profiler
        self.op_stats = defaultdict(OpStats)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        op_name = str(func)
        
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        duration = (end_time - start_time) * 1000  # Convert to milliseconds
        
        stats = self.profiler.op_stats[op_name]
        stats.count += 1
        stats.total_time += duration
        stats.times.append(duration)
        
        if self.profiler.verbose:
            print(f"Op: {op_name}, Duration: {duration:.3f}ms")
        
        return result

class AtenProfiler(BaseProfiler):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose, framework='torch')
        self.op_stats: Dict[str, OpStats] = defaultdict(OpStats)
        self.tracker = None

    def enable_logging(self, modules=None) -> bool:
        """Enable operator-level profiling."""
        if self.tracker is not None:
            if self.verbose:
                print("Operator tracking already enabled")
            return True
        self.tracker = OperatorTracker(self)
        if self.verbose:
            print("Enabled ATen operator tracking")
        return True

    def disable_logging(self) -> None:
        """Disable operator-level profiling."""
        if self.tracker is not None:
            self.tracker.__exit__(None, None, None)
            self.tracker = None
            if self.verbose:
                print("Disabled ATen operator tracking")

    def get_logged_operations(self) -> List[Dict]:  # Changed to List[Dict] for consistency
        return [
            {
                'module_type': op_name,  # Align with BaseProfiler's expectation
                'count': stats.count,
                'total_time_ms': stats.total_time,
                'mean_time_ms': mean(stats.times) if stats.times else 0.0,
                'median_time_ms': median(stats.times) if stats.times else 0.0,
            }
            for op_name, stats in self.op_stats.items()
        ]

    def summarize_operations(self) -> Dict[str, int]:
        return {op_name: stats.count for op_name, stats in self.op_stats.items()}