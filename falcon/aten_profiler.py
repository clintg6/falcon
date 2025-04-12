from collections import defaultdict
from typing import Optional, Dict, List
import time
from dataclasses import dataclass
from statistics import mean, median
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from .base_profiler import BaseProfiler

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
        super().__init__(verbose=verbose)
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
    
    def print_statistics(self, top_k: Optional[int] = None, sort_by: str = 'time'):
        """
        Print operator usage statistics
        
        Parameters:
            top_k: Optional[int] - Number of top operators to show
            sort_by: str - Sort criterion ('time', 'count', or 'avg_time')
        """
        print("\n=== Operator Usage Statistics ===")
        if not self.op_stats:
            print("No operators were called during tracking period.")
            return
            
        # Find the longest operator name for formatting
        max_name_length = max(len(name) for name in self.op_stats.keys())
        
        # Print header
        header = (
            f"{'Operator':<{max_name_length}} | {'Count':>10} | {'Total Time':>12} | "
            f"{'Avg Time':>12} | {'Med Time':>12} | {'% Time':>8}"
        )
        print(f"\n{header}")
        print("-" * len(header))
        
        # Calculate total time
        total_time = sum(stats.total_time for stats in self.op_stats.values())
        
        # Sort operators based on criterion
        if sort_by == 'time':
            sorted_ops = sorted(self.op_stats.items(), key=lambda x: (-x[1].total_time, x[0]))
        elif sort_by == 'count':
            sorted_ops = sorted(self.op_stats.items(), key=lambda x: (-x[1].count, x[0]))
        elif sort_by == 'avg_time':
            sorted_ops = sorted(self.op_stats.items(), 
                              key=lambda x: (-x[1].total_time/x[1].count if x[1].count > 0 else 0, x[0]))
        else:
            raise ValueError("sort_by must be 'time', 'count', or 'avg_time'")
        
        # Take top k if specified
        if top_k is not None:
            sorted_ops = sorted_ops[:top_k]
        
        # Print statistics
        for op_name, stats in sorted_ops:
            avg_time = stats.total_time / stats.count if stats.count > 0 else 0
            med_time = median(stats.times) if stats.times else 0
            time_percentage = (stats.total_time / total_time * 100) if total_time > 0 else 0
            
            print(
                f"{op_name:<{max_name_length}} | "
                f"{stats.count:>10,} | "
                f"{stats.total_time:>11.3f}ms | "  # Already in ms from OperatorTracker
                f"{avg_time:>11.3f}ms | "
                f"{med_time:>11.3f}ms | "
                f"{time_percentage:>7.2f}%"
            )
        
        print(f"\nTotal time: {total_time:.3f}ms")
        print(f"Total operator calls: {sum(stats.count for stats in self.op_stats.values()):,}")