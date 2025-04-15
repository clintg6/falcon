import time
import torch
import pandas as pd
from collections import defaultdict
from typing import Any, List, Dict, Optional
from dataclasses import dataclass
from statistics import mean, median

@dataclass
class OpStats:
    count: int = 0
    total_time: float = 0.0
    times: list = None

    def __post_init__(self):
        self.times = []

class OperatorTracker(torch.utils._python_dispatch.TorchDispatchMode):
    def __init__(self, profiler: 'AtenProfiler'):
        super().__init__()
        self.profiler = profiler
        self.use_cuda_events = profiler.use_cuda_events
        if self.use_cuda_events and not torch.cuda.is_available():
            raise RuntimeError("CUDA events requested but CUDA is not available.")

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        op_name = str(func)

        # Extract input shapes and datatypes
        input_shapes = []
        input_dtypes = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                input_shapes.append(tuple(arg.shape))
                input_dtypes.append(str(arg.dtype))
        input_shape_str = str(tuple(input_shapes)) if input_shapes else "N/A"
        input_dtype_str = input_dtypes[0] if len(input_dtypes) == 1 else str(tuple(input_dtypes))

        # Create unique key for this operator configuration
        unique_key = f"{op_name}|shape={input_shape_str}|dtype={input_dtype_str}"

        if self.use_cuda_events:
            # Use CUDA events for GPU timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()  # Record start time
            result = func(*args, **kwargs)
            end_event.record()  # Record end time

            # Synchronize events and compute duration
            end_event.synchronize()
            duration = start_event.elapsed_time(end_event)  # Returns milliseconds
        else:
            # Original CPU-based timing
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            end_time = time.perf_counter()
            duration = (end_time - start_time) * 1000  # Convert to milliseconds

        # Subtract estimated overhead (if any)
        duration = max(0, duration - self.profiler.overhead_per_op)

        # Update aggregate stats (for print_statistics)
        agg_stats = self.profiler.op_stats[op_name]
        agg_stats.count += 1
        agg_stats.total_time += duration
        agg_stats.times.append(duration)

        # Update detailed stats (for unique configurations)
        detailed_stats = self.profiler.detailed_op_stats[unique_key]
        detailed_stats.count += 1
        detailed_stats.total_time += duration
        detailed_stats.times.append(duration)
        self.profiler.detailed_op_info[unique_key] = {
            'op_name': op_name,
            'input_shapes': input_shape_str,
            'input_dtypes': input_dtype_str
        }

        if self.profiler.verbose:
            print(f"Op: {op_name}, Shapes: {input_shape_str}, Dtype: {input_dtype_str}, Duration: {duration:.3f}ms")

        return result

class AtenProfiler:
    """Standalone profiler for PyTorch ATen operators."""

    def __init__(self, verbose: bool = True, calibrate_overhead: bool = True):
        self.verbose = verbose
        self.use_cuda_events = torch.cuda.is_available()
        self.op_stats: Dict[str, OpStats] = defaultdict(OpStats)  # Aggregate stats by op name
        self.detailed_op_stats: Dict[str, OpStats] = defaultdict(OpStats)  # Stats by op+shape+dtype
        self.detailed_op_info: Dict[str, Dict] = {}  # Metadata for unique configs
        self.tracker = None
        self.overhead_per_op = 0.0  # Estimated overhead per operation (ms)

        if calibrate_overhead and self.use_cuda_events:
            print('Calibrating overhead')
            self.calibrate_overhead()

    def calibrate_overhead(self, num_trials: int = 100):
        """
        Estimate logging overhead by timing a no-op operation.

        Args:
            num_trials: Number of trials to average overhead.
        """
        if not torch.cuda.is_available():
            if self.verbose:
                print("CUDA not available, skipping overhead calibration.")
            return

        def no_op(*args, **kwargs):
            pass

        overheads = []
        for _ in range(num_trials):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            no_op()
            end_event.record()

            end_event.synchronize()
            overhead = start_event.elapsed_time(end_event)
            overheads.append(overhead)

        self.overhead_per_op = mean(overheads)
        if self.verbose:
            print(f"Calibrated logging overhead: {self.overhead_per_op:.3f}ms per operation")

    def enable_logging(self) -> bool:
        """Enable ATen operator-level profiling."""
        if self.tracker is not None:
            if self.verbose:
                print("Operator tracking already enabled")
            return True
        self.tracker = OperatorTracker(self)
        self.tracker.__enter__()
        if self.verbose:
            print("Enabled ATen operator tracking")
        return True

    def disable_logging(self) -> bool:
        """Disable ATen operator-level profiling."""
        if self.tracker is not None:
            self.tracker.__exit__(None, None, None)
            self.tracker = None
            if self.verbose:
                print("Disabled ATen operator tracking")
            return True
        return False

    def get_logged_operations(self) -> List[Dict[str, Any]]:
        """Return detailed logged ATen operations with shapes and dtypes."""
        return [
            {
                'operator': info['op_name'],
                'input_shapes': info['input_shapes'],
                'input_dtypes': info['input_dtypes'],
                'count': stats.count,
                'total_time_ms': stats.total_time,
                'mean_time_ms': mean(stats.times) if stats.times else 0.0,
                'median_time_ms': median(stats.times) if stats.times else 0.0,
            }
            for unique_key, stats in self.detailed_op_stats.items()
            if (info := self.detailed_op_info.get(unique_key))
        ]

    def summarize_operations(self) -> Dict[str, int]:
        """Summarize ATen operator calls by count (aggregate)."""
        return {op_name: stats.count for op_name, stats in self.op_stats.items()}

    def print_statistics(self, top_k: Optional[int] = None, sort_by: str = 'time', save_csv: Optional[str] = None):
        """Print aggregated ATen operator usage statistics."""
        print("\n=== ATen Operator Usage Statistics (Aggregated) ===")
        if not self.op_stats:
            print("No operators were called during tracking period.")
            return

        max_name_length = max(len(name) for name in self.op_stats.keys())
        header = (
            f"{'Operator':<{max_name_length}} | {'Count':>10} | {'Total Time':>12} | "
            f"{'Avg Time':>12} | {'Med Time':>12} | {'% Time':>8}"
        )
        print(f"\n{header}")
        print("-" * len(header))

        total_time = sum(stats.total_time for stats in self.op_stats.values())

        rows = []
        if sort_by == 'time':
            sorted_ops = sorted(self.op_stats.items(), key=lambda x: (-x[1].total_time, x[0]))
        elif sort_by == 'count':
            sorted_ops = sorted(self.op_stats.items(), key=lambda x: (-x[1].count, x[0]))
        elif sort_by == 'avg_time':
            sorted_ops = sorted(self.op_stats.items(),
                              key=lambda x: (-x[1].total_time/x[1].count if x[1].count > 0 else 0, x[0]))
        else:
            raise ValueError("sort_by must be 'time', 'count', or 'avg_time'")

        if top_k is not None:
            sorted_ops = sorted_ops[:top_k]

        for op_name, stats in sorted_ops:
            avg_time = stats.total_time / stats.count if stats.count > 0 else 0
            med_time = median(stats.times) if stats.times else 0
            time_percentage = (stats.total_time / total_time * 100) if total_time > 0 else 0

            print(
                f"{op_name:<{max_name_length}} | "
                f"{stats.count:>10,} | "
                f"{stats.total_time:>11.3f}ms | "
                f"{avg_time:>11.3f}ms | "
                f"{med_time:>11.3f}ms | "
                f"{time_percentage:>7.2f}%"
            )

            rows.append({
                'operator': op_name,
                'count': stats.count,
                'total_time_ms': stats.total_time,
                'avg_time_ms': avg_time,
                'median_time_ms': med_time,
                'percent_time': time_percentage
            })

        print(f"\nTotal time: {total_time:.3f}ms")
        print(f"Total operator calls: {sum(stats.count for stats in self.op_stats.values()):,}")

        # Save to CSV if requested
        if save_csv:
            try:
                df = pd.DataFrame(rows)
                df.to_csv(save_csv, index=False)
                if self.verbose:
                    print(f"Saved statistics to {save_csv}")
            except Exception as e:
                print(f"Failed to save CSV to {save_csv}: {str(e)}")

    def print_detailed_statistics(self, top_k: Optional[int] = None, sort_by: str = 'time', operators: Optional[List[str]] = None, save_csv: Optional[str] = None):
        """Print detailed ATen operator stats sorted by total time, including shapes and dtypes, for specified operators or all."""
        print("\n=== ATen Operator Detailed Statistics ===")
        if not self.detailed_op_stats:
            print("No operators were called during tracking period.")
            return

        # Prepare data for display and CSV
        rows = []
        for unique_key, stats in self.detailed_op_stats.items():
            info = self.detailed_op_info.get(unique_key, {})
            op_name = info.get('op_name', 'unknown')
            # Filter by specified operators if provided
            if operators is not None and op_name not in operators:
                continue
            avg_time = stats.total_time / stats.count if stats.count > 0 else 0
            med_time = median(stats.times) if stats.times else 0
            rows.append({
                'operator': op_name,
                'input_shapes': info.get('input_shapes', 'N/A'),
                'input_dtypes': info.get('input_dtypes', 'N/A'),
                'count': stats.count,
                'total_time_ms': stats.total_time,
                'avg_time_ms': avg_time,
                'median_time_ms': med_time,
                'percent_time': 0.0  # Placeholder, calculated after sorting
            })

        if not rows:
            print("No matching operators found.")
            return

        # Sort rows
        if sort_by == 'time':
            sorted_rows = sorted(rows, key=lambda x: (-x['total_time_ms'], x['operator'], x['input_shapes']))
        elif sort_by == 'count':
            sorted_rows = sorted(rows, key=lambda x: (-x['count'], x['operator'], x['input_shapes']))
        elif sort_by == 'avg_time':
            sorted_rows = sorted(rows, key=lambda x: (-x['avg_time_ms'], x['operator'], x['input_shapes']))
        else:
            raise ValueError("sort_by must be 'time', 'count', or 'avg_time'")

        if top_k is not None:
            sorted_rows = sorted_rows[:top_k]

        # Calculate total time for percentage (based on filtered operators)
        total_time = sum(row['total_time_ms'] for row in sorted_rows)

        # Update percent_time in sorted rows
        for row in sorted_rows:
            row['percent_time'] = (row['total_time_ms'] / total_time * 100) if total_time > 0 else 0

        # Determine column widths
        max_op_length = max(len(row['operator']) for row in sorted_rows) if sorted_rows else 8
        max_shape_length = max(len(row['input_shapes']) for row in sorted_rows) if sorted_rows else 12
        max_dtype_length = max(len(row['input_dtypes']) for row in sorted_rows) if sorted_rows else 10

        # Print header
        header = (
            f"{'Operator':<{max_op_length}} | "
            f"{'Input Shapes':<{max_shape_length}} | "
            f"{'Dtype':<{max_dtype_length}} | "
            f"{'Count':>10} | {'Total Time':>12} | {'Avg Time':>12} | {'Med Time':>12} | {'% Time':>8}"
        )
        print(f"\n{header}")
        print("-" * len(header))

        # Print rows
        for row in sorted_rows:
            print(
                f"{row['operator']:<{max_op_length}} | "
                f"{row['input_shapes']:<{max_shape_length}} | "
                f"{row['input_dtypes']:<{max_dtype_length}} | "
                f"{row['count']:>10,} | "
                f"{row['total_time_ms']:>11.3f}ms | "
                f"{row['avg_time_ms']:>11.3f}ms | "
                f"{row['median_time_ms']:>11.3f}ms | "
                f"{row['percent_time']:>7.2f}%"
            )

        print(f"\nTotal time: {total_time:.3f}ms")
        print(f"Total operator calls: {sum(row['count'] for row in sorted_rows):,}")

        # Save to CSV if requested
        if save_csv:
            try:
                df = pd.DataFrame(sorted_rows)
                df.to_csv(save_csv, index=False)
                if self.verbose:
                    print(f"Saved statistics to {save_csv}")
            except Exception as e:
                print(f"Failed to save CSV to {save_csv}: {str(e)}")

    def compare_benchmark_results(self, system_a_file: str, system_b_file: str, top_n: int = 10, detailed: bool = False) -> pd.DataFrame:
        """
        Compare benchmark results between two systems from CSV files generated by print_detailed_statistics or print_statistics.
        
        Args:
            system_a_file: Path to CSV file with results from system A
            system_b_file: Path to CSV file with results from system B
            top_n: Number of top differences to display
            detailed: If True, compare detailed stats (operator + shapes); if False, compare aggregated stats (operator only)
            
        Returns:
            DataFrame with comparison results sorted by largest total time difference
        """
        # Load data
        try:
            df_a = pd.read_csv(system_a_file)
            df_b = pd.read_csv(system_b_file)
        except Exception as e:
            print(f"Failed to load CSV files: {str(e)}")
            return pd.DataFrame()

        # Create merge key
        if detailed:
            # For detailed stats, merge on operator and input_shapes
            df_a['original_key'] = df_a['operator'] + '|' + df_a['input_shapes']
            df_b['original_key'] = df_b['operator'] + '|' + df_b['input_shapes']
        else:
            # For aggregated stats, merge on operator only
            df_a['original_key'] = df_a['operator']
            df_b['original_key'] = df_b['operator']

        # Merge dataframes
        comparison = pd.merge(
            df_a,
            df_b,
            on='original_key',
            suffixes=('_a', '_b'),
            how='outer'
        )

        # Calculate differences (in milliseconds)
        comparison['avg_time_diff_ms'] = comparison['avg_time_ms_b'] - comparison['avg_time_ms_a']
        comparison['avg_time_ratio'] = comparison['avg_time_ms_b'] / comparison['avg_time_ms_a'].replace(0, float('nan'))
        comparison['total_time_diff_ms'] = comparison['total_time_ms_b'] - comparison['total_time_ms_a']
        comparison['total_time_ratio'] = comparison['total_time_ms_b'] / comparison['total_time_ms_a'].replace(0, float('nan'))

        # Sort by largest total time difference
        comparison_sorted = comparison.sort_values(
            by='total_time_diff_ms',
            ascending=False
        )

        # Summary of biggest differences
        print("\nOperators with biggest performance difference:")
        top_diff = comparison_sorted.nlargest(top_n, 'total_time_diff_ms')
        for idx, row in top_diff.iterrows():
            op_name = row['operator_a'] if pd.notna(row['operator_a']) else row['operator_b']
            input_shapes = row.get('input_shapes_a') if pd.notna(row.get('input_shapes_a', '')) else row.get('input_shapes_b', 'N/A')
            total_time_a = row['total_time_ms_a'] / 1000 if pd.notna(row['total_time_ms_a']) else 0
            total_time_b = row['total_time_ms_b'] / 1000 if pd.notna(row['total_time_ms_b']) else 0
            avg_time_a = row['avg_time_ms_a'] / 1000 if pd.notna(row['avg_time_ms_a']) else 0
            avg_time_b = row['avg_time_ms_b'] / 1000 if pd.notna(row['avg_time_ms_b']) else 0
            total_time_diff = row['total_time_diff_ms'] / 1000 if pd.notna(row['total_time_diff_ms']) else 0
            total_time_ratio = row['total_time_ratio'] if pd.notna(row['total_time_ratio']) else float('nan')

            print(f"Operator: {op_name}, Input Shapes: {input_shapes}")
            print(f"  System A: {avg_time_a:.6f}s per call, {total_time_a:.2f}s total")
            print(f"  System B: {avg_time_b:.6f}s per call, {total_time_b:.2f}s total")
            print(f"  Difference: {total_time_diff:.2f}s ({total_time_ratio:.2f}x)")

        return comparison_sorted