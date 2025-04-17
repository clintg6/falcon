import re
import ast 
import json
import pandas as pd
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Type, Tuple

class BaseProfiler(ABC):
    """Base class for profiling GenAI applications in different frameworks."""
    
    def __init__(self, num_runs: int, verbose: bool = True):
        self.verbose = verbose
        self.num_runs = num_runs
        self.logged_operations = []
        self.patched_modules = set()
        self.original_methods = {}

    @abstractmethod
    def enable_logging(self, modules: Optional[List[Type]] = None) -> bool:
        pass

    @abstractmethod
    def disable_logging(self) -> bool:
        pass

    @abstractmethod
    def get_logged_operations(self) -> List[Dict]:
        pass

    @abstractmethod
    def summarize_operations(self) -> Dict[str, int]:
        pass

    def compare_benchmark_results(self, system_a_file: str, system_b_file: str, top_n: int = 10) -> pd.DataFrame:
        """
        Compare benchmark results between two systems.
        
        Args:
            system_a_file: Path to CSV file with benchmark results from system A
            system_b_file: Path to CSV file with benchmark results from system B
            
        Returns:
            DataFrame with comparison results sorted by largest time difference
        """
        # Load data
        df_a = pd.read_csv(system_a_file)
        df_b = pd.read_csv(system_b_file)
        
        # Merge dataframes on original_key
        comparison = pd.merge(
            df_a, 
            df_b, 
            on='original_key', 
            suffixes=('_a', '_b'),
            how='outer'
        )
        
        # Calculate differences
        comparison['time_per_call_diff'] = comparison['time_per_call_b'] - comparison['time_per_call_a']
        comparison['time_per_call_ratio'] = comparison['time_per_call_b'] / comparison['time_per_call_a']
        comparison['total_time_diff'] = comparison['total_time_b'] - comparison['total_time_a']
        comparison['total_time_ratio'] = comparison['total_time_b'] / comparison['total_time_a']
        
        # Sort by largest total time difference
        comparison_sorted = comparison.sort_values(
            by='total_time_diff', 
            ascending=False
        )

        # Summary of biggest differences
        print("\nLayers with biggest performance difference:")
        top_diff = comparison_sorted.nlargest(top_n, 'total_time_diff')
        for idx, row in top_diff.iterrows():
            print(f"Layer: {row['layer_type_a']}, Input: {row['input_shape_a']}")
            print(f"  SystemA: {row['time_per_call_a']:.6f}s per call, {row['total_time_a']:.2f}s total")
            print(f"  SystemB: {row['time_per_call_b']:.6f}s per call, {row['total_time_b']:.2f}s total")
            print(f"  Difference: {row['total_time_diff']:.2f}s ({row['total_time_ratio']:.2f}x)")
        
        return comparison_sorted

    def save_benchmark_results(self, df: pd.DataFrame, system_name: str, file_path: str = None):
        """Save benchmark results to a CSV file."""
        df['system'] = system_name
        if file_path is None:
            file_path = f"benchmark_results_{system_name}.csv"
        df.to_csv(file_path, index=False)
        print(f"Benchmark results saved to {file_path}")
        return file_path
    
    def clear_logs(self) -> bool:
        """Clear the internal log of operations."""
        self.logged_operations = []
        print("Cleared operation logs")
        return True

