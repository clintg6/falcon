import re
import ast 
import json
import pandas as pd
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Type, Tuple

class BaseProfiler(ABC):
    """Base class for profiling GenAI applications in different frameworks."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.logged_operations = []
        self.patched_modules = set()
        self.original_methods = {}

    @abstractmethod
    def _get_input_info(self, args: tuple) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _get_module_params(self, module) -> Dict[str, str]:
        pass

    @abstractmethod
    def enable_logging(self, modules: Optional[List[Type]] = None) -> bool:
        pass

    @abstractmethod
    def disable_logging(self) -> bool:
        pass

    @abstractmethod
    def benchmark_layer(self, layer_name: str, input_shape: Tuple, input_dtype: str, kwargs: Dict) -> float:
        pass

    @abstractmethod
    def create_layer(self, layer_name: str, kwargs: Dict) -> Any:
        pass
    
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

    def get_logged_operations(self) -> List[Dict[str, Any]]:
        """Return the list of logged operations."""
        return self.logged_operations
