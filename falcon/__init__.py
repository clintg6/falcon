from .base_profiler import BaseProfiler

def create_profiler(backend: str, num_runs: int = 3, level: str = 'aten', verbose: bool = False) -> BaseProfiler:
    """Factory function to create a profiler instance."""
    if backend.lower() == 'jax':
        from .jax_profiler import JAXProfiler
        return JAXProfiler(verbose=verbose, num_runs=num_runs)
    elif backend.lower() == 'torch' and level.lower() == 'layer':
        from .torch_profiler import TorchProfiler
        return TorchProfiler(verbose=verbose, num_runs=num_runs)
    elif backend.lower() == 'torch' and level.lower() == 'aten':
        from .aten_profiler import AtenProfiler
        return AtenProfiler(verbose=verbose, num_runs=num_runs)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

