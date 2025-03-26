from .jax_profiler import JAXProfiler
from .torch_profiler import TorchProfiler
from .base_profiler import BaseProfiler

def create_profiler(framework: str, verbose: bool = True) -> BaseProfiler:
    """Factory function to create a profiler instance."""
    if framework.lower() == 'jax':
        return JAXProfiler(verbose=verbose)
    elif framework.lower() == 'torch':
        return TorchProfiler(verbose=verbose)
    else:
        raise ValueError(f"Unsupported framework: {framework}")
