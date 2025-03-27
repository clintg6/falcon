from .base_profiler import BaseProfiler

def create_profiler(framework: str, verbose: bool = True) -> BaseProfiler:
    """Factory function to create a profiler instance."""
    if framework.lower() == 'jax':
        from .jax_profiler import JAXProfiler
        return JAXProfiler(verbose=verbose)
    elif framework.lower() == 'torch':
        from .torch_profiler import TorchProfiler
        return TorchProfiler(verbose=verbose)
    else:
        raise ValueError(f"Unsupported framework: {framework}")

