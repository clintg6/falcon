from .base_profiler import BaseProfiler

def create_profiler(backend: str, verbose: bool = True) -> BaseProfiler:
    """Factory function to create a profiler instance."""
    if backend.lower() == 'jax':
        from .jax_profiler import JAXProfiler
        return JAXProfiler(verbose=verbose)
    elif backend.lower() == 'torch':
        from .torch_profiler import TorchProfiler
        return TorchProfiler(verbose=verbose)
    elif backend.lower() == 'aten':
        from .aten_profiler import AtenProfiler
        return AtenProfiler(verbose=verbose)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

