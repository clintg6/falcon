import ast
import jax.numpy as jnp
from typing import Dict, Any

class LayerFactory:
    @staticmethod
    def create_jax_layer(layer_name: str, kwargs: Dict, input_dtype: Any = jnp.float32) -> Any:
        import jax
        import jax.numpy as jnp
        from flax import nnx
        from flax import linen as nn

        """Create a Flax nnx layer based on name and kwargs."""
        if layer_name == 'Conv':
            kernel_size = kwargs.get('kernel_size', (1, 1))
            in_features = int(kwargs.get('in_features', 1))
            out_features = int(kwargs.get('out_features', 1))
            strides = kwargs.get('strides', 1)
            if isinstance(strides, str):
                strides = int(strides)
            use_bias = kwargs.get('use_bias', False)
            padding = kwargs.get('padding', 0)
            feature_group_count = int(kwargs.get('feature_group_count', 1))
            rngs = nnx.Rngs(0)
            
            return nnx.Conv(
                in_features=in_features,
                out_features=out_features,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                feature_group_count=feature_group_count,
                use_bias=use_bias,
                rngs=rngs
            )
        
        elif layer_name == 'Linear':
            in_features = int(kwargs.get('in_features', 1))
            out_features = int(kwargs.get('out_features', 1))
            use_bias = kwargs.get('use_bias', False)
            rngs = nnx.Rngs(0)
            
            return nnx.Linear(
                in_features=in_features,
                out_features=out_features,
                use_bias=use_bias,
                rngs=rngs,
            )
        
        elif layer_name == 'LayerNorm':
            epsilon = float(kwargs.get('epsilon', 1e-5))
            use_bias = kwargs.get('use_bias', False)
            use_scale = kwargs.get('use_scale', True)
            num_features = kwargs.get('num_features')
            rngs = nnx.Rngs(0)
            if num_features and isinstance(num_features, str) and num_features.isdigit():
                num_features = int(num_features)
            
            return nnx.LayerNorm(
                epsilon=epsilon,
                use_bias=use_bias,
                use_scale=use_scale,
                num_features=num_features,
                rngs=rngs,
            )
        
        elif layer_name == 'GroupNorm':
            num_features = kwargs.get('num_features')
            if num_features and isinstance(num_features, str) and num_features.isdigit():
                num_features = int(num_features)
            num_groups = kwargs.get('num_groups')
            if num_groups and isinstance(num_groups, str) and num_groups.isdigit():
                num_groups = int(num_groups)
            group_size = kwargs.get('group_size')
            if group_size and isinstance(group_size, str) and group_size.isdigit():
                group_size = int(group_size)
            epsilon = float(kwargs.get('epsilon', 1e-5))
            use_bias = kwargs.get('use_bias', False)
            use_scale = kwargs.get('use_scale', True)
            rngs = nnx.Rngs(0)
            
            return nnx.GroupNorm(
                num_features=num_features,
                num_groups=num_groups,
                group_size=group_size,
                epsilon=epsilon,
                use_bias=use_bias,
                use_scale=use_scale,
                rngs=rngs
            )
        
        elif layer_name == 'Dense':
            features = int(kwargs.get('features', 1))  # out_features equivalent
            use_bias = kwargs.get('use_bias', None)
            dtype = kwargs.get('dtype', None)
            return nn.Dense(
                features=features,
                use_bias=use_bias,
                dtype=input_dtype,       
                param_dtype=input_dtype
            )
        elif layer_name == 'DenseGeneral':
            features = kwargs.get('features', 1)  # Can be int or tuple
            if isinstance(features, str):
                features = ast.literal_eval(features)
            batch_dims = kwargs.get('batch_dims', ())
            if isinstance(batch_dims, str):
                batch_dims = ast.literal_eval(batch_dims)
            use_bias = kwargs.get('use_bias', None)
            dtype = kwargs.get('dtype', None)
            return nn.DenseGeneral(
                features=features,
                batch_dims=batch_dims,
                use_bias=use_bias,
                dtype=input_dtype,     
                param_dtype=input_dtype 
            )
        
        else:
            raise ValueError(f"Unsupported JAX layer type: {layer_name}")

    @staticmethod
    def create_torch_layer(layer_name: str, kwargs: Dict) -> Any:
        import torch
        import torch.nn as nn

        """Create a PyTorch layer based on name and kwargs."""
        if layer_name == 'Conv2d':
            in_channels = int(kwargs.get('in_channels', 1))
            out_channels = int(kwargs.get('out_channels', 1))
            kernel_size = kwargs.get('kernel_size', (1, 1))
            stride = kwargs.get('stride', 1)
            padding = kwargs.get('padding', 0)
            groups = int(kwargs.get('groups', 1))
            bias = kwargs.get('bias', True)
            return nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias
            )
        
        elif layer_name == 'Linear':
            in_features = int(kwargs.get('in_features', 1))
            out_features = int(kwargs.get('out_features', 1))
            bias = kwargs.get('bias', True)
            return nn.Linear(
                in_features=in_features,
                out_features=out_features,
                bias=bias
            )
        
        elif layer_name == 'LayerNorm':
            normalized_shape = kwargs.get('normalized_shape')
            if isinstance(normalized_shape, str):
                normalized_shape = ast.literal_eval(normalized_shape)
            eps = float(kwargs.get('eps', 1e-5))
            elementwise_affine = kwargs.get('elementwise_affine', True)
            return nn.LayerNorm(
                normalized_shape=normalized_shape,
                eps=eps,
                elementwise_affine=elementwise_affine
            )
        
        elif layer_name == 'GroupNorm':
            num_groups = int(kwargs.get('num_groups', 1))
            num_channels = int(kwargs.get('num_channels', 1))
            eps = float(kwargs.get('eps', 1e-5))
            affine = kwargs.get('affine', True)
            return nn.GroupNorm(
                num_groups=num_groups,
                num_channels=num_channels,
                eps=eps,
                affine=affine
            )
        
        else:
            raise ValueError(f"Unsupported PyTorch layer type: {layer_name}")