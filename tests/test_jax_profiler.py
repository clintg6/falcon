import unittest
import jax
import jax.numpy as jnp
from flax import nnx
import flax.linen as nn
from falcon.jax_profiler import JAXProfiler

class TestJAXProfiler(unittest.TestCase):
    def setUp(self):
        self.profiler = JAXProfiler(verbose=False)
        self.key = jax.random.PRNGKey(0)

    def test_linear_module(self):
        # Test nnx.Linear
        layer = nnx.Linear(in_features=10, out_features=5, rngs=nnx.Rngs(0))
        x = jax.random.normal(self.key, (2, 10), dtype=jnp.float16)
        
        self.profiler.enable_logging(modules=[nnx.Linear])
        output = layer(x)
        
        # Check logging
        logs = self.profiler.get_logged_operations()
        self.assertGreater(len(logs), 0)
        self.assertEqual(logs[0]['module_type'], 'Linear')
        self.assertEqual(logs[0]['input_shape'], '(2, 10)')
        self.assertEqual(logs[0]['input_dtype'], 'float16')
        self.assertEqual(logs[0]['output_shape'], '(2, 5)')
        self.assertEqual(logs[0]['output_dtype'], 'float16')
        
        # Check benchmarking
        time = self.profiler.benchmark_layer('Linear', (2, 10), 'float16', {'in_features': 10, 'out_features': 5})
        self.assertGreaterEqual(time, 0.0)

    def test_conv_module(self):
        # Test nnx.Conv
        layer = nnx.Conv(in_features=3, out_features=6, kernel_size=(3, 3), rngs=nnx.Rngs(0))
        x = jax.random.normal(self.key, (2, 3, 16, 16), dtype=jnp.float16)
        
        self.profiler.enable_logging(modules=[nnx.Conv])
        output = layer(x)
        
        # Check logging
        logs = self.profiler.get_logged_operations()
        self.assertGreater(len(logs), 0)
        self.assertEqual(logs[0]['module_type'], 'Conv')
        self.assertEqual(logs[0]['input_shape'], '(2, 3, 16, 16)')
        self.assertEqual(logs[0]['input_dtype'], 'float16')
        self.assertEqual(logs[0]['output_shape'], '(2, 6, 14, 14)')  # 16-3+1=14
        self.assertEqual(logs[0]['output_dtype'], 'float16')
        
        # Check benchmarking
        time = self.profiler.benchmark_layer('Conv', (2, 3, 16, 16), 'float16', 
                                           {'in_features': 3, 'out_features': 6, 'kernel_size': (3, 3)})
        self.assertGreaterEqual(time, 0.0)

    def test_layernorm_module(self):
        # Test nnx.LayerNorm
        layer = nnx.LayerNorm(num_features=10, rngs=nnx.Rngs(0))
        x = jax.random.normal(self.key, (2, 10), dtype=jnp.float16)
        
        self.profiler.enable_logging(modules=[nnx.LayerNorm])
        output = layer(x)
        
        # Check logging
        logs = self.profiler.get_logged_operations()
        self.assertGreater(len(logs), 0)
        self.assertEqual(logs[0]['module_type'], 'LayerNorm')
        self.assertEqual(logs[0]['input_shape'], '(2, 10)')
        self.assertEqual(logs[0]['input_dtype'], 'float16')
        self.assertEqual(logs[0]['output_shape'], '(2, 10)')
        self.assertEqual(logs[0]['output_dtype'], 'float16')
        
        # Check benchmarking
        time = self.profiler.benchmark_layer('LayerNorm', (2, 10), 'float16', {'num_features': 10})
        self.assertGreaterEqual(time, 0.0)

    def test_dense_module(self):
        # Test flax.linen Dense
        layer = nn.Dense(features=5, param_dtype=jnp.float16)
        x = jax.random.normal(self.key, (2, 10), dtype=jnp.float16)
        params = layer.init(self.key, x)
        
        self.profiler.enable_logging(modules=[nn.Dense])
        output = layer.apply(params, x)
        
        # Check logging
        logs = self.profiler.get_logged_operations()
        self.assertGreater(len(logs), 0)
        self.assertEqual(logs[0]['module_type'], 'Dense')
        self.assertEqual(logs[0]['input_shape'], '(2, 10)')
        self.assertEqual(logs[0]['input_dtype'], 'float16')
        self.assertEqual(logs[0]['output_shape'], '(2, 5)')
        self.assertEqual(logs[0]['output_dtype'], 'float16')
        
        # Check benchmarking
        time = self.profiler.benchmark_layer('Dense', (2, 10), 'float16', {'features': 5})
        self.assertGreaterEqual(time, 0.0)
        # Check parameter dtype
        self.assertEqual(params['params']['kernel'].dtype, jnp.float16)

if __name__ == '__main__':
    unittest.main()