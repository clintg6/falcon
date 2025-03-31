import unittest
import torch
import torch.nn as nn
from falcon.torch_profiler import TorchProfiler

class TestTorchProfiler(unittest.TestCase):
    def setUp(self):
        self.profiler = TorchProfiler(verbose=False)
        self.dtype = torch.float16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_linear_module(self, dtype=torch.float32):
        # Test nn.Linear
        layer = nn.Linear(in_features=10, out_features=5).to(self.device).to(self.dtype)
        x = torch.randn(2, 10, dtype=self.dtype).to(self.device)
        
        self.profiler.enable_logging(modules=[nn.Linear])
        output = layer(x)
        
        # Check logging
        logs = self.profiler.get_logged_operations()
        self.assertGreater(len(logs), 0)
        self.assertEqual(logs[0]['module_type'], 'Linear')
        self.assertEqual(logs[0]['input_shape'], '(2, 10)')
        self.assertEqual(logs[0]['input_dtype'], str(self.dtype))
        self.assertEqual(logs[0]['output_shape'], '(2, 5)')
        self.assertEqual(logs[0]['output_dtype'], str(self.dtype))
        
        # Disable logging
        self.profiler.disable_logging()
        
        # Check benchmarking
        time = self.profiler.benchmark_layer('Linear', (2, 10), str(self.dtype), {'in_features': 10, 'out_features': 5})
        self.assertGreaterEqual(time, 0.0)

    def test_conv2d_module(self, dtype=torch.float32):
        # Test nn.Conv2d
        layer = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3).to(self.device).to(self.dtype)
        x = torch.randn(2, 3, 16, 16, dtype=self.dtype).to(self.device)
        
        self.profiler.enable_logging(modules=[nn.Conv2d])
        output = layer(x)
        
        # Check logging
        logs = self.profiler.get_logged_operations()
        self.assertGreater(len(logs), 0)
        self.assertEqual(logs[0]['module_type'], 'Conv2d')
        self.assertEqual(logs[0]['input_shape'], '(2, 3, 16, 16)')
        self.assertEqual(logs[0]['input_dtype'], str(self.dtype))
        self.assertEqual(logs[0]['output_shape'], '(2, 6, 14, 14)')  # 16-3+1=14
        self.assertEqual(logs[0]['output_dtype'], str(self.dtype))

        # Disable logging
        self.profiler.disable_logging()
        
        # Check benchmarking
        time = self.profiler.benchmark_layer('Conv2d', (2, 3, 16, 16), str(self.dtype), 
                                           {'in_channels': 3, 'out_channels': 6, 'kernel_size': 3})
        self.assertGreaterEqual(time, 0.0)

    def test_layernorm_module(self, dtype=torch.float32):
        # Test nn.LayerNorm
        layer = nn.LayerNorm(normalized_shape=10).to(self.device).to(self.dtype)
        x = torch.randn(2, 10, dtype=self.dtype).to(self.device)
        
        self.profiler.enable_logging(modules=[nn.LayerNorm])
        output = layer(x)
        
        # Check logging
        logs = self.profiler.get_logged_operations()
        self.assertGreater(len(logs), 0)
        self.assertEqual(logs[0]['module_type'], 'LayerNorm')
        self.assertEqual(logs[0]['input_shape'], '(2, 10)')
        self.assertEqual(logs[0]['input_dtype'], str(self.dtype))
        self.assertEqual(logs[0]['output_shape'], '(2, 10)')
        self.assertEqual(logs[0]['output_dtype'], str(self.dtype))

        # Disable logging
        self.profiler.disable_logging()
        
        # Check benchmarking
        time = self.profiler.benchmark_layer('LayerNorm', (2, 10), str(self.dtype), {'normalized_shape': 10})
        self.assertGreaterEqual(time, 0.0)

if __name__ == '__main__':
    unittest.main()
