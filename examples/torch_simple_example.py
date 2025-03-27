import torch
import torch.nn as nn
import torch.nn.functional as F
from falcon import create_profiler

# Create profiler for PyTorch
torch_profiler = create_profiler('torch', verbose=False)

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # 1 input channel, 32 output channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # 32 input channels, 64 output channels
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)  # 2x2 pooling with stride 2
        self.linear1 = nn.Linear(1600, 256)  # Adjusted to match flattened size
        self.linear2 = nn.Linear(256, 10)  # Output layer

    def forward(self, x):
        x = self.avg_pool(F.relu(self.conv1(x)))
        x = self.avg_pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten to (1, 1600)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

# Instantiate the model
model = CNN().cuda()

# Create some test data (batch_size, channels, height, width)
x = torch.ones(1, 1, 28, 28).cuda()

torch_profiler.enable_logging(modules=[nn.Conv2d, nn.Linear])

print("Running model forward pass...")
output = model(x)

torch_profiler.disable_logging()

results = torch_profiler.benchmark_modules()
print(results.head())