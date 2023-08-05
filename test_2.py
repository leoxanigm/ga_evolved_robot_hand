import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(3, 8),
    nn.ReLU(),
    nn.Linear(8, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)

inputs = torch.rand((3,))

print(model(inputs).tolist())