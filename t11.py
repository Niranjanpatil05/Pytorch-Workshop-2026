import torch

# Example 1: Simple numbers
x = torch.tensor([-3, -1, 0, 2, 5])
y = torch.relu(x)
print(y)  # Output: tensor([0, 0, 0, 2, 5])

# Example 2: In a neural network layer
x = torch.tensor([-2.5, 1.3, -0.7, 4.2])
output = torch.relu(x)
print(output)  # Output: tensor([0.0000, 1.3000, 0.0000, 4.2000])
