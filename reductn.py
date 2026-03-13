import torch

# Create tensor
x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)

# Sum of elements
print("Sum:", torch.sum(x))

# Mean of elements
print("Mean:", torch.mean(x))

# Maximum value
print("Max:", torch.max(x))

# Minimum value
print("Min:", torch.min(x))