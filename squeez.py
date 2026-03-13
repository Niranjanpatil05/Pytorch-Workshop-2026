import torch
x = torch.tensor([1,2,3,4,5,6])
y =x.unsqueeze(0)
print("x shape=" , x.shape)
print("y shape=" , y.shape)


import torch
x = torch.tensor([1,2])
y = torch.tensor([10,3])

z = x+y
a = x/y
b= x*y
c = x-z
print(z)
print(a)
print(b)
print(c)

import torch

# Create first 3D tensor
tensor1 = torch.tensor([
    [[1,2],[3,4]],
    [[5,6],[7,8]]
])

# Create second 3D tensor
tensor2 = torch.tensor([
    [[2,2],[2,2]],
    [[2,2],[2,2]]
])
add = tensor1 + tensor2
sub = tensor1 - tensor2
mul = tensor1 * tensor2
div = tensor1 / tensor2

print("Tensor 1:\n", tensor1)
print("\nTensor 2:\n", tensor2)
print("\nAddition:\n", add)
print("\nSubtraction:\n", sub)
print("\nMultiplication:\n", mul)
print("\nDivision:\n", div)