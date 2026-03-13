import torch
x = torch.ones(3,2)
y = torch.rand_like(x)

print(x)
print(y)

torch.manual_seed(2)
x = torch.rand(5)
print(x)

import torch
x = torch.zeros(3,2)
print(x)

import torch
x = torch.arange(3,10,2)
print(x)