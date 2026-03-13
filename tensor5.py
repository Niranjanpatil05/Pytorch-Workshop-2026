import torch
x =  torch.FloatTensor([1.5,2.64,3.987,4.432])
print(x)
print(x.dtype)

import torch
x = torch.tensor([1,2,3], dtype=torch.float32)
print(x.dtype)


import torch
x = torch.rand(3,3)

print(x)
print(x.dtype)

import torch
x = torch.rand(3,3,2)

print(x)
print(x.dtype)


import torch
x = torch.randn(3,3,2)
print(x)

import torch
x = torch.rand(3,2)
print(x)

import torch
x = torch.randint(0,10,(3,3))
print(x)