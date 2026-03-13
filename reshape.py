import torch
x = torch.arange(6)
y = x.reshape(2,3)
print(y)

import torch
x = torch.tensor([[10,20,30],
                  [40,50,60]])
y = x.reshape(3,2)
print(y)

import torch
x = torch.arange(6)
y = x.reshape(3,2)
print(y)

import torch
x = torch.arange(6)
y = x.view(2,3)
print(y)

import torch
x = torch.tensor([[1,2],
                  [3,4]])
y = x.flatten()
print(y)
