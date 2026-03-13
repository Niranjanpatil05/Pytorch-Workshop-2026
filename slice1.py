import torch

x = torch.tensor([10, 20, 30, 40, 50])
y = x[1:4]

print(y)

import torch

x = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print(x[0:2])

import torch

x = torch.tensor([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])

print(x[0:2, 1:3])

import torch
A = torch.tensor([[10,20,30],
                  [40,50,60],
                  [70,80,90]])
print(A[0:2])
print(A[:1])
