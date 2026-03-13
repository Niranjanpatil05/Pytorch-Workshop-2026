import torch

x = torch.tensor([1,2,3,4,5])

print("before:",x)

y = torch.square(x)
print("after:",y)


import torch
x = torch.tensor([[1,2],
                 [3,4]])
print("before:",x)
y = torch.square(x)
print("after:",y)

import torch
x = torch.tensor([4,16,64,36])

print("before:",x)

y = torch.square(x)
print("after:",y)
