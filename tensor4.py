import torch
x = torch.tensor([1,2,3,4])
y = torch.tensor([1,5,2,3,3.7])
z = torch.tensor([True , False , True])

print(x.dtype)
print(y.dtype)
print(z.dtype)

A=torch.CharTensor([1,2,3,4])
print(A)
print(A.dtype)