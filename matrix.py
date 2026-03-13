import torch
a = torch.tensor([[1,2],
                 [3,4]])
b = torch.tensor([[5,6],
                  [7,8]])
c = torch.matmul(a,b)
print(c)



import torch
tensor1 = torch.tensor([
    [[1, 2],
     [3, 4]],

    [[5, 6],
     [7, 8]]
])
tensor2 = torch.tensor([
    [[1, 0],
     [0, 1]],

    [[2, 1],
     [1, 2]]
])

result = torch.matmul(tensor1, tensor2)
print("Tensor 1:\n", tensor1)
print("\nTensor 2:\n", tensor2)
print("\nMatrix Multiplication Result:\n", result)