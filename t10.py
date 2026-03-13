#MSE

import torch
import torch.nn as nn

criterion = nn.MSELoss()

y_pred = torch.tensor([3.0])
y_true = torch.tensor([5.0])

loss = criterion(y_pred, y_true)

print(loss)

#BSE
import torch
import torch.nn as nn

criterion = nn.BCELoss()

y_pred = torch.tensor([0.9])
y_true = torch.tensor([1.0])

loss = criterion(y_pred, y_true)

print(loss)


#Cross Entropy
import torch
import torch.nn as nn

criterion = nn.CrossEntropyLoss()

pred = torch.tensor([[2.0, 1.0, 0.1]])
target = torch.tensor([0])

loss = criterion(pred, target)

print(loss)
