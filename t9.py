import torch
x = torch.tensor (3.0 , requires_grad = True )
# Normal operation ( gradients tracked )
y1 = x**2
print (f" Normal operation - y1. grad_fn : {y1. grad_fn }")

# Inside no_grad context ( gradients not tracked )

with torch . no_grad ():
    y2 = x**2
    print (f" Inside no_grad - y2. grad_fn : {y2. grad_fn }") # None

# Back to normal ( gradients tracked again )
y3 = x**2
print (f" After no_grad - y3. grad_fn : {y3. grad_fn }")

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
