import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1,16,3)
        self.conv2 = nn.Conv2d(16,32,3)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(32*5*5,128)
        self.fc2 = nn.Linear(128,10)

    def forward(self,x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1,32*5*5)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(5):
    running_loss = 0
    for images,labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

# Testing
correct = 0
total = 0
with torch.no_grad():
    for images,labels in test_loader:
        outputs = model(images)
        _,predicted = torch.max(outputs,1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()

print("Test Accuracy:",100*correct/total)

import matplotlib.pyplot as plt

dataiter = iter(test_loader)
images, labels = next(dataiter)

outputs = model(images)
_, predicted = torch.max(outputs,1)

plt.figure(figsize=(10,4))

for i in range(6):
    plt.subplot(1,6,i+1)
    plt.imshow(images[i].squeeze(), cmap="gray")
    plt.title(f"P:{predicted[i].item()} T:{labels[i].item()}")
    plt.axis("off")

plt.show()