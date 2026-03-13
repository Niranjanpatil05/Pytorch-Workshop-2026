import torch
import torch.nn as nn

class MyNeuralNetwork(nn.Module):



      def __init__(self):

          # Call parent constructor
          super(MyNeuralNetwork, self).__init__()

          # Define layers
          self.fc1 = nn.Linear(4, 8)   # Input layer
          self.fc2 = nn.Linear(8, 3)   # Output layer


      def forward(self, x):

          # Input → fc1
          x = self.fc1(x)

          # Apply activation function
          x = torch.relu(x)

          # Pass to fc2
          x = self.fc2(x)

          # Return output
          return x


  # ------------------------------------------
  # Creating Model Object
  # ------------------------------------------

model = MyNeuralNetwork()

  # Print model architecture
print("Model Architecture:")
print(model)

  # Create random input
x = torch.rand(1, 4)

print("\nInput Data:")
print(x)

  # Forward pass
output = model(x)

print("\nModel Output:")
print(output)
print("\nModel Parameters:")

for name, param in model.named_parameters():
      print(name, param.shape)

  # Training mode
model.train()
print("\nModel set to Training Mode")

  # Evaluation mode
model.eval()
print("Model set to Evaluation Mode")

torch.save(model.state_dict(), "model.pth")
print("\nModel saved successfully")

model.load_state_dict(torch.load("model.pth"))
print("Model loaded successfully")

class BinaryClassifier(nn.Module):

      def __init__(self):
          super(BinaryClassifier, self).__init__()

          self.fc1 = nn.Linear(2, 4)
          self.fc2 = nn.Linear(4, 1)

      def forward(self, x):

          x = torch.relu(self.fc1(x))

          # Sigmoid output for binary classification
          x = torch.sigmoid(self.fc2(x))

          return x


  # Create classifier
binary_model = BinaryClassifier()

print("\nBinary Classification Model:")
print(binary_model)

  # Example input
sample = torch.tensor([[0.5, 0.8]])

  # Prediction
prediction = binary_model(sample)

print("\nBinary Classification Output:")
print(prediction)
print(output)  # Output: tensor([0.0000, 1.3000, 0.0000, 4.2000])
