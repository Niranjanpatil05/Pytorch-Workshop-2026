import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ---- Create random image tensor ----
img_tensor = torch.rand(1, 3, 128, 128)   # [batch, channels, height, width]

# ---- Define CNN layers ----
conv1 = nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1)
relu = nn.ReLU()
pool = nn.MaxPool2d(2, 2)
conv2 = nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1)

# ---- Pass image through CNN ----
with torch.no_grad():
    x1 = conv1(img_tensor)
    x2 = relu(x1)
    x3 = pool(x2)
    x4 = conv2(x3)
    x5 = relu(x4)
    x6 = pool(x5)
    x_flat = x6.view(1, -1)

# ---- Print size reductions ----
print("------ CNN Feature Map Sizes ------")
print("Input Image:", img_tensor.shape)
print("After Conv1:", x1.shape)
print("After Pool1:", x3.shape)
print("After Conv2:", x4.shape)
print("After Pool2:", x6.shape)
print("Flattened:", x_flat.shape)

# ---- Visualization function ----
def show_feature_maps(tensor, title, num_channels=8):
    fig, axes = plt.subplots(1, num_channels, figsize=(15,5))
    for i in range(num_channels):
        axes[i].imshow(tensor[0, i].detach().cpu(), cmap='viridis')
        axes[i].axis('off')
    fig.suptitle(title)
    plt.show()

# ---- Show feature maps ----
show_feature_maps(x1, "After Conv1")
show_feature_maps(x2, "After ReLU1")
show_feature_maps(x3, "After Pool1")
show_feature_maps(x4, "After Conv2")
show_feature_maps(x5, "After ReLU2")
show_feature_maps(x6, "After Pool2")