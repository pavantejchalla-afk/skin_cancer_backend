import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# Dataset path
data_dir = "dataset"

# Smaller image transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Load dataset
dataset = datasets.ImageFolder(
    data_dir,
    transform=transform
)

# Smaller batch size
train_loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

# Load MobileNetV2
model = models.mobilenet_v2(weights="DEFAULT")

# Modify classifier
model.classifier[1] = nn.Linear(
    model.last_channel,
    len(dataset.classes)
)

# Use CPU
device = torch.device("cpu")

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 2

for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        # Print progress
        if i % 100 == 0:
            print(f"Epoch {epoch+1}, Batch {i}")

    print(f"Epoch {epoch+1} completed")

# Save model
os.makedirs("models", exist_ok=True)

torch.save(
    model.state_dict(),
    "models/best_model.pth"
)

print("Model saved successfully!")