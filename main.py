import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

data_dir = "chest-xray-pneumonia/versions/2/chest_xray"  # contains train/val/test folders

train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # converts grayscale -> 3 channels
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # converts grayscale -> 3 channels
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_data = datasets.ImageFolder(f"{data_dir}/train", transform=train_transforms)
val_data = datasets.ImageFolder(f"{data_dir}/val", transform=val_transforms)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False  # freeze base layers

# Replace final layer for 2 classes
model.fc = nn.Linear(model.fc.in_features, 2)
#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

print("Beginning training...")
for epoch in range(5):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss {running_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "pneumonia_resnet18old.pth")



