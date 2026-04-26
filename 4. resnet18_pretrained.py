# --------------------------------------------------
# 1. IMPORT REQUIRED LIBRARIES
# --------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import seaborn as sns


# --------------------------------------------------
# 2. IMAGE TRANSFORMATIONS
# --------------------------------------------------
# Training image transformations
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel grayscale
    transforms.Resize((224, 224)),                # Resize to 224x224 (ResNet input size)
    transforms.RandomHorizontalFlip(),            # Data augmentation
    transforms.RandomRotation(10),                # Small rotation augmentation
    transforms.ToTensor(),                        # Convert image to tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],               # ImageNet mean
        std=[0.229, 0.224, 0.225]                 # ImageNet standard deviation
    )
])

# Testing image transformations (no augmentation)
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# 3. DATASET LOADING
# --------------------------------------------------
train_dir = "train"
test_dir  = "test"

train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
test_dataset  = datasets.ImageFolder(test_dir, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

class_names = train_dataset.classes
num_classes = 7


# --------------------------------------------------
# 4. DEVICE CONFIGURATION
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# --------------------------------------------------
# 5. LOAD PRETRAINED RESNET18
# --------------------------------------------------
# Load ResNet18 with ImageNet pretrained weights
resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Replace final fully connected layer for custom classification
resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)

# Move model to selected device
resnet18 = resnet18.to(device)


# --------------------------------------------------
# 6. LOSS FUNCTION AND OPTIMIZER
# --------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet18.parameters(), lr=1e-4)

epochs = 20

train_losses, test_losses = [], []
train_accs, test_accs = [], []


# --------------------------------------------------
# 7. TRAINING AND EVALUATION LOOP
# --------------------------------------------------
for epoch in range(epochs):

    # -------- TRAINING PHASE --------
    resnet18.train()
    running_loss, correct, total = 0, 0, 0

    for images, labels in train_loader:

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()            # Clear gradients
        outputs = resnet18(images)       # Forward pass
        loss = criterion(outputs, labels)

        loss.backward()                  # Backpropagation
        optimizer.step()                 # Update weights

        running_loss += loss.item()

        _, preds = outputs.max(1)        # Predicted class
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

    train_losses.append(running_loss / len(train_loader))
    train_accs.append(correct / total)


    # -------- TESTING PHASE --------
    resnet18.eval()
    test_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in test_loader:

            images, labels = images.to(device), labels.to(device)

            outputs = resnet18(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

    test_losses.append(test_loss / len(test_loader))
    test_accs.append(correct / total)

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Train Loss: {train_losses[-1]:.4f} | Train Acc: {train_accs[-1]:.4f} "
          f"Test Loss: {test_losses[-1]:.4f} | Test Acc: {test_accs[-1]:.4f}")

    # Save model after each epoch
    torch.save(resnet18.state_dict(), 
               f"resnet18_pretrained_epoch_{epoch+1}.pth")


# --------------------------------------------------
# 8. PLOT LOSS CURVES
# --------------------------------------------------
plt.figure(figsize=(8,6))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curves - ResNet18 (Pretrained)")
plt.legend()
plt.grid(True)
plt.show()


# --------------------------------------------------
# 9. PLOT ACCURACY CURVES
# --------------------------------------------------
plt.figure(figsize=(8,6))
plt.plot(train_accs, label="Train Accuracy")
plt.plot(test_accs, label="Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy Curves - ResNet18 (Pretrained)")
plt.legend()
plt.grid(True)
plt.show()


# --------------------------------------------------
# 10. CONFUSION MATRIX
# --------------------------------------------------
resnet18.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = resnet18(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8,6))
ax = sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - ResNet18 (Pretrained)")

# Rotate x-axis labels for readability
plt.setp(ax.get_xticklabels(), rotation=45, 
         ha="right", rotation_mode="anchor")

plt.tight_layout()
plt.show()
