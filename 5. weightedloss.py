# --------------------------------------------------
# 1. IMPORT REQUIRED LIBRARIES
# --------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from collections import Counter


# --------------------------------------------------
# 2. DEVICE CONFIGURATION
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# --------------------------------------------------
# 3. IMAGE TRANSFORMATIONS
# --------------------------------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),   # Convert to 3-channel grayscale
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],               # ImageNet mean
        std=[0.229, 0.224, 0.225]                 # ImageNet std
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# 4. DATASET LOADING
# --------------------------------------------------
train_dir = "train"
test_dir  = "test"

train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
test_dataset  = datasets.ImageFolder(test_dir, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

num_classes = len(train_dataset.classes)
class_names = train_dataset.classes

print("Classes:", class_names)


# --------------------------------------------------
# 5. COMPUTE CLASS WEIGHTS
# --------------------------------------------------
# Count samples per class
class_counts = Counter(train_dataset.targets)
total_samples = len(train_dataset)

# Inverse frequency weighting
class_weights = [
    total_samples / class_counts[i] for i in range(num_classes)
]

# Convert to tensor and move to device
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

print("Class weights:", class_weights)


# --------------------------------------------------
# 6. LOAD PRETRAINED RESNET18
# --------------------------------------------------
resnet18 = models.resnet18(
    weights=models.ResNet18_Weights.IMAGENET1K_V1
)

# Replace final fully connected layer
resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)

resnet18 = resnet18.to(device)

print(next(resnet18.parameters()).device)


# --------------------------------------------------
# 7. LOSS FUNCTION WITH CLASS WEIGHTS
# --------------------------------------------------
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(resnet18.parameters(), lr=1e-4)

epochs = 20

train_losses, test_losses = [], []
train_accs, test_accs = [], []


# --------------------------------------------------
# 8. TRAINING AND EVALUATION LOOP
# --------------------------------------------------
for epoch in range(epochs):

    # -------- TRAINING --------
    resnet18.train()
    running_loss, correct, total = 0, 0, 0

    for images, labels in train_loader:

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = resnet18(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc  = correct / total

    train_losses.append(train_loss)
    train_accs.append(train_acc)


    # -------- TESTING --------
    resnet18.eval()
    running_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in test_loader:

            images, labels = images.to(device), labels.to(device)

            outputs = resnet18(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

    test_loss = running_loss / len(test_loader)
    test_acc  = correct / total

    test_losses.append(test_loss)
    test_accs.append(test_acc)

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
          f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")


# --------------------------------------------------
# 9. LOSS CURVES
# --------------------------------------------------
plt.figure(figsize=(8,6))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curves - ResNet18 (Pretrained, Weighted Loss)")
plt.legend()
plt.grid(True)
plt.show()


# --------------------------------------------------
# 10. ACCURACY CURVES
# --------------------------------------------------
plt.figure(figsize=(8,6))
plt.plot(train_accs, label="Train Accuracy")
plt.plot(test_accs, label="Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy Curves - ResNet18 (Pretrained, Weighted Loss)")
plt.legend()
plt.grid(True)
plt.show()


# --------------------------------------------------
# 11. CONFUSION MATRIX
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
plt.title("Confusion Matrix - ResNet18 (Pretrained, Weighted Loss)")

plt.setp(ax.get_xticklabels(), rotation=45, 
         ha="right", rotation_mode="anchor")

plt.tight_layout()
plt.show()
