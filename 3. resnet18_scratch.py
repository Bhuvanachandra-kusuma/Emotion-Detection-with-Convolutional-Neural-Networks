# --------------------------------------------------
# 1. IMPORT REQUIRED LIBRARIES
# --------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# --------------------------------------------------
# 2. DEVICE CONFIGURATION
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------------------------------
# 3. DATASET PATHS (UPDATE TO YOUR LOCAL PATH)
# --------------------------------------------------
train_dir = r"C:\YourPath\train"
test_dir  = r"C:\YourPath\test"

# --------------------------------------------------
# 4. IMAGE TRANSFORMATIONS
# --------------------------------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# --------------------------------------------------
# 5. DATA LOADING
# --------------------------------------------------
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
test_dataset  = datasets.ImageFolder(test_dir, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

class_names = train_dataset.classes
num_classes = 7

# --------------------------------------------------
# 6. RESNET18 MODEL (FROM SCRATCH)
# --------------------------------------------------
resnet18 = models.resnet18(weights=None)

# Modify first convolution layer for grayscale input
resnet18.conv1 = nn.Conv2d(
    1, 64, kernel_size=7, stride=2, padding=3, bias=False
)

# Modify final fully connected layer
resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)

resnet18 = resnet18.to(device)

# --------------------------------------------------
# 7. CUSTOM WEIGHT INITIALIZATION
# --------------------------------------------------
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)

resnet18.apply(init_weights)

# --------------------------------------------------
# 8. LOSS FUNCTION AND OPTIMIZER
# --------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet18.parameters(), lr=1e-4)

epochs = 20

train_losses, test_losses = [], []
train_accs, test_accs = [], []

# --------------------------------------------------
# 9. TRAINING & EVALUATION LOOP
# --------------------------------------------------
for epoch in range(epochs):

    # ----- TRAINING -----
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

    # ----- TESTING -----
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
# 10. SAVE TRAINED MODEL
# --------------------------------------------------
torch.save(resnet18.state_dict(), "resnet18_epoch20.pth")
print("Model saved successfully.")

# --------------------------------------------------
# 11. PLOT LOSS CURVES
# --------------------------------------------------
plt.figure(figsize=(8,6))
plt.plot(train_losses, label="Training Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curves - ResNet18")
plt.legend()
plt.grid(True)
plt.show()

# --------------------------------------------------
# 12. PLOT ACCURACY CURVES
# --------------------------------------------------
plt.figure(figsize=(8,6))
plt.plot(train_accs, label="Training Accuracy")
plt.plot(test_accs, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curves - ResNet18")
plt.legend()
plt.grid(True)
plt.show()

# --------------------------------------------------
# 13. CONFUSION MATRIX
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
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - ResNet18")
plt.tight_layout()
plt.show()
