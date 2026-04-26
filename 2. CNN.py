
#Import libraries
import torch      
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from PIL import Image

# Set global font size for all plots
plt.rcParams.update({      
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16
})

# Folder paths for training and testing images
train_dir = r"C:\Users\Devika Jayaram\OneDrive\Desktop\Neural\Project\train"  
test_dir = r"C:\Users\Devika Jayaram\OneDrive\Desktop\Neural\Project\test"


# Transformations applied to test and train images
train_transform = transforms.Compose([        
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])                                      
test_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Loading datasets
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
test_dataset  = datasets.ImageFolder(test_dir, transform=test_transform)

# Creating batches
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

class_names = train_dataset.classes
num_classes = 7

#CNN model
class EmotionCNN(nn.Module):       
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            # Conv 1 → output: (32, 24, 24)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Conv 2 → output: (64, 12, 12)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Conv 3 → output: (128, 6, 6)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Dropout(0.25)
        )
        #Fully Connected Layers
        self.fc_layers = nn.Sequential(
            nn.Linear(128*6*6, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten tensor
        x = self.fc_layers(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   #Device setup
print("Using device:", device)

model = EmotionCNN().to(device)  # Initialize model
criterion = nn.CrossEntropyLoss() # Cross-entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam optimizer

epochs = 25

train_losses, test_losses = [], []
train_accs, test_accs = [], []

#TRAINING LOOP
for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_losses.append(running_loss / len(train_loader))
    train_accs.append(correct / total)

    #Testing Phase
    model.eval()
    test_loss, t_correct, t_total = 0, 0, 0   

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            t_total += labels.size(0)
            t_correct += predicted.eq(labels).sum().item()


     # Store test metrics
    test_losses.append(test_loss / len(test_loader))
    test_accs.append(t_correct / t_total)

    print(f"Epoch [{epoch+1}/{epochs}]  "
          f"Train Loss: {train_losses[-1]:.4f} | Train Acc: {train_accs[-1]:.4f}  "
          f"Test Loss: {test_losses[-1]:.4f} | Test Acc: {test_accs[-1]:.4f}")
    
    torch.save(model.state_dict(), f"cnn_epoch_{epoch+1}.pth")  # Save model checkpoint

#Loss curves
plt.figure(figsize=(8, 6))       
plt.plot(train_losses, label="Training Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.title("Loss curves - CNN", fontsize=16)
plt.legend(fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True)
plt.show()
 
#Accuracy curves
plt.figure(figsize=(8, 6))     
plt.plot(train_accs, label="Training Accuracy")
plt.plot(test_accs, label="Test Accuracy")
plt.xlabel("Epoch", fontsize=16)
plt.ylabel("Accuracy", fontsize=16)
plt.title("Accuracy curves - CNN", fontsize=16)
plt.legend(fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True)
plt.show()

#Confusion Matrix
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():

    for images, labels in test_loader:
        outputs = model(images.to(device))
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)     
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)

fig, ax = plt.subplots(figsize=(8, 8))

disp.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45,
    values_format="d"
)
ax.set_xlabel("Predicted Label", fontsize=16) # Axis labels and title
ax.set_ylabel("True Label", fontsize=16)
ax.set_title("Confusion Matrix - CNN", fontsize=16)
ax.tick_params(axis='both', labelsize=16)  # Tick font sizes
plt.tight_layout()
plt.show()




