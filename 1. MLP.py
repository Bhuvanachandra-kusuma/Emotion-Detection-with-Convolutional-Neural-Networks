import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Increase font sizes for better visualization of plots
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16
})



# 1. SETUP AND DATA PREPARATION

# Path to training dataset
TRAIN_DIR = r'C:\Users\train'

# Path to testing dataset
TEST_DIR = r'C:\Users\test'

# Hyperparameters
BATCH_SIZE = 128        # Number of samples processed before updating weights
NUM_EPOCHS = 200        # Number of times the entire dataset is trained
LEARNING_RATE = 0.001   # Controls how much weights change during optimization


# Training transformations
train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale (1 channel)
    transforms.RandomHorizontalFlip(p=0.5),       # Randomly flip image horizontally
    transforms.RandomRotation(10),                # Randomly rotate image up to ±10 degrees
    transforms.ToTensor(),                        # Convert image to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])   # Normalize pixel values to range [-1, 1]
])

# Testing transformations
test_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Loading training dataset using folder structure
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)

# Loading testing dataset
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transforms)


# DataLoader loads dataset in batches
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Training data loaded with {len(train_dataset)} images.")
print(f"Test data loaded with {len(test_dataset)} images.")

# Store class names (emotion labels)
CLASS_NAMES = train_dataset.classes


# BEST PLOT SAVE FUNCTION

# Create directory to store best plots
os.makedirs("best_plots", exist_ok=True)

def save_best_plot(epoch, train_losses, test_losses, train_accs, test_accs):
    """
    Saves loss and accuracy curves whenever a new best model is found.
    """
    plt.figure(figsize=(10, 4))

    # Plot training and testing loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.title(f"Loss (Best @ Epoch {epoch})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    # Plot training and testing accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(test_accs, label="Test Acc")
    plt.title(f"Accuracy (Best @ Epoch {epoch})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"best_plots/best_plot_epoch_{epoch}.png", dpi=200)
    plt.close()



# 2. BUILDING THE MLP MODEL ---

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (Fully Connected Neural Network)
    Input: 48x48 grayscale image (flattened to 2304 values)
    Output: Number of emotion classes
    """
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),                     # Convert 2D image to 1D vector
            nn.Linear(input_size, 512),       # First hidden layer
            nn.ReLU(),                        # Activation function
            nn.Dropout(0.5),                  # Drop 50% neurons to prevent overfitting
            nn.Linear(512, 256),              # Second hidden layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, output_size)       # Output layer (one neuron per class)
        )

    def forward(self, x):
        return self.network(x)

# Input size = 48x48 image
INPUT_SIZE = 48 * 48

# Output size = number of emotion classes
OUTPUT_SIZE = len(CLASS_NAMES)

# Create model
model = MLP(INPUT_SIZE, OUTPUT_SIZE)

print("\nMLP Model Architecture:")
print(model)

# Loss function for multi-class classification
criterion = nn.CrossEntropyLoss()

# Adam optimizer (adaptive learning rate optimization)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# 3. TRAIN AND EVALUATE THE MODEL

# Lists to store training history
train_loss_history = []
test_loss_history = []
train_acc_history = []
test_acc_history = []
best_test_accuracy = 0.0

print("\n--- Starting Model Training ---")

for epoch in range(NUM_EPOCHS):

    # TRAINING
    model.train()  # Enable dropout

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in train_loader:
        optimizer.zero_grad()               # Clear previous gradients
        outputs = model(images)             # Forward pass
        loss = criterion(outputs, labels)   # Compute loss
        loss.backward()                     # Backpropagation
        optimizer.step()                    # Update weights

        running_loss += loss.item() * images.size(0)

        # Get predicted class
        _, predicted = torch.max(outputs.data, 1)

        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    # Compute training metrics
    epoch_train_loss = running_loss / total_samples
    epoch_train_acc = correct_predictions / total_samples

    train_loss_history.append(epoch_train_loss)
    train_acc_history.append(epoch_train_acc)

    # TESTING 
    model.eval()  # Disable dropout

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # No gradient calculation during testing
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    # Compute testing metrics
    epoch_test_loss = running_loss / total_samples
    epoch_test_acc = correct_predictions / total_samples

    test_loss_history.append(epoch_test_loss)
    test_acc_history.append(epoch_test_acc)

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
          f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
          f"Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_acc:.4f}")

    # Save best model based on test accuracy
    if epoch_test_acc > best_test_accuracy:
        best_test_accuracy = epoch_test_acc
        torch.save(model.state_dict(), 'best_mlp_model.pth')
        print(f"  -> New best model saved with accuracy: {best_test_accuracy:.4f}")

        save_best_plot(
            epoch + 1,
            train_loss_history,
            test_loss_history,
            train_acc_history,
            test_acc_history
        )
        print(f"  -> Best plot saved for epoch {epoch+1}")

print("--- Finished Training ---")


# 4. FINAL PLOTS

# Plot final loss and accuracy curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_loss_history, label='Training Loss')
plt.plot(test_loss_history, label='Test Loss')
plt.title('Loss Curves-MLP')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_acc_history, label='Training Accuracy')
plt.plot(test_acc_history, label='Test Accuracy')
plt.title('Accuracy Curves-MLP')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# 5. CONFUSION MATRIX

print("\n--- Generating Confusion Matrix ---")

# Load best saved model
model.load_state_dict(torch.load('best_mlp_model.pth'))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(labels.numpy())

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)

fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=45)
plt.title('Confusion Matrix-MLP')
plt.tight_layout()
plt.show()
