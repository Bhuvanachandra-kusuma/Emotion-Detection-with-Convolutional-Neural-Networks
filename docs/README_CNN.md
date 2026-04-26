Facial Emotion Recognition using CNN 

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify facial images into seven different emotion categories. The system performs image preprocessing, model training, evaluation, and visualization of results including loss curves, accuracy curves, and a confusion matrix.

1. Project Overview

The goal of this project is to build a deep learning model capable of recognizing human emotions from grayscale facial images. The CNN is trained using labeled image datasets and evaluated using multiple performance metrics.

The project includes:

Image preprocessing and augmentation

Custom CNN architecture

Training and testing pipeline

Performance visualization

Confusion matrix analysis

Model checkpoint saving

2. Dataset Structure

The dataset must be organized as follows:

Project/
│
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
│
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/


Each folder contains facial images corresponding to one emotion class.

3. Model Architecture
Input

Grayscale images resized to 48 × 48 pixels

Convolution Layers
Layer	Output Channels	Operation
Conv1	32	Conv → ReLU → MaxPool
Conv2	64	Conv → ReLU → MaxPool
Conv3	128	Conv → ReLU → MaxPool
Fully Connected Layers

128 × 6 × 6 → 512 → 7

Dropout (0.25) for regularization

Other Settings

Loss Function: CrossEntropyLoss

Optimizer: Adam (lr = 0.001)

Epochs: 25

4. Requirements

Install the required Python libraries:

pip install torch torchvision matplotlib scikit-learn numpy pillow

5. How to Run the Project
a. Set Dataset Paths

Modify the following lines in the code:

train_dir = "path_to_train_folder"
test_dir = "path_to_test_folder"

b. Run the Script
python emotion_cnn.py

c. Training Output

During training, the following will be displayed:

Training loss and accuracy

Testing loss and accuracy

Model checkpoints saved per epoch

Example output:

Epoch [1/25] Train Loss: 0.85 | Train Acc: 0.72 | Test Acc: 0.69

6. Results Visualization

The program automatically generates:

Loss Curve-Shows training and testing loss over epochs.

Accuracy Curve-Displays training and testing accuracy progression.

Confusion Matrix-Illustrates class-wise prediction performance.

These plots help in analyzing model learning behavior and classification quality.

7. Model Saving and Loading
Saving (Automatic)

The model is saved after every epoch:

cnn_epoch_1.pth
cnn_epoch_2.pth
...

Loading (Manual)
model.load_state_dict(torch.load("cnn_epoch_25.pth"))
model.eval()

8. Performance Evaluation

The model is evaluated using:

Classification Accuracy

Cross-Entropy Loss

Confusion Matrix

These metrics help understand both overall and per-class performance.



