Emotion Classification Using ResNet18 (Pretrained + Weighted Loss)
==========================================

Overview
--------
This project implements emotion classification using a pretrained ResNet18 model combined with a weighted cross-entropy loss function. 
Class weights are computed based on inverse class frequency to address dataset imbalance in FER2013. 
This approach improves minority class learning during training.

Features
--------
- ResNet18 with ImageNet pretrained weights.
- Weighted CrossEntropyLoss for class imbalance handling.
- Automatic computation of class weights.
- 3-channel grayscale input.
- ImageNet normalization.
- Training and testing loss tracking.
- Accuracy evaluation per epoch.
- Confusion matrix visualization.
- GPU accelerated (if CUDA is available).

Requirements
------------
- Python 3.8+
- PyTorch
- Torchvision
- Matplotlib
- NumPy
- Seaborn
- Scikit-learn

Install dependencies with:
pip install torch torchvision matplotlib numpy seaborn scikit-learn

Files
-----
- resnet18_weighted_loss.py  – Training script using weighted loss.
- resnet18_pretrained_epoch_*.pth – Saved model checkpoints.
- README.txt                  – Project documentation.

Usage
-----
1. Place the FER2013 dataset in the following structure:
   train/
   test/
2. Run the script:
   python resnet18_weighted_loss.py
3. After training:
   - Loss and accuracy curves will be displayed.
   - Confusion matrix will be generated.
   - Model checkpoints will be saved.

Model & Transforms
------------------
- Model: ResNet18 with ImageNet pretrained weights.
- Final fully connected layer adjusted to match number of emotion classes.
- Loss: CrossEntropyLoss with class weights.
- Class weights computed as:
  total_samples / samples_per_class
- Input transforms:
  - Resize to 224x224.
  - Convert to grayscale with 3 channels.
  - Random horizontal flip.
  - Convert to Tensor.
  - Normalize with ImageNet mean and std.

Notes
-----
- Weighted loss helps reduce bias toward majority classes.
- Especially useful when class distribution is highly imbalanced.

Troubleshooting
---------------
- Incorrect class weights: Ensure dataset labels are loaded correctly.
- CUDA errors: Script runs on CPU if GPU is unavailable.
- High loss instability: Verify normalization and learning rate settings.
