Emotion Classification Using ResNet18 (Pretrained)
==========================================

Overview
--------
This project implements emotion classification using a ResNet18 model initialized with ImageNet pretrained weights. 
Transfer learning is used to improve performance and convergence speed on the FER2013 dataset. 
Training and evaluation metrics are generated for analysis.

Features
--------
- ResNet18 with ImageNet pretrained weights.
- Transfer learning approach.
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
- resnet18_pretrained.py           – Training script using pretrained ResNet18.
- resnet18_pretrained_epoch_*.pth  – Saved model checkpoints.
- README.txt                       – Project documentation.

Usage
-----
1. Place the FER2013 dataset in the following structure:
   train/
   test/
2. Run the script:
   python resnet18_pretrained.py
3. After training:
   - Loss and accuracy curves will be displayed.
   - Confusion matrix will be generated.
   - Model checkpoints will be saved per epoch.

Model & Transforms
------------------
- Model: ResNet18 initialized with ImageNet pretrained weights.
- Final fully connected layer replaced to match emotion classes.
- Input transforms:
  - Convert to grayscale with 3 channels.
  - Resize to 224x224.
  - Random horizontal flip and rotation (training).
  - Convert to Tensor.
  - Normalize with ImageNet mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225].

Notes
-----
- Pretrained weights significantly improve convergence speed and test accuracy compared to training from scratch.

Troubleshooting
---------------
- Pretrained weights not downloading: Check internet connection.
- CUDA errors: Script automatically switches to CPU if GPU is unavailable.
- Dataset path errors: Verify directory structure.
