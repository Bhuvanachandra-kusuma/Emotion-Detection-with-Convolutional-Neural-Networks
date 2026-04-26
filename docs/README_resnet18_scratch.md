Emotion Classification Using ResNet18 (From Scratch)
==========================================

Overview
--------
This project implements emotion classification using a ResNet18 model trained from scratch with PyTorch. 
The model is trained on the FER2013 dataset and learns all feature representations without using pretrained weights. 
Training and evaluation metrics such as loss, accuracy, and confusion matrix are generated for performance analysis.

Features
--------
- ResNet18 architecture trained from scratch.
- Custom weight initialization (He and Xavier).
- Grayscale input (1-channel).
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
- resnet18_scratch.py      – Training script for ResNet18 from scratch.
- resnet18_epoch20.pth     – Saved trained model weights.
- README.txt               – Project documentation.

Usage
-----
1. Place the FER2013 dataset in the following structure:
   train/
   test/
2. Run the script:
   python resnet18_scratch.py
3. After training:
   - Loss and accuracy curves will be displayed.
   - Confusion matrix will be generated.
   - Model weights will be saved.

Model & Transforms
------------------
- Model: ResNet18 initialized without pretrained weights.
- First convolution layer modified to accept 1-channel input.
- Fully connected layer adjusted to match number of emotion classes.
- Input transforms:
  - Resize to 224x224.
  - Convert to grayscale (1 channel).
  - Random horizontal flip (training only).
  - Convert to Tensor.
  - Normalize with mean [0.5] and std [0.5].

Notes
-----
- Since the model is trained from scratch, training may take longer and require more epochs for convergence.

Troubleshooting
---------------
- CUDA not available: The script automatically runs on CPU.
- Path errors: Ensure train/ and test/ folders exist in the correct location.
- Memory issues: Reduce batch size if GPU memory is insufficient.
