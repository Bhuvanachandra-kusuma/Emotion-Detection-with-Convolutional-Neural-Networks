Real-Time Emotion Detection Using (CNN/RestNet18)
==========================================

Overview
--------
This project implements a real-time emotion detection system using a pretrained models with PyTorch. 
The system captures video from a webcam, detects faces using OpenCV's Haar cascades, and predicts the emotion 
of each detected face. The predicted emotion along with the confidence score is displayed on the video feed in real-time.

Features
--------
- Detects faces in real-time from webcam feed.
- Classifies facial emotions using a trained models.
- Supports multiple classes (as defined in the checkpoint).
- Displays emotion labels with confidence on the video frames.
- GPU accelerated (if CUDA is available).

Requirements
------------
- Python 3.8+
- PyTorch
- Torchvision
- OpenCV
- NumPy

Install dependencies with:
pip install torch torchvision opencv-python numpy

Files
-----
- emotion_detector.py      – Main Python script for real-time detection.
- best_checkspoint.pth     – Pretrained models checkpoint.
- README.txt               – Project documentation.

Usage
-----
1. Ensure you have a working webcam connected.
2. Run the script:
   python emotion_detector.py
3. Controls:
   - Press 'q' to quit the application.

Model & Transforms
------------------
- Model: - Outsput: Fully connected layer matches the number of emotion classes.
- Input transforms (must match training transforms):
  - Convert image to PIL format.
  - Convert to grayscale with 3 channels.
  - Resize to 48x48.
  - Convert to Tensor.
  - Normalize with [0.5, 0.5, 0.5] mean and [0.5, 0.5, 0.5] std.

Notes
-----
- Ensure the best_checkspoint.pth checkpoint file is in the same directory as the script.

Troubleshooting
---------------
- Webcam not opening: Make sure no other application is using the webcam.
- Transform errors: Ensure detected faces are not empty (face_img.size > 0) and are properly cropped.
- CUDA errors: If you don’t have a GPU, the script will automatically use CPU.


