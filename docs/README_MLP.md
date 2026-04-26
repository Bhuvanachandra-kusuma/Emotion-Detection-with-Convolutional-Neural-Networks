FACIAL EMOTION RECOGNITION USING MULTI-LAYER PERCEPTRON (MLP)
==============================================================

1. Project Overview
-------------------
This project implements a Facial Emotion Recognition (FER) system using a Multi-Layer Perceptron (MLP) built with PyTorch.

The model classifies grayscale facial images of size 48x48 pixels into different emotion categories using the FER2013 dataset.

The pipeline includes:
- Data preprocessing
- Data augmentation
- Model training and evaluation
- Saving best model checkpoint
- Visualization of loss, accuracy, and confusion matrix

--------------------------------------------------------------

2. Python Version
-----------------
This project was developed and tested using Python 3.10.

Check your Python version:
python --version

--------------------------------------------------------------

3. Dataset Information
---------------------
Dataset: FER2013 (Facial Expression Recognition 2013)

Emotion Classes:
1. Angry
2. Disgust
3. Fear
4. Happy
5. Sad
6. Surprise
7. Neutral

--------------------------------------------------------------

4. Dataset Download
------------------
Download FER2013 dataset from:

- Kaggle: https://www.kaggle.com/datasets/msambare/fer2013

Note: You need a Kaggle account to download the dataset.

--------------------------------------------------------------

5. Dataset Structure
-------------------
After extracting, organize your dataset as follows:

archive/
    train/
        angry/
        disgust/
        fear/
        happy/
        sad/
        surprise/
        neutral/
    test/
        angry/
        disgust/
        fear/
        happy/
        sad/
        surprise/
        neutral/

Each folder corresponds to one emotion class.
Images should be grayscale, 48x48 pixels.

--------------------------------------------------------------

6. Required Libraries
---------------------
Install required Python libraries:

pip install torch torchvision matplotlib numpy scikit-learn

--------------------------------------------------------------

7. Model Architecture
---------------------
Input: 48x48 grayscale image (flattened to 2304 features)

Hidden Layer 1:
- Linear (2304 -> 512)
- ReLU Activation
- Dropout (0.5)

Hidden Layer 2:
- Linear (512 -> 256)
- ReLU Activation
- Dropout (0.5)

Output Layer:
- Linear (256 -> Number of Classes)

Loss Function: CrossEntropyLoss
Optimizer: Adam

--------------------------------------------------------------

8. Hyperparameters
------------------
Batch Size       : 128
Epochs           : 200
Learning Rate    : 0.001
Dropout Rate     : 0.5

--------------------------------------------------------------

9. Data Preprocessing
---------------------
Training Transformations:
- Convert to Grayscale
- Random Horizontal Flip (50% probability)
- Random Rotation (±10 degrees)
- Convert to Tensor
- Normalize pixel values to [-1, 1]

Testing Transformations:
- Convert to Grayscale
- Convert to Tensor
- Normalize pixel values

Data augmentation improves generalization and reduces overfitting.

--------------------------------------------------------------

10. Training Process
-------------------
For each epoch:
1. Forward Pass
2. Loss Calculation
3. Backpropagation
4. Weight Update
5. Accuracy Computation

The model with the highest test accuracy is saved automatically:
best_mlp_model.pth

Best performance plots are saved inside:
best_plots/

--------------------------------------------------------------

11. Outputs Generated
--------------------
- Training vs Testing Loss Curves
- Training vs Testing Accuracy Curves
- Confusion Matrix
- Saved best model checkpoint

Confusion matrix helps analyze which emotion classes are misclassified.

--------------------------------------------------------------

12. How to Run the Project
--------------------------
Step 1: Install Python 3.10
Download from: https://www.python.org/downloads/

Step 2: Install Dependencies
pip install torch torchvision matplotlib numpy scikit-learn

Step 3: Update Dataset Paths in Script
TRAIN_DIR = "path_to_train_folder"
TEST_DIR  = "path_to_test_folder"

Step 4: Run the Script
python your_script_name.py




13.Technologies Used
--------------------
- Python 3.10
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Scikit-learn


