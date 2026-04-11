# Emotion-Detection-with-Convolutional-Neural-Networks


Semester project for the module **Neural Network and Memristive Hardware Accelerators (NNMHA)**  
Faculty of Electrical and Computer Engineering · TU Dresden

---

## What We Did

In this project, we built and trained deep learning models to detect human emotions from facial expressions using the **FER-2013 dataset**. The dataset contains 48×48 grayscale face images labeled across 7 emotion categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

### Steps covered:

1. **Dataset Exploration** – Visualized sample images and analyzed the class distribution. The dataset is imbalanced, with *Disgust* being heavily underrepresented.

2. **Data Preprocessing & Augmentation** – Applied normalization, random rotation, and flipping using `torchvision.transforms`.

3. **MLP Model** – Built and trained a simple Multilayer Perceptron as a baseline. Logged training/test loss and accuracy per epoch, plotted curves and confusion matrix.

4. **Custom CNN Model** – Built a 3-layer Convolutional Neural Network and trained it on the same data. Significant improvement over the MLP baseline.

5. **ResNet18** – Fine-tuned a pretrained ResNet18 from `torchvision` for emotion classification. Best performing model overall.

6. **Evaluation** – Computed Accuracy, Precision, Recall, and F1-Score for all models. Plotted and compared confusion matrices.

7. **Imbalanced Data Handling** – Addressed class imbalance using weighted loss / oversampling. Retrained and compared results.

8. **Webcam Demonstrator** – Built a real-time inference demo using OpenCV that detects faces from webcam input and predicts the emotion live.

9. **GradCAM (Optional)** – Applied Gradient-weighted Class Activation Mapping to visualize which facial regions the CNN and ResNet18 focus on when making predictions.

---

## Dataset

FER-2013 — available on Kaggle:  
🔗 https://www.kaggle.com/datasets/msambare/fer2013

---

## Tech Stack

- Python, PyTorch, torchvision
- OpenCV (webcam demonstrator)
- scikit-learn (metrics & confusion matrix)
- Matplotlib / Seaborn (plots)

---

