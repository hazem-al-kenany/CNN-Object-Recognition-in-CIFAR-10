# Grayscale CIFAR-10 Image Classification with CNN

This project implements a convolutional neural network (CNN) for image classification on the CIFAR-10 dataset, converted to grayscale. The model incorporates advanced preprocessing techniques, data augmentation, and custom image predictions to enhance classification accuracy.

---

## Features

### Dataset
- **CIFAR-10**:
  - Contains 60,000 32x32 images across 10 classes: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, and `truck`.
  - Training set: 50,000 images.
  - Test set: 10,000 images.

### Preprocessing
- **Grayscale Conversion**:
  - CIFAR-10 images are converted from RGB to grayscale using OpenCV.
- **Histogram Equalization**:
  - Enhances image contrast for better feature extraction.
- **Custom Image Handling**:
  - Allows predictions on unseen grayscale images.
- **Data Normalization**:
  - Normalizes pixel values to a range of 0 to 1.

### Data Augmentation
- **Techniques Used**:
  - Random rotation (15 degrees).
  - Horizontal and vertical shifts (10%).
  - Horizontal flipping.

### Model Architecture
- **Custom CNN**:
  - Input layer: Grayscale images (`32x32x1`).
  - Two convolutional layers with `ReLU` activation and max pooling.
  - Fully connected (dense) layers with dropout for regularization.
  - Output layer: Softmax activation for 10 classes.

### Training
- **Learning Rate Scheduler**:
  - Adjusts the learning rate dynamically after 10 epochs.
- **Early Stopping**:
  - Stops training when validation loss plateaus for 5 epochs.
- **Class Weights**:
  - Balances class distribution in the training set.

### Evaluation
- **Metrics**:
  - Accuracy and loss for both training and validation.
- **Confusion Matrix**:
  - Visualizes model performance on the test set.
- **Classification Report**:
  - Detailed performance metrics (precision, recall, F1-score) for each class.

### Predictions
- **Custom Image Predictions**:
  - Predicts class labels for user-supplied images with confidence scores.

---

## Code Structure

### Main Components
- **Data Preprocessing**:
  - Converts CIFAR-10 images to grayscale.
  - Enhances contrast using histogram equalization.
  - Augments training data.
- **Model Definition**:
  - Builds a CNN with two convolutional layers, dropout, and dense layers.
  - Configures model compilation with Adam optimizer.
- **Training and Validation**:
  - Splits the training set into training and validation subsets.
  - Trains using augmented data and monitors validation performance.
- **Evaluation**:
  - Tests the model on unseen data and generates evaluation metrics.
- **Custom Predictions**:
  - Predicts class labels for new images.

---

## How to Run

### Prerequisites
- Python 3.7 or higher.
- Required Libraries:
  ```bash
  pip install tensorflow numpy matplotlib scikit-learn opencv-python
