# Retinal OCT Image Classification

## Overview

This project focuses on the classification of retinal Optical Coherence Tomography (OCT) images into eight different categories using deep learning. Two distinct models are explored: a custom Atrous Convolutional Classifier (V1) and a DenseNet-based classifier (V2). The goal is to develop and evaluate accurate models for diagnosing various retinal conditions from OCT scans.

## Dataset

The dataset used in this project is the "Retinal OCT C-8" dataset. It consists of a large collection of OCT images categorized into the following eight classes:
- Age-related Macular Degeneration (AMD)
- Choroidal Neovascularization (CNV)
- Central Serous Retinopathy (CSR)
- Diabetic Macular Edema (DME)
- Diabetic Retinopathy (DR)
- DRUSEN
- Macular Hole (MH)
- NORMAL

## Methodology

The project follows a standard deep learning workflow:
1. **Data Loading and Preprocessing**: Images are loaded from the dataset directories and transformed using `torchvision.transforms`. This includes resizing, random horizontal flipping, random rotation, and normalization.
2. **Model Architecture**: Two different convolutional neural network (CNN) architectures are implemented and trained.
3. **Training and Validation**: The models are trained using the training set and evaluated on the validation set at the end of each epoch to monitor performance and prevent overfitting. The best-performing model based on validation loss is saved.
4. **Testing**: The final model is evaluated on the test set to assess its generalization capabilities.
5. **Visualization**: Training and validation loss and accuracy are plotted over epochs to visualize the learning process. For explainability of the model Grad-CAM visualization is performed on test images.

## Models

### V1: Atrous Convolutional Classifier

This version implements a custom CNN architecture that utilizes atrous (dilated) convolutions. The model consists of several `AtrousConvBlock` layers with increasing dilation rates, followed by fully connected layers for classification.

- **Key Features**:
  - Atrous convolutions to capture multi-scale contextual information.
  - Max pooling for down-sampling.
  - Dropout for regularization.

### V2: DenseNet Classifier

This version leverages the power of transfer learning by using a pre-trained `DenseNet121` model from the `monai` library. The model is adapted for the 8-class classification task.

- **Key Features**:
  - DenseNet121 architecture, known for its feature reuse and strong performance.
  - Pre-trained on a large dataset, allowing for effective transfer learning.

## Results

| Model                    | Test Accuracy |
| ------------------------ | :-----------: |
| Atrous Classifier (V1)   |    90.5%      |
| DenseNet Classifier (V2) |   96.46%      |

Both models achieve high accuracy on the test set, with the DenseNet classifier showing a performance advantage. The training and validation curves demonstrate effective learning and generalization for both architectures.
