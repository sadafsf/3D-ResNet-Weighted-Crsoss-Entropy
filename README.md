
# Predicting Student Engagement with 3D ResNet-18 and Weighted Cross-Entropy Loss

This repository contains the code and methodologies from the paper **"Predicting Student Engagement with 3D ResNet-18 and Weighted Cross-Entropy Loss"**, presented at the 35th Canadian Conference on Artificial Intelligence. The study explores detecting student engagement levels in online learning environments using deep learning.

## Overview

The COVID-19 pandemic significantly increased the adoption of online learning, making it critical to assess and enhance student engagement in virtual classrooms. This project uses videos from the **DAiSEE dataset** to classify student engagement into four levels: *very low*, *low*, *high*, and *very high*.

We implemented a **3D ResNet-18 model** with weighted cross-entropy loss and a weighted random sampler to address class imbalance in the dataset. Our method achieved a classification accuracy of 51.82% on the four-class problem.

---

## Repository Structure and Files

### Main Folders
- **`data/`**: Scripts for handling and preprocessing the dataset.
- **`model/`**: Implementation of the 3D ResNet-18 architecture.
- **`training/`**: Scripts for training and evaluating the model.
- **`results/`**: Contains evaluation metrics and visualization outputs.

### Supporting Files
- **`requirements.txt`**: Lists the Python dependencies needed to run the project.
- **`README.md`**: Documentation for the repository (this file).

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sadafsf/3D-ResNet-Weighted-Cross-Entropy.git
   ```
2. Navigate to the project directory:
   ```bash
   cd 3D-ResNet-Weighted-Cross-Entropy

---

## Usage

### Preprocess the Dataset
### Main Folders
- **`data/`**: Contains scripts for handling and preprocessing the dataset.
  - `preprocess.py`: Prepares the DAiSEE dataset for training by normalizing, augmenting, and resizing videos.
  - `augmentation.py`: Applies spatial and temporal data augmentations such as Gaussian blur, brightness adjustment, and contrast enhancement.

- **`model/`**: Implements the 3D ResNet-18 architecture.
  - `resnet3d.py`: Contains the implementation of the 3D ResNet-18 model, designed to extract spatial-temporal features from video frames.
  - `loss.py`: Defines the weighted cross-entropy loss function to handle class imbalance effectively.

- **`training/`**: Includes scripts for training and validating the model.
  - `train.py`: Script to train the 3D ResNet-18 model. Incorporates weighted random sampling to balance batches.
  - `validate.py`: Evaluates the model's performance on the validation set.
  - `sampler.py`: Implements a custom weighted random sampler for handling imbalanced classes.

- **`results/`**: Stores outputs from model evaluation.
  - `confusion_matrix.png`: Visualization of the model's classification performance across engagement levels.
  - `accuracy_comparison.png`: Comparison chart of model accuracy with other methods on the DAiSEE dataset.

- **`notebooks/`**: Contains Jupyter notebooks for analysis and visualization.
  - `exploration.ipynb`: Explores the DAiSEE dataset and visualizes class distributions.
  - `visualization.ipynb`: Generates charts and graphs for evaluation metrics.

---

## Results

### A. Accuracy Comparison
Our proposed 3D ResNet-18 model achieved an accuracy of **51.82%**, outperforming some baseline models but falling short of the highest-performing methods. Below is a summary of accuracy comparisons:

| Model                           | Accuracy (%) |
|---------------------------------|--------------|
| Video-level InceptionNet        | 46.4         |
| Frame-level InceptionNet        | 48.7         |
| 3D-ResNet-18 (Proposed)         | **51.82**    |
| ResNet + TCN                   | 63.9         |
| EfficientNetB7 + TCN            | 66.39        |
| EfficientNetB7 + LSTM           | **67.48**    |

### B. Confusion Matrix
The confusion matrix provides insights into the classification performance across the four engagement levels:

| **Actual \ Predicted** | Very Low | Low  | High | Very High |
|-------------------------|----------|------|------|-----------|
| **Very Low**            | 0        | 0    | 3    | 1         |
| **Low**                 | 0        | 0    | 47   | 37        |
| **High**                | 0        | 0    | 537  | 328       |
| **Very High**           | 0        | 0    | 427  | 370       |

Key observations:
- The model performed well on majority classes (*High* and *Very High*).
- It struggled significantly with minority classes (*Very Low* and *Low*), classifying most of them as *High* or *Very High*.

---

## Future Work

- Implementing **contrastive learning** to address dataset imbalance.
- Validating the model on alternative datasets, such as VRESEE.
- Exploring additional temporal downsampling techniques.


