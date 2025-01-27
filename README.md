
# Predicting Student Engagement with 3D ResNet-18 and Weighted Cross-Entropy Loss

This repository contains the code and methodologies from the paper **"Predicting Student Engagement with 3D ResNet-18 and Weighted Cross-Entropy Loss"**, presented at the 35th Canadian Conference on Artificial Intelligence. The study explores the detection of student engagement levels in online learning environments using deep learning.

## Overview

The COVID-19 pandemic significantly increased the adoption of online learning, making it critical to assess and enhance student engagement in virtual classrooms. This project uses videos from the **DAiSEE dataset** to classify student engagement into four levels: *very low*, *low*, *high*, and *very high*.

We implemented a **3D ResNet-18 model** with weighted cross-entropy loss and a weighted random sampler to address class imbalance in the dataset. Our method achieved a classification accuracy of 51.82% on the four-class problem.

## Features

- **Dataset**: DAiSEE, a publicly available dataset with annotated videos of students in online learning.
- **Deep Learning Model**: 3D ResNet-18 for spatiotemporal feature extraction.
- **Class Imbalance Handling**: Weighted random sampling and weighted cross-entropy loss.
- **Performance Comparison**: Benchmarked against state-of-the-art methods.

## Key Results

- **Accuracy**: 51.82% on the DAiSEE dataset.
- Our model outperformed several baseline approaches but did not exceed the current state-of-the-art methods.

## Repository Structure and Files

Here’s a breakdown of the files in this repository:

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

### Configuration Files
- **`configs/train_config.yaml`**: Configuration file for training, including hyperparameters like learning rate, batch size, and optimizer settings.

### Supporting Files
- **`requirements.txt`**: Lists the Python dependencies needed to run the project.
- **`README.md`**: Documentation for the repository (this file).

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sadafsf/3D-ResNet-Weighted-Cross-Entropy.git
