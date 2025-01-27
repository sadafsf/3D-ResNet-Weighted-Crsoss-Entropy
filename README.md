
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

## Repository Structure

- `data/`: Scripts for dataset preprocessing and augmentation.
- `model/`: Implementation of the 3D ResNet-18 architecture.
- `training/`: Training scripts with support for weighted loss and sampling.
- `results/`: Model evaluation metrics and confusion matrices.
- `notebooks/`: Jupyter notebooks for analysis and visualization.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/student-engagement-resnet.git
