# SVNH-CNN-PyTorch

## Project Overview

This repository contains a Convolutional Neural Network (CNN) implemented in PyTorch for digit classification on the SVHN dataset.

Key features include:

- A lightweight, well-structured CNN model
- Training and evaluation pipeline in PyTorch
- Image preprocessing and augmentation
- Clear modular structure suitable for learning and experimentation
- Fully reproducible environment

## Detailed Description (See Report)

A full explanation of the architecture, training procedure, evaluation results, and design decisions is available in the `StudyReport.pdf` included in this repository.

The section describing this repository begins on page 12 of the report.

If you're reading this on GitHub, here is a quick link: [Open StudyReport.pdf](StudyReport.pdf) (Scroll to page 12.)

## Getting Started

1. Install Dependencies
   ```
   pip install torch torchvision matplotlib numpy
   ```

2. Run Training
   ```
   python train.py
   ```

3. Evaluate Model
   ```
   python evaluate.py
   ```

## Dataset

This project uses the SVHN dataset:
http://ufldl.stanford.edu/housenumbers/

The dataset is automatically downloaded by torchvision if not present.

## Model Summary

- Several convolutional layers with ReLU
- Max-pooling layers for spatial reduction
- Fully connected classifier
- Cross-entropy loss and Adam optimizer

Architecture details are documented in the `StudyReport.pdf` (page 12).

## Results

Evaluation metrics (accuracy, loss curves, training behaviour) are available in the report.

## License

MIT License — feel free to use this for learning or research.
