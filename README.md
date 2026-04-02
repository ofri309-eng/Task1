# Iris Flower Classification System

## Overview
This project implements a custom machine learning classification system designed to categorize flowers based on physical measurements (sepal and petal dimensions). It features an expanded dataset with 4 distinct categories, including the high-yield 'Iris-gigantica'.

## Business Value
- **Automated Sorting**: Reduces manual labor costs in commercial flower sorting facilities.
- **High Accuracy**: Minimizes misclassification, ensuring premium pricing for correctly identified species.
- **Scalability**: The system is designed to handle new categories and larger datasets as the business grows.

## Features
- **Custom ML Engine**: Built using pure NumPy/Pandas for maximum transparency and performance.
- **Iterative Learning**: Tracks error reduction (MSE) to ensure convergence.
- **Visual Analytics**: Automatically generates confusion matrices and trend graphs.
- **Interactive Interface**: Predict species for new flower samples in real-time.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the classification system:
   ```bash
   python main.py
   ```
3. View results in the `outputs/` folder.
