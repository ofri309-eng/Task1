# Product Requirements Document (PRD)

## Project: Iris Flower Classification System (V2)

### 1. Objective
Develop a robust machine learning system capable of classifying flowers into four distinct categories with at least 90% accuracy, providing actionable insights through visualization and a real-time prediction interface.

### 2. Functional Requirements
- **FR1**: Process `iris_extended.csv` with 200 samples and 4 categories.
- **FR2**: Implement an 80/20 train-test split for unbiased evaluation.
- **FR3**: Implement iterative weight optimization using Mean Square Error (MSE).
- **FR4**: Generate 3 key visualizations (Confusion Matrix, MSE Convergence, Point Clouds).
- **FR5**: Interactive CLI for predicting new flower species.

### 3. Non-Functional Requirements
- **NFR1**: Minimal dependencies (NumPy, Pandas, Matplotlib only).
- **NFR2**: High performance (Training under 5 seconds).
- **NFR3**: Modular and documented code.

### 4. Target Users
- Flower sorting factory managers.
- Data scientists interested in custom ML implementations.
- Quality assurance teams in horticulture.

### 5. Success Metrics
- **Accuracy**: > 90% on test data.
- **Convergence**: MSE decreases steadily across 2000+ iterations.
- **Usability**: Interactive prediction works correctly for all 4 categories.
