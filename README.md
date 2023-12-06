# Pediatric Bone Marrow Transplantation Survival Prediction

## Overview

This project aims to build a classification pipeline to predict the survival status of pediatric patients undergoing bone marrow transplantation. The dataset used for this project is sourced from UCI’s Machine Learning Repository and contains various characteristics related to the transplantation process.

## Getting Started

### Prerequisites

Make sure you have the following installed:

- Python (version X.X.X)
- Libraries: NumPy, Pandas, Scikit-learn

### Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/your-username/pediatric-bmt-survival-prediction.git
cd pediatric-bmt-survival-prediction
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Project Tasks

1. **Investigate the Data:**
   - Explore the dataset containing bone marrow transplantation characteristics.
   - Identify relevant input and output features.

2. **Prepare Data:**
   - Separate target variable (`survival_status`) and features.
   - Categorize columns as numerical or categorical based on unique values.

3. **Handle Missing Values:**
   - Identify and handle missing values in the dataset.

4. **Split the Data:**
   - Split the dataset into training and testing sets.

5. **Create Preprocessing Pipeline:**
   - Implement preprocessing pipelines for numerical and categorical features.
   - Use techniques such as imputation and one-hot encoding.

6. **Create Classification Pipeline:**
   - Build a classification pipeline including preprocessing, dimensionality reduction (PCA), and a logistic regression classifier.

7. **Hyperparameter Tuning:**
   - Explore hyperparameter tuning using GridSearchCV.
   - Optimize the number of PCA dimensions for each classifier.

8. **Evaluate Model:**
   - Assess the performance of the final model on the test set.
   - Compare results with the initial model.

## Project Extensions

- Explore other classification models for improved performance.
- Tune additional parameters of the chosen model.
- Experiment with alternative feature selection/creation techniques.

## Conclusion

This project demonstrates the use of a comprehensive pipeline for predicting the survival status of pediatric patients undergoing bone marrow transplantation. By leveraging preprocessing steps, hyperparameter tuning, and thoughtful feature selection, the model achieves a notable improvement in accuracy.

Feel free to experiment further and contribute to the enhancement of this project!
