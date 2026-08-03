# Job Candidate Hiring Outcome Classification

Machine learning project for predicting whether a job candidate will be hired based on candidate profile and experience data.

[View the executed Jupyter Notebook](candidate_acceptance_classification.ipynb)

## Overview

This project was developed as the final assignment for the Introduction to Machine Learning course in Spring 2024.

The task was a binary classification problem with 55,462 labeled training records and 18,000 unlabeled test records. The notebook covers exploratory data analysis, preprocessing, feature engineering, model comparison, evaluation, and generation of probability predictions.

## Approach

- Explored feature distributions, missing values, correlations, and anomalies
- Converted binary, ordinal, and categorical features into numerical representations
- Handled missing values using rule-based methods, correlation-based filling, and KNN imputation
- Created `len_of_stack` from each candidate's technology experience
- Applied one-hot encoding and PCA to the `stack_experience` feature
- Built reusable utilities for model training, cross-validation, evaluation, and visualization
- Explored AIF360 Reweighing for sensitive attributes
- Created a final pipeline for preprocessing, model training, and test-set prediction

## Models and Results

The models were evaluated using five-fold cross-validation and an internal holdout set.

| Model | Cross-Validation ROC-AUC | Holdout ROC-AUC |
|---|---:|---:|
| Gaussian Naive Bayes | 0.9356 | 0.9342 |
| Logistic Regression | 0.9576 | 0.9553 |
| Multi-Layer Perceptron | 0.9535 | 0.9532 |
| AdaBoost | 0.9579 | 0.9558 |

AdaBoost achieved the highest holdout ROC-AUC and was used in the final prediction pipeline. The notebook also includes ROC curves and an AdaBoost confusion matrix.

## Data and Reproduction

The training and test datasets were provided for the course and are not included in this repository.

The notebook contains its saved outputs and can be reviewed directly on GitHub. To rerun it, place the following files in the same directory as the notebook:

- `train.csv`
- `test.csv`

The final pipeline saves the generated probabilities to `results_23.csv`.

## Technologies

Python, Jupyter Notebook, pandas, NumPy, scikit-learn, Matplotlib, seaborn, SciPy, and IBM AIF360.

## Authors

Developed collaboratively by:

- [Ofek Zilber](https://github.com/Ofekzil4)
- [Hadar Asraf](https://github.com/Hoder232)

Most of the research, analysis, implementation, evaluation, and decision-making was completed jointly. During the final organization phase, Ofek refactored substantial parts of the code into reusable functions, while Hadar finalized the explanations, function documentation, and notebook presentation. The final report was written jointly.
