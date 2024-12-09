# ML_Income_Classifier_UsingUCL_Adult_Data_Set

This project is about predicting whether a person's income is above or below $50,000 based on different factors like education, age, work class, and more. The goal is to explore and apply machine learning techniques like Lasso Regression, Ridge Regression, and Random Forest to analyze and improve prediction accuracy.

Data Cleaning: Converts variables into binary indicators (e.g., income, gender, race).
Feature Engineering: Creates new variables like squared age and standardizes variables.
Data Splitting: Splits the dataset into training (70%) and testing (30%) sets.

Model Building:
Lasso and Ridge regression models with cross-validation to tune the shrinkage parameter (λ).
Random Forest models with different numbers of trees (100, 200, and 300) and feature splits.

Model Evaluation:
Finds the best model based on accuracy.
Evaluates the best model on the testing data and calculates the classification accuracy.
Uses a confusion matrix to analyze errors like false positives and false negatives.


Results:
The best model was a Random Forest with 300 trees, achieving a classification accuracy of 81.72% on the testing set. This means the model correctly predicted income categories for most individuals in the test data.
