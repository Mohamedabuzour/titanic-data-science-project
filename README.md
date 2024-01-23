# titanic-data-science-project
# Titanic Survival Prediction

This repository contains code for predicting the survival of passengers on the Titanic using machine learning. The project includes Exploratory Data Analysis (EDA), data preprocessing, feature engineering, and model training using the Gradient Boosting Classifier. The code is written in Python and utilizes popular libraries such as NumPy, pandas, seaborn, and scikit-learn.

## Contents

1. **Introduction**
   - Overview of the project and its goal.

2. **Data Loading and Exploration**
   - Loading the dataset (`train.csv`).
   - Displaying the initial rows of the dataset.
   - Dropping unnecessary columns and reordering columns.
   - Checking data types, null values, and unique values.

3. **Handling Null Values**
   - Analyzing and handling null values in the dataset.
   - Replacing missing values in the 'Age' column with the median.
   - Dropping the 'Cabin' column due to a high percentage of null values.
   - Removing rows with null values in the 'Embarked' column.

4. **Representing Columns as Categorical**
   - Converting selected columns to categorical data types.

5. **Handling Duplicates**
   - Identifying and removing duplicate rows in the dataset.

6. **Statistical Analysis**
   - Providing statistical summaries for both numeric and categorical data.
   - Checking for outliers using z-scores and box plots.

7. **Exploratory Data Analysis (EDA)**
   - Visualizing relationships between survival and fare, age, and other features.
   - Examining the distribution of categorical features.

8. **Feature Engineering**
   - Creating a new column, 'family_size,' by combining 'SibSp' and 'Parch.'

9. **Data Splitting**
   - Splitting the data into training and testing sets.

10. **Encoding Categorical Variables**
    - Using label encoding for categorical columns.

11. **Normalization**
    - Scaling numeric columns using Min-Max normalization.

12. **Machine Learning Model (Gradient Boosting Classifier)**
    - Building and training a Gradient Boosting Classifier.
    - Hyperparameter tuning using GridSearchCV.

13. **Model Evaluation**
    - Assessing the model's accuracy on the training and testing sets.

## Requirements
- Python 3.x
- NumPy
- pandas
- seaborn
- scikit-learn

## Getting Started
1. Clone this repository to your local machine.
2. Install the required dependencies using:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the provided Jupyter Notebook or execute the code in your preferred environment.

## Acknowledgments
- The code is inspired by the Kaggle competition on predicting the survival of Titanic passengers.

Feel free to explore, modify, and use the code for your own Titanic survival prediction projects! If you have any questions or suggestions, please open an issue or reach out.

**May the predictions be in your favor!**
