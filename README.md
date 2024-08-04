# DBMS-ASSIGNMENT-7
ETL Jobs in Machine Learning Applications

## Project Overview
This project focuses on predicting the survival of passengers on the Titanic using machine learning techniques. We preprocess the data, apply feature engineering, and train a logistic regression model to predict the target variable (Survived).

## Dataset
The dataset used in this project is the Titanic dataset, which can be found on Kaggle. It includes information about the passengers on the Titanic, such as their age, sex, fare, and more.

## Installation
### To set up this project locally, follow these steps:

1) Clone the repository:
   https://github.com/Rushi-Varmora05/DBMS-ASSIGNMENT-7.git
   cd DBMS-ASSIGNMENT-7
2) Create a virtual environment and activate it:
   python -m venv venv
   venv\Scripts\activate
3)Install the required packages:
  pip install -r requirements.txt

## Usage
1) **Load the dataset:** The code first loads the Titanic dataset (train.csv) and displays the first few rows.

2) **Preprocessing:**
  Missing values in the 'Age' column are filled with the median age. /n
  Missing values in the 'Embarked' column are filled with the most common embarkation point. /n
  The 'Cabin' column is dropped due to a high number of missing values. /n
  Titles are extracted from names, and a family size feature is created.

3) **Feature Engineering:**
  Numerical features (Age, Fare, FamilySize) are scaled using StandardScaler. /n
  Categorical features (Sex, Embarked, Title) are one-hot encoded.

4) **Data Transformation:**
  The transformed data is stored in a SQLite database (titanic.db).

5) **Model Training and Evaluation:**
  The data is split into training and testing sets. /n
  A logistic regression model is trained and evaluated on the test set.

