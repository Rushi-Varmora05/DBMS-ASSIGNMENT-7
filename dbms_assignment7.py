# Load the dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sqlalchemy import create_engine


data = pd.read_csv('train.csv')

# Display the first few rows of the dataset
print(data.head())

# Fill missing values in 'Age' with the median age
data = data.assign(Age=data['Age'].fillna(data['Age'].median()))

# Fill missing values in 'Embarked' with the most common embarkation point
data = data.assign(Embarked=data['Embarked'].fillna(data['Embarked'].mode()[0]))

# Drop the 'Cabin' column due to a high number of missing values
data = data.drop(columns=['Cabin'])

# Extract titles from names
data['Title'] = data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

# Create a family size feature
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

# Select numerical and categorical columns
numerical_cols = ['Age', 'Fare', 'FamilySize']
categorical_cols = ['Sex', 'Embarked', 'Title']

# Define the transformers for numerical and categorical features
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Apply the transformations to the data
data_transformed = preprocessor.fit_transform(data)

# Get the transformed column names
transformed_columns = numerical_cols + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols))

# Convert the transformed data back to a DataFrame
data_transformed_df = pd.DataFrame(data_transformed, columns=transformed_columns)

# Create a SQLite database engine
engine = create_engine('sqlite:///titanic.db')

# Store the transformed data in a SQLite database
data_transformed_df.to_sql('titanic_transformed', engine, index=False, if_exists='replace')

# Function to load data from the SQLite database
def load_data_from_db(engine):
    query = "SELECT * FROM titanic_transformed"
    data_from_db = pd.read_sql(query, engine)
    return data_from_db

# Load the transformed data from the database
data_loaded = load_data_from_db(engine)
print(data_loaded.head())

# Define the features and target variable
X = data_loaded
y = data['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

