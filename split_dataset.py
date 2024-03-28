import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset from CSV
dataset = pd.read_csv('cicddos2019_dataset.csv')

# Split the dataset into features and target variable
X = dataset.drop(columns=['Class'])  # Replace 'target_column' with the name of your target column
y = dataset['Class']

# Split the dataset into training, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 80% for training + validation, 20% for testing

# Further split the training + validation set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)  # 60% for training, 20% for validation

# Save the split datasets to separate CSV files
X_train.to_csv('X_train.csv', index=False)
X_val.to_csv('X_val.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False, header=True)
y_val.to_csv('y_val.csv', index=False, header=True)
y_test.to_csv('y_test.csv', index=False, header=True)
