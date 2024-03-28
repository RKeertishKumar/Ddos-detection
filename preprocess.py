import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Load the split datasets from CSV
X_train = pd.read_csv('X_train.csv')
X_val = pd.read_csv('X_val.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')['Class']
y_val = pd.read_csv('y_val.csv')['Class']
y_test = pd.read_csv('y_test.csv')['Class']

# Preprocessing
# 1. Handling Missing Values (if any)
X_train.dropna(inplace=True)
X_val.dropna(inplace=True)
X_test.dropna(inplace=True)

# 2. Encoding Categorical Variables (if necessary)
# Encoding 'Protocol'
encoder_protocol = LabelEncoder()
X_train['Protocol'] = encoder_protocol.fit_transform(X_train['Protocol'])
X_val['Protocol'] = encoder_protocol.transform(X_val['Protocol'])
X_test['Protocol'] = encoder_protocol.transform(X_test['Protocol'])

# Encoding 'Label'
encoder_label = LabelEncoder()
X_train['Label'] = encoder_label.fit_transform(X_train['Label'])
X_val['Label'] = encoder_label.transform(X_val['Label'])
X_test['Label'] = encoder_label.transform(X_test['Label'])

# 3. Convert Target Variable to Numerical (if necessary)
# Assuming 'Class' contains categorical values 'Attack' and 'Benign'
y_train = y_train.map({'Attack': 1, 'Benign': 0})
y_val = y_val.map({'Attack': 1, 'Benign': 0})
y_test = y_test.map({'Attack': 1, 'Benign': 0})

# 4. Scaling Numerical Features
scaler = MinMaxScaler()
numerical_columns = ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Fwd Packets Length Total',
                     'Bwd Packets Length Total', 'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
                     'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
                     'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std',
                     'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max',
                     'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
                     'Fwd Packets/s', 'Bwd Packets/s', 'Packet Length Min', 'Packet Length Max', 'Packet Length Mean',
                     'Packet Length Std', 'Packet Length Variance', 'Avg Packet Size', 'Avg Fwd Segment Size',
                     'Avg Bwd Segment Size', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate',
                     'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets',
                     'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init Fwd Win Bytes',
                     'Init Bwd Win Bytes', 'Fwd Act Data Packets', 'Fwd Seg Size Min', 'Active Mean', 'Active Std',
                     'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_val[numerical_columns] = scaler.transform(X_val[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# Save the preprocessed datasets to CSV files
X_train.to_csv('X_train_preprocessed.csv', index=False)
X_val.to_csv('X_val_preprocessed.csv', index=False)
X_test.to_csv('X_test_preprocessed.csv', index=False)
y_train.to_csv('y_train_preprocessed.csv', index=False, header=True)
y_val.to_csv('y_val_preprocessed.csv', index=False, header=True)
y_test.to_csv('y_test_preprocessed.csv', index=False, header=True)
