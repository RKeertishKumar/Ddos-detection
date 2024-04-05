import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Load preprocessed datasets
X_train = np.loadtxt('X_train_preprocessed.csv', delimiter=',', skiprows=1)
X_val = np.loadtxt('X_val_preprocessed.csv', delimiter=',', skiprows=1)
X_test = np.loadtxt('X_test_preprocessed.csv', delimiter=',', skiprows=1)
y_train = np.loadtxt('y_train_preprocessed.csv', delimiter=',', skiprows=1)
y_val = np.loadtxt('y_val_preprocessed.csv', delimiter=',', skiprows=1)
y_test = np.loadtxt('y_test_preprocessed.csv', delimiter=',', skiprows=1)

# Define the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
start_time = time.time()
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val))
end_time = time.time()
training_time = end_time - start_time
print(f"Training time: {training_time} seconds")

# Predictions
y_train_prob = model.predict(X_train)
y_val_prob = model.predict(X_val)
y_test_prob = model.predict(X_test)

# Convert probabilities to class labels
y_train_pred = (y_train_prob > 0.5).astype(int)
y_val_pred = (y_val_prob > 0.5).astype(int)
y_test_pred = (y_test_prob > 0.5).astype(int)

# Generate classification report
report = classification_report(y_test, y_test_pred, target_names=['Normal', 'DDoS'])

# Print classification report
print("Classification Report for Test Data:")
print(report)

# Compute confusion matrices
train_cm = confusion_matrix(y_train, y_train_pred)
val_cm = confusion_matrix(y_val, y_val_pred)
test_cm = confusion_matrix(y_test, y_test_pred)

# Define labels
labels = ['Normal', 'DDoS']

# Plot confusion matrix
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.heatmap(train_cm, annot=True, cmap='Blues', fmt='g', xticklabels=labels, yticklabels=labels)
plt.title('Training Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.subplot(1, 3, 2)
sns.heatmap(val_cm, annot=True, cmap='Oranges', fmt='g', xticklabels=labels, yticklabels=labels)
plt.title('Validation Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.subplot(1, 3, 3)
sns.heatmap(test_cm, annot=True, cmap='Greens', fmt='g', xticklabels=labels, yticklabels=labels)
plt.title('Test Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.show()
