# Step 1: Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Step 2: Load Dataset
credit_card_data = pd.read_csv('/content/creditcard.csv')

credit_card_data.head()

credit_card_data.tail()

print("\nDataset Info:")
credit_card_data.info()
# Step 4: Check for Missing Values
print("\nMissing Values in Dataset:")
print(credit_card_data.isnull().sum())

# Step 5: Display Class Distribution (Imbalance Check)
print("\nClass Distribution:")
print(credit_card_data['Class'].value_counts())
# Count fraud (Class == 1) and legit (Class == 0)
fraud_count = df['Class'].value_counts()[1]
legit_count = df['Class'].value_counts()[0]

print(f" Total Fraud Transactions: {fraud_count}")
print(f" Total Legit Transactions: {legit_count}")
# Step 6: Visualize Class Distribution
sns.countplot(x=credit_card_data['Class'])
plt.title("Class Distribution")
plt.show()

# Step 7: Separate Legit & Fraud Transactions
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]
print("\nLegit Transactions:", legit.shape)
print("Fraud Transactions:", fraud.shape)
# Step 8: Statistical Summary
print("\nLegit Transactions Amount Statistics:")
print(legit['Amount'].describe())
print("\nFraud Transactions Amount Statistics:")
print(fraud['Amount'].describe())
# ⚖ Step 11: Handle Class Imbalance with SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print("After SMOTE:\n", pd.Series(y_res).value_counts())
# Step 9: Create Balanced Dataset Using SMOTE
X = credit_card_data.drop(columns=['Class'], axis=1)
Y = credit_card_data['Class']
# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, Y_resampled = smote.fit_resample(X, Y)

print("\nAfter SMOTE Balancing:")
print(Y_resampled.value_counts())  # Should show equal fraud and legit transactions
import seaborn as sns
import matplotlib.pyplot as plt

# Original class distribution before SMOTE
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.countplot(x=Y, palette="coolwarm")
plt.title("Class Distribution Before SMOTE")
plt.xlabel("Class (0: Legit, 1: Fraud)")
plt.ylabel("Count")

# Class distribution after SMOTE
plt.subplot(1, 2, 2)
sns.countplot(x=Y_resampled, palette="coolwarm")
plt.title("Class Distribution After SMOTE")
plt.xlabel("Class (0: Legit, 1: Fraud)")
plt.ylabel("Count")

plt.show()
# Step 10: Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X_resampled, Y_resampled, test_size=0.2, stratify=Y_resampled, random_state=2)
print("\nTrain-Test Split Shapes:", X_train.shape, X_test.shape)


# Step 11: Standardization (Feature Scaling)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 12: Build Sequential Model (LSTM)
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Step 13: Train Model
X_train_reshaped = np.expand_dims(X_train, axis=-1)
X_test_reshaped = np.expand_dims(X_test, axis=-1)
history = model.fit(X_train_reshaped, Y_train, epochs=10, batch_size=64, validation_data=(X_test_reshaped, Y_test))
# Step 14: Model Evaluation
Y_pred = model.predict(X_test_reshaped)
Y_pred = (Y_pred > 0.5).astype(int)

print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, Y_pred))

print("\nClassification Report:")
print(classification_report(Y_test, Y_pred))
# Step 15: Accuracy Check
accuracy = accuracy_score(Y_test, Y_pred)
print("\nModel Accuracy:", accuracy)

# Step 16: Visualization of Training Performance
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()