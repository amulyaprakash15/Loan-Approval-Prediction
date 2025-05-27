# loan_prediction.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------- STEP 1: Load Dataset --------------------
# Make sure 'loan_data.csv' is in the same folder as this script
if not os.path.exists("loan_data.csv"):
    print("❌ ERROR: 'loan_data.csv' not found! Please download and place it here.")
    exit()

df = pd.read_csv("loan_data.csv")

# -------------------- STEP 2: Initial Data Info --------------------
print("\n✅ First 5 Rows:")
print(df.head())

print("\n🧼 Missing Values Before Cleanup:")
print(df.isnull().sum())

# -------------------- STEP 3: Fill Missing Values --------------------
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

# -------------------- STEP 4: Drop Unused Columns --------------------
df.drop('Loan_ID', axis=1, inplace=True)

# -------------------- STEP 5: Label Encoding --------------------
le = LabelEncoder()
columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']

for col in columns:
    df[col] = le.fit_transform(df[col])

# -------------------- STEP 6: Split Data --------------------
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------- STEP 7: Train the Model --------------------
model = RandomForestClassifier()
model.fit(X_train, y_train)

# -------------------- STEP 8: Evaluate the Model --------------------
y_pred = model.predict(X_test)

print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Plot confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -------------------- STEP 9: Predict on a Sample --------------------
sample = X_test.iloc[0:1]
prediction = model.predict(sample)
status = "Approved" if prediction[0] == 1 else "Rejected"
print("\n🔍 Sample Input:\n", sample)
print("🎯 Predicted Loan Status:", status)
import joblib

# Save model
joblib.dump(model, "model.pkl")
print("✅ Model saved as model.pkl")
