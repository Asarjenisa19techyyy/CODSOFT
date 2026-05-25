# TITANIC_PREDICTION.PY
# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
data = pd.read_csv("Titanic-Dataset.csv")

print("\n========== TITANIC DATASET ==========\n")

# Show first rows
print(data.head())

# Select useful columns
data = data[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]

# Fill missing values
data['Age'] = data['Age'].fillna(data['Age'].mean())
data['Fare'] = data['Fare'].fillna(data['Fare'].mean())

# Convert Gender into numbers
encoder = LabelEncoder()
data['Sex'] = encoder.fit_transform(data['Sex'])

# Remove remaining missing values
data.dropna(inplace=True)

# DATA VISUALIZATION

# Survival count graph
plt.figure(figsize=(5,4))
sns.countplot(x='Survived', data=data)

plt.title("Titanic Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Passengers")

plt.show()

# Features and Target
X = data[['Pclass', 'Sex', 'Age', 'Fare']]
y = data['Survived']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Model
model = RandomForestClassifier(n_estimators=100)

# Train Model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL RESULT ==========\n")

print("Model Accuracy:", accuracy)

# Custom Prediction
print("\n========== CUSTOM PASSENGER TEST ==========\n")

# Example Passenger
# Pclass = 1
# Sex = female (1)
# Age = 24
# Fare = 100

custom_data = [[1, 1, 24, 100]]

prediction = model.predict(custom_data)

if prediction[0] == 1:
    print("Passenger is likely to SURVIVE")
else:
    print("Passenger is likely to NOT SURVIVE")

print("\n========== PROJECT COMPLETED ==========")