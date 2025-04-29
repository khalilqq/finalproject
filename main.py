# 1. Import libraries
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# 2. Load Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
titanic = pd.read_csv(url)

# 3. Cleaning data
# Drop columns with too many missing values or irrelevant ones
titanic.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)

# Fill missing Age values with median
titanic['Age'].fillna(titanic['Age'].median(), inplace=True)

# Fill missing Embarked with most common value
titanic['Embarked'].fillna(titanic['Embarked'].mode()[0], inplace=True)

# Convert 'Sex' and 'Embarked' to numeric
titanic['Sex'] = titanic['Sex'].map({'male': 0, 'female': 1})
titanic['Embarked'] = titanic['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# 4. EDA - Visualizations

# Visualization 1: Survival Count (Bar Plot)
sns.countplot(x='Survived', data=titanic)
plt.title('Visualization 1: Survival Count')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# Visualization 2: Survival Rate by Sex (Bar Plot)
sns.barplot(x='Sex', y='Survived', data=titanic)
plt.title('Visualization 2: Survival Rate by Sex')
plt.xlabel('Sex (0 = Male, 1 = Female)')
plt.ylabel('Survival Rate')
plt.show()

# Visualization 3: Overall Survival Rate (Pie Chart)
survived_count = titanic['Survived'].sum()
not_survived_count = len(titanic) - survived_count

labels = ['Did Not Survive', 'Survived']
sizes = [not_survived_count, survived_count]
colors = ['lightcoral', 'lightgreen']

plt.figure(figsize=(6,6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Visualization 3: Overall Survival Rate (Pie Chart)')
plt.show()

# 5. Machine Learning Model

# Features and label
X = titanic.drop('Survived', axis=1)
y = titanic['Survived']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation

# Print evaluation scores
print("Model Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix values manually
true_negative = np.sum((y_test == 0) & (y_pred == 0))
false_positive = np.sum((y_test == 0) & (y_pred == 1))
false_negative = np.sum((y_test == 1) & (y_pred == 0))
true_positive = np.sum((y_test == 1) & (y_pred == 1))

# Visualization 4: Model Prediction Accuracy (Bar Plot)

# Correct vs Incorrect counts
correct_predictions = true_negative + true_positive
incorrect_predictions = false_positive + false_negative

labels = ['Correct Predictions', 'Incorrect Predictions']
counts = [correct_predictions, incorrect_predictions]

plt.figure(figsize=(6,6))
sns.barplot(x=labels, y=counts, palette=['lightgreen', 'lightcoral'])
plt.title('Visualization 4: Model Prediction Accuracy (Bar Plot)')
plt.ylabel('Number of Predictions')
plt.xlabel('Prediction Outcome')
plt.ylim(0, max(counts) + 20)
plt.show()

# 6. Statistical Analysis - T-test

# Test if Age of survivors is significantly different from non-survivors
survived_ages = titanic[titanic['Survived'] == 1]['Age']
not_survived_ages = titanic[titanic['Survived'] == 0]['Age']

t_stat, p_val = stats.ttest_ind(survived_ages, not_survived_ages)
print(f"T-test result: T-statistic = {t_stat:.2f}, p-value = {p_val:.4f}")
