import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Dataset creation
df = pd.read_csv('student_study_attendance_pass.csv')


# 1. Data Exploration

# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Outlier detection with boxplots
plt.figure(figsize=(12, 4))
plt.subplot(2, 2, 1)
sns.boxplot(x=df['Study Hours'])
plt.title("Study Hours - Boxplot")

plt.subplot(2, 2, 2)
sns.boxplot(x=df['Attendance'])
plt.title("Attendance - Boxplot")


# Relationship plot
plt.subplot(2,2,3)
sns.scatterplot(data=df, x="Study Hours", y="Attendance", hue="Pass", style="Pass", palette="coolwarm", s=100)
plt.title("Study Hours vs Attendance colored by Pass")
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Model Training

# Features and target
X = df[['Study Hours', 'Attendance']]
y = df['Pass']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# 3. Model Evaluation

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)


# 4. Deliverables

# 1. Classification Model:
# Logistic Regression model trained using 'Study Hours' and 'Attendance' to predict 'Pass'.

# 2. Evaluation Metrics:
print("\nEvaluation Metrics:")
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)

# 3. Insights:
print("\nInsights:")
print("- Students with higher study hours and higher attendance are more likely to pass.")
print("- Both features are positively correlated with passing.")
print("- Logistic regression quantifies this relationship with learned weights.")
