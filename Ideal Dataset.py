import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc

# Load dataset
creditcard= pd.read_csv("/kaggle/input/credit-card-explore/creditcard_2023.csv")  # Ensure correct filename

creditcard.isnull().values.any() #check if there is any missing values

# Piechart of Class Distribution
count_classes = creditcard['Class'].value_counts()
count_classes.plot(kind='pie', autopct='%1.1f%%', labels=['Legitimate (0)', 'Fraudulent (1)'], colors=["skyblue", "red"])
plt.ylabel('')  #to hide y-axis
plt.title("Transaction Class Distribution")
plt.show()

# Counting number of fraud and legit transactions
outlier_fraction = len(fraud)/float(len(normal))
print(outlier_fraction)
print("Fraud transactions: {}".format(len(fraud)))
print("Legit transactions: {}".format(len(normal)))

# 'Class' column (1 = Fraud, 0 = Not Fraud)
target_column = 'Class'

# Separate features and target
X = creditcard.drop(columns=[target_column])
y = creditcard[target_column]

# Standardise the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Training Logistic Regression Model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predictions
y_pred = log_reg.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
