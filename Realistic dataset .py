import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree  
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# Load the dataset
creditcard2 = pd.read_csv("/kaggle/input/fyp2025/creditcard.csv")

# Display information
creditcard2.info()

# Check for missing values
print(creditcard2.isnull().sum())

# Piechart of Class Distribution
class_counts = {"Legitimate (0)": 284315, "Fraudulent (1)": 492}
creditcard2_class_distribution = pd.Series(class_counts)

plt.figure(figsize=(6,6))
creditcard2_class_distribution.plot(kind='pie', autopct='%1.2f%%', labels=creditcard2_class_distribution.index, 
                           colors=["skyblue", "red"], startangle=90, explode=[0, 0.1], textprops={'ha': 'center'})
plt.ylabel('')
plt.title("Transaction Class Distribution")
plt.show()

# Boxplot for 'Amount'
sns.boxplot(x=creditcard2["Class"], y=creditcard2["Amount"])
plt.title("Transaction Amount Distribution for Legitimate and Fraudulent Transactions")
plt.show()

# Log-transform 'Amount'
creditcard2['Amount'] = np.log1p(creditcard2['Amount'] + 0.0001)
# Boxplot for Amount feature
sns.boxplot(x=creditcard2["Class"], y=creditcard2["Amount"])
plt.title("Transaction Amount Distribution for Legitimate and Fraudulent Transactions")
plt.ylabel("Log-Amount")  # Hide y-axis label
plt.show()

# Heatmap analysis
plt.figure(figsize=(10,8))
sns.heatmap(creditcard2.corr(), cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Convert Time from seconds to hours (easier to interpret)
creditcard2['Hour'] = creditcard2['Time'] / 3600  # 3600 seconds in 1 hour

# Plot the diagram one above the other
plt.figure(figsize=(12, 10))

# Legitimate Transactions time distribution
plt.subplot(2, 1, 1)
sns.histplot(creditcard2[creditcard2["Class"] == 0]["Hour"], bins=50, color="blue", kde=True) # hours
plt.title("Legitimate Transactions Over Time")
plt.xlabel("Time (hours)")
plt.ylabel("Count")

# Fraudulent Transactions time distribution
plt.subplot(2, 1, 2)
sns.histplot(creditcard2[creditcard2["Class"] == 1]["Hour"], bins=50, color="red", kde=True) # hours
plt.title("Fraudulent Transactions Over Time")
plt.xlabel("Time (hours)")
plt.ylabel("Count")

plt.tight_layout()
plt.show()

# Drop 'Time' variable
creditcard2 = creditcard2.drop(columns=['Time'])

# Separate features and target
X = creditcard2.drop(columns=['Class'])
y = creditcard2['Class']

# Standardise features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (80% train and 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# ========================
# LOGISTIC REGRESSION WITHOUT SMOTE
# ========================
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)

# Model Evaluation
print("Logistic Regression (Without SMOTE)")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

# ========================
# LOGISTIC REGRESSION WITH SMOTE
# ========================
# Feature selection
selected_features = ["V4", "V11", "V12", "V15", "V17", "V1", "V3", "V9", "V10", "V5", "V18", "V22", "V25", "Amount"]
X = creditcard2[selected_features]
y = creditcard2["Class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Logistic Regression on SMOTE data
lr_model_smote = LogisticRegression()
lr_model_smote.fit(X_train_resampled, y_train_resampled)

y_pred_smote = lr_model_smote.predict(X_test)

# Model Evaluation
print("\nLogistic Regression (With SMOTE)")
print("Accuracy:", accuracy_score(y_test, y_pred_smote))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_smote))
print("\nClassification Report:\n", classification_report(y_test, y_pred_smote, digits=4))

# ========================
# DECISION TREE
# ========================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree  
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# Load the dataset
creditcard2 = pd.read_csv("/kaggle/input/fyp22025/creditcard.csv")

# Display information
creditcard2.info()

# Check for missing values
print(creditcard2.isnull().sum())

# Piechart of Class Distribution
class_counts = {"Legitimate (0)": 284315, "Fraudulent (1)": 492}
creditcard2_class_distribution = pd.Series(class_counts)

plt.figure(figsize=(6,6))
creditcard2_class_distribution.plot(kind='pie', autopct='%1.2f%%', labels=creditcard2_class_distribution.index, 
                           colors=["skyblue", "red"], startangle=90, explode=[0, 0.1])
plt.ylabel('')
plt.title("Transaction Class Distribution")
plt.show()

# Boxplot for 'Amount'
sns.boxplot(x=creditcard2["Class"], y=creditcard2["Amount"])
plt.title("Transaction Amount Distribution for Legitimate and Fraudulent Transactions")
plt.show()

# Log-transform 'Amount'
creditcard2['Amount'] = np.log1p(creditcard2['Amount'] + 0.0001)

# Heatmap analysis
plt.figure(figsize=(10,8))
sns.heatmap(creditcard2.corr(), cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Drop 'Time' variable
creditcard2 = creditcard2.drop(columns=['Time'])

# Separate features and target
X = creditcard2.drop(columns=['Class'])
y = creditcard2['Class']

# Standardise features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (80% train and 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# ========================
# LOGISTIC REGRESSION WITHOUT SMOTE
# ========================
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)

print("Logistic Regression (Without SMOTE)")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

# ========================
# LOGISTIC REGRESSION WITH SMOTE
# ========================
# Feature selection
selected_features = ["V4", "V11", "V12", "V15", "V17", "V1", "V3", "V9", "V10", "V5", "V18", "V22", "V25", "Amount"]
X = creditcard2[selected_features]
y = creditcard2["Class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Logistic Regression on SMOTE data
lr_model_smote = LogisticRegression()
lr_model_smote.fit(X_train_resampled, y_train_resampled)

y_pred_smote = lr_model_smote.predict(X_test)

print("\nLogistic Regression (With SMOTE)")
print("Accuracy:", accuracy_score(y_test, y_pred_smote))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_smote))
print("\nClassification Report:\n", classification_report(y_test, y_pred_smote, digits=4))

# ========================
# DECISION TREE
# ========================
dt_param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}

# Grid Search and training
dt = DecisionTreeClassifier(random_state=42)
grid_search_dt = GridSearchCV(dt, dt_param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
grid_search_dt.fit(X_train_resampled, y_train_resampled)

best_dt = grid_search_dt.best_estimator_
y_pred_dt = best_dt.predict(X_test)

print("\nDecision Tree")
print("Best Parameters:", grid_search_dt.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
print("\nClassification Report:\n", classification_report(y_test, y_pred_dt, digits=4))

# Decision Tree diagram
plt.figure(figsize=(20, 10))
plot_tree(best_dt, feature_names=creditcard2[selected_features].columns, class_names=['Not Fraud', 'Fraud'], filled=True, rounded=True)
plt.show()
# ========================
# RANDOM FOREST
# ========================

# Choosing the best parameter
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': ['balanced']
}

rf = RandomForestClassifier(random_state=42)
grid_search_rf = GridSearchCV(rf, rf_param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
grid_search_rf.fit(X_train_resampled, y_train_resampled)

best_rf = grid_search_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)

# Model Evaluation
print("\nRandom Forest")
print("Best Parameters:", grid_search_rf.best_params_) # Display the chosen parameters
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf, digits=4))
