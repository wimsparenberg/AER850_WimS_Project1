import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

#1 - Data Processing
df = pd.read_csv("Data/Project 1 Data.csv")

X = df[['X','Y','Z']]
y = df['Step']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#2 - Data Visualization
desc_stat = df[['X', 'Y', 'Z',]].describe()
desc_stat_step = df['Step'].value_counts().sort_index()
df.hist()

#3 - Data correlation
corr = df[['X','Y','Z']].corr()
plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

#4 - Model Development
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = {
    "SVM": (SVC(), {"C": [0.1, 1, 10, 100], "kernel": ["linear","rbf"], "gamma":["scale","auto"]}),
    "RandomForest": (RandomForestClassifier(), {"n_estimators":[50, 100, 200], "max_depth":[None,5,10]}),
    "LogisticRegression": (LogisticRegression(max_iter=1000, multi_class="multinomial"), {"C":[0.01, 0.1, 1, 10, 100], "solver":["lbfgs","saga"]})
    }