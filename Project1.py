import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import joblib

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
    "SVM": (SVC(kernel="linear"), {"C": [0.1, 1, 10, 100], "gamma":["scale","auto"]}),
    "RandomForest": (RandomForestClassifier(), {"n_estimators":[50, 100, 200], "max_depth":[None,5,10]}),
    "LogisticRegression": (LogisticRegression(max_iter=1000), {"C":[0.01, 0.1, 1, 10, 100], "solver":["lbfgs","saga"]})
    }

best_models = {}
for name, (model, params) in models.items():
    grid = GridSearchCV(model, params, cv=cv, scoring="f1_weighted", n_jobs=-1)
    grid.fit(X_train_scaled, y_train)
    best_models[name] = grid.best_estimator_
    print(f"\n{name} best params: {grid.best_params_}")
    
#5 - Model Performance
    def evaluate_model(model, X_test, y_test):
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")
        print(f"\n{name} Performance:")
        print(f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, F1 Score: {f1:.3f}")
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm).plot()
        plt.title(f"Confusion Matrix - {name}")
        plt.show()
        return acc, prec, f1

    results = {}
    for name, model in best_models.items():
        results[name] = evaluate_model(model, X_test_scaled, y_test, name)