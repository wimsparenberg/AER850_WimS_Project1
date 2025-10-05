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
sns.heatmap(corr, annot=True, fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

#4 - Model Development
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = {
    "SVM": (SVC(kernel="linear"), {"C": [0.1, 1, 10, 100], "gamma":["scale","auto"]}),
    "LogisticRegression": (LogisticRegression(max_iter=1000), {"C":[0.01, 0.1, 1, 10, 100], "solver":["lbfgs","saga"]})
    }
best_models = {}
for name, (model, params) in models.items():
    grid = GridSearchCV(model, params, cv=cv)
    grid.fit(X_train_scaled, y_train)
    best_models[name] = grid.best_estimator_
    print(f"\n{name} best params: {grid.best_params_}")
    
rf_params = {
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [None, 5, 10, 15],
}
randomsearch = RandomizedSearchCV(
    RandomForestClassifier(),
    rf_params,
    cv=cv,
    n_iter=10,
    random_state=42
)
randomsearch.fit(X_train_scaled, y_train)
best_models["RandomForest"] = randomsearch.best_estimator_
print("\nRandomForest best params:", randomsearch.best_params_)
    
#5 - Model Performance
def evaluate_model(model, X_test, y_test, name="Model"):
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

#6 - Stacking Models
stack_model = StackingClassifier(estimators=[("svm", best_models["SVM"]), ("lr", best_models["LogisticRegression"])],final_estimator=LogisticRegression(max_iter=1000))
stack_model.fit(X_train_scaled, y_train)
results["Stacked"] = evaluate_model(stack_model, X_test_scaled, y_test, "Stacked Model")
best_model_name = max(results, key=lambda k: results[k][2])
final_model = best_models.get(best_model_name, stack_model)

best_model = max(results, key=results.get)
joblib.dump(best_models.get(best_model, stack_model), "final_model.joblib")
print("Best model saved:", best_model)

#7 - Testing Model
coords = np.array([[9.375,3.0625,1.51],[6.995,5.125,0.3875], [0,3.0625,1.93], [9.4,3,1.8],[9.4,3,1.3]])
coords_scaled = scaler.transform(coords)
prediction = final_model.predict(coords_scaled)
print("\nPredictions for given points:", prediction)