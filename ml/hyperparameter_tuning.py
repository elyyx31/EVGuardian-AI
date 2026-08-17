from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ===============================
# Load Dataset
# ===============================
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR.parent / "dataset" / "new_battery_dataset.csv"

df = pd.read_csv(DATASET_PATH)

# ===============================
# Create Battery Health Classes
# ===============================
def battery_health(soh):
    if soh >= 90:
        return "Healthy"
    elif soh >= 80:
        return "Moderate"
    else:
        return "Poor"

df["BatteryHealth"] = df["SOH"].apply(battery_health)

# ===============================
# Features and Target
# ===============================
X = df.drop(columns=["BatteryID", "BatchID", "SOH", "BatteryHealth"])
y = df["BatteryHealth"]

# ===============================
# Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ===============================
# Grid Search Parameters
# ===============================
param_grid = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.05, 0.1],
    "max_depth": [2, 3, 4],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

# ===============================
# Grid Search
# ===============================
grid_search = GridSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

print("Searching for best parameters...\n")

grid_search.fit(X_train, y_train)

print("Best Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation Accuracy:")
print(grid_search.best_score_)

# ===============================
# Best Model
# ===============================
gb_model = grid_search.best_estimator_

# ===============================
# Prediction
# ===============================
y_pred = gb_model.predict(X_test)

# ===============================
# Performance Metrics
# ===============================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print("\n========== Gradient Boosting Performance ==========")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\n========== Classification Report ==========")
print(classification_report(y_test, y_pred))

print("\n========== Confusion Matrix ==========")
print(confusion_matrix(y_test, y_pred))