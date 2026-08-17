import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from xgboost import XGBClassifier

# ============================
# Load Dataset
# ============================

df = pd.read_csv("../dataset/new_battery_dataset.csv")

print("Dataset Loaded Successfully!")
print(df.head())

# ============================
# Convert SOH into Classes
# ============================

def battery_health(soh):
    if soh >= 90:
        return "Healthy"
    elif soh >= 80:
        return "Moderate"
    else:
        return "Poor"

df["BatteryHealth"] = df["SOH"].apply(battery_health)

print("\nBattery Health Distribution:")
print(df["BatteryHealth"].value_counts())

# ============================
# Features and Target
# ============================

X = df.drop(columns=["BatteryID", "BatchID", "SOH", "BatteryHealth"])
y = df["BatteryHealth"]

# ============================
# Label Encoding
# ============================

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print("\nClass Mapping:")
for i, label in enumerate(label_encoder.classes_):
    print(f"{label} -> {i}")

# ============================
# Train-Test Split
# ============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# ============================
# Train XGBoost Model
# ============================

xgb_model = XGBClassifier(
    objective="multi:softmax",
    num_class=3,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    eval_metric="mlogloss"
)

xgb_model.fit(X_train, y_train)

print("\nXGBoost Model Trained Successfully!")

# ============================
# Prediction
# ============================

y_pred = xgb_model.predict(X_test)

print("\nFirst 10 Predictions:")

prediction_df = pd.DataFrame({
    "Actual": label_encoder.inverse_transform(y_test[:10]),
    "Predicted": label_encoder.inverse_transform(y_pred[:10])
})

print(prediction_df)

# ============================
# Evaluation
# ============================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print("\n========== XGBoost Performance ==========")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# ============================
# Confusion Matrix
# ============================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# ============================
# Classification Report
# ============================

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
))