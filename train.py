import os
import numpy as np
import joblib

from utils import process_video
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# ================= PATH =================
DATASET_PATH = "data/ISL_VIDEO"

# ================= LOAD DATA =================
X, y = [], []

for label in os.listdir(DATASET_PATH):
    label_path = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(label_path):
        continue

    for video in os.listdir(label_path):
        video_path = os.path.join(label_path, video)
        X.append(process_video(video_path))
        y.append(label)

X = np.array(X)
encoder = LabelEncoder()
y = encoder.fit_transform(y)

os.makedirs("features", exist_ok=True)
np.save("features/X.npy", X)
np.save("features/y.npy", y)

# ================= SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ================= MODELS =================
models = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ]),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf"))
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier())
    ]),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, max_depth=30, random_state=42, n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingClassifier()
}

# ================= TRAIN & EVALUATE =================
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")

    results[name] = {"accuracy": acc, "f1": f1}
    print(f"{name} | Accuracy: {acc:.4f} | F1: {f1:.4f}")

# ================= BEST MODEL =================
best_model_name = max(results, key=lambda x: results[x]["f1"])
best_model = models[best_model_name]

print("\nFINAL BEST MODEL:", best_model_name)
print("Metrics:", results[best_model_name])

os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(encoder, "models/label_encoder.pkl")

print("\nModel saved successfully.")
