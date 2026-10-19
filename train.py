import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from dataprocessing import build_preprocessor, load_data, split_xy

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ---------- 1) data ----------
df = load_data()
X, y = split_xy(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ---------- 2) all the classification algorithms ----------
# class_weight="balanced" because churners are only ~26% of the data
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
    "SVM": SVC(probability=True, class_weight="balanced", random_state=42),
}

# ---------- 3) train + evaluate ----------
results = []
trained = {}

for name, clf in models.items():
    pipe = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", clf),
    ])
    pipe.fit(X_train, y_train)

    predictions = pipe.predict(X_test)
    probabilities = pipe.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1": f1_score(y_test, predictions),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
    })
    trained[name] = pipe
    print(f"done: {name}")

# ---------- 4) compare ----------
results_df = pd.DataFrame(results).sort_values("F1", ascending=False).reset_index(drop=True)
print("\n", results_df.round(3).to_string(index=False))

best_name = results_df.loc[0, "Model"]
print(f"\nBest model (by F1): {best_name}")

# ---------- 5) save ----------
results_df.to_csv(MODELS_DIR / "results.csv", index=False)
joblib.dump(trained[best_name], MODELS_DIR / "churn_model.joblib")
(MODELS_DIR / "best_model_name.txt").write_text(best_name)
