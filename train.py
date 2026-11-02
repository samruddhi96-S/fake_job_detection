"""
Train the Fake Job Detection model.
Run:  python train.py
"""
import os

import joblib
import matplotlib

matplotlib.use("Agg")  # no popup window, saves image directly
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

REAL_DATA = "data/fake_job_postings.csv"   # real Kaggle dataset (use this for final submission)
SAMPLE_DATA = "data/sample_jobs.csv"       # tiny demo dataset

# 1. Load data (real dataset if available, otherwise the sample one)
if os.path.exists(REAL_DATA):
    data_path = REAL_DATA
else:
    if not os.path.exists(SAMPLE_DATA):
        import subprocess
        import sys
        subprocess.run([sys.executable, "make_sample_data.py"], check=True)
    data_path = SAMPLE_DATA
print("Using dataset:", data_path)
df = pd.read_csv(data_path)

# 2. Combine all text columns into one column
text_cols = ["title", "company_profile", "description", "requirements", "benefits"]
df[text_cols] = df[text_cols].fillna("")
df["text"] = df[text_cols].agg(" ".join, axis=1)

X = df["text"]
y = df["fraudulent"]  # 0 = real, 1 = fake
print("Total rows:", len(df), "| Fake jobs:", int(y.sum()))

# 3. Train / test split (stratify keeps the fake:real ratio same in both)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. TF-IDF converts text into numbers
tfidf = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# 5. Compare 3 models (good table for your report)
models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
}
results = []
trained = {}
for name, clf in models.items():
    clf.fit(X_train_vec, y_train)
    pred = clf.predict(X_test_vec)
    f1 = f1_score(y_test, pred)
    results.append({"Model": name, "F1 (Fake class)": round(f1, 4)})
    trained[name] = clf
print("\nModel comparison:")
print(pd.DataFrame(results).to_string(index=False))

# 6. Final model = Logistic Regression (fast, works well with TF-IDF, gives probability)
final_model = trained["Logistic Regression"]
y_pred = final_model.predict(X_test_vec)
print("\nClassification report (Logistic Regression):")
print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

# 7. Confusion matrix image (use in report / PPT)
cm = confusion_matrix(y_test, y_pred)
os.makedirs("model", exist_ok=True)
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("model/confusion_matrix.png", dpi=150)

# 8. Save model so we do not retrain every time
joblib.dump(final_model, "model/model.pkl")
joblib.dump(tfidf, "model/tfidf.pkl")
print("\nSaved: model/model.pkl, model/tfidf.pkl, model/confusion_matrix.png")
