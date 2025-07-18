import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import pickle

# Load dataset
df = pd.read_csv("data/crop_recommendation.csv")

# Features and target
X = df.drop("label", axis=1)
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data (important for boosting models)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Base models
rf = RandomForestClassifier(n_estimators=200, random_state=42)
et = ExtraTreesClassifier(n_estimators=200, random_state=42)
gb = GradientBoostingClassifier(n_estimators=200, random_state=42)
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
lgb_clf = lgb.LGBMClassifier(random_state=42)

# Train base models
rf.fit(X_train, y_train)
et.fit(X_train, y_train)
gb.fit(X_train, y_train)
xgb_clf.fit(X_train, y_train)
lgb_clf.fit(X_train, y_train)

# Evaluate each model
models = {"RandomForest": rf, "ExtraTrees": et, "GradientBoosting": gb, "XGBoost": xgb_clf, "LightGBM": lgb_clf}
for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Voting Classifier (Ensemble)
voting_clf = VotingClassifier(
    estimators=[
        ("rf", rf),
        ("et", et),
        ("gb", gb),
        ("xgb", xgb_clf),
        ("lgb", lgb_clf)
    ],
    voting="soft"
)

voting_clf.fit(X_train, y_train)

# Final accuracy
y_pred = voting_clf.predict(X_test)
print("\nFinal Ensemble Model Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save final model
with open("crop_recommendation/model.pkl", "wb") as f:
    pickle.dump(voting_clf, f)

print("✅ Ensemble Model Trained and Saved!")
