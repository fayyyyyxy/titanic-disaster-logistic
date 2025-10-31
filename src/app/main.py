# src/app/main.py
import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

TRAIN = "src/data/train.csv"
TEST  = "src/data/test.csv"
OUT   = "src/data/pred_test.csv"

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
target = "Survived"

def load_csv(path, tag):
    print(f"\n[LOAD-{tag}] {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    print(f"[LOAD-{tag}] shape={df.shape}")
    return df

def make_pipeline():
    num_cols = ["Pclass","Age","SibSp","Parch","Fare"]
    cat_cols = ["Sex","Embarked"]

    print("\n[PREP] Using features:", features)
    print("[PREP] Impute Age (median); One-hot encoding Sex & Embarked; keep Pclass/SibSp/Parch/Fare numeric.")

    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
    ])
    model = LogisticRegression(max_iter=1000)
    return Pipeline([("pre", pre), ("clf", model)])

def main():
    print("=== Titanic Prediction: ===")

    # 1) Load train
    train = load_csv(TRAIN, "TRAIN")

    # 2) Split X/y
    X_tr = train[features].copy()
    y_tr = train[target].astype(int)

    # 3) Build pipeline and fit
    pipe = make_pipeline()
    print("\n[TRAIN] Fitting model...")
    pipe.fit(X_tr, y_tr)
    print("[TRAIN] Fit complete.")

    # 4) Train accuracy
    yhat_tr = pipe.predict(X_tr)
    acc_tr = accuracy_score(y_tr, yhat_tr)
    print(f"[METRIC] TRAIN accuracy = {acc_tr:.4f}")

    # 5) Predict test and save
    test = load_csv(TEST, "TEST")
    X_te = test[features].copy()
    print("\n[TEST] Predicting on test.csv …")
    yhat_te = pipe.predict(X_te)
    print(f"[TEST] Predicted rows: {len(yhat_te)} | sample: {yhat_te[:10]}")

    out = pd.DataFrame({
        "PassengerId": test.get("PassengerId", pd.Series(range(1, len(yhat_te)+1))),
        "Survived": yhat_te.astype(int)
    })
    out.to_csv(OUT, index=False)
    print(f"[SAVE] Save predictions to {OUT}")
    print("\n[DONE] ")

if __name__ == "__main__":
    main()