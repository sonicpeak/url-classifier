import pandas as pd, joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from xgboost import XGBClassifier
from features import load_data, split

def main():
    df, X, y = load_data()
    Xtr, Xte, ytr, yte = split(X, y)

    models = {
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "RF": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
        "XGB": XGBClassifier(n_estimators=300, max_depth=6, eval_metric="logloss", random_state=42),
    }

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    rows = []
    for name, m in models.items():
        cv_f1 = cross_val_score(m, Xtr, ytr, cv=cv, scoring="f1").mean()
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        pr, rc, f1, _ = precision_recall_fscore_support(yte, p, average="binary")
        rows.append(dict(model=name, acc=accuracy_score(yte, p), precision=pr,
                         recall=rc, f1=f1, cv_f1=cv_f1))
        print("done", name)

    results = pd.DataFrame(rows).sort_values("f1", ascending=False).round(4)
    print(results.to_markdown(index=False))

    best_name = results.iloc[0].model
    joblib.dump(models[best_name], "best_model.joblib", compress=3)
    print("saved", best_name)

if __name__ == "__main__":
    main()