import joblib, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from features import load_data, split

df, X, y = load_data()
_, Xte, _, yte = split(X, y)   # same random_state, so same test set as train.py
best = joblib.load("best_model.joblib")
pred = best.predict(Xte)

print(confusion_matrix(yte, pred))
print(classification_report(yte, pred))

imp = pd.Series(best.feature_importances_, index=X.columns).sort_values()
imp.plot.barh(title="Feature importance")
plt.tight_layout()
plt.savefig("importance.png", dpi=120)

fn = df.loc[Xte.index[((yte == 1) & (pred == 0)).values], "URL"]
print(len(fn))
print(fn.head(20).to_string())