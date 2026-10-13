"""
CODSOFT Data Science Internship - Task 4: Sales Prediction using Python
Predict product sales from advertising spend on TV, Radio and Newspaper.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RS = 42
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------- 1. Load
df = pd.read_csv("advertising.csv")
print("Shape:", df.shape, "| Missing:", int(df.isnull().sum().sum()), "| Duplicates:", int(df.duplicated().sum()))
print(df.describe().round(2))

# ---------------------------------------------------------------- 2. EDA
print("\nCorrelation with Sales:\n", df.corr()["Sales"].drop("Sales").round(3))
sns.pairplot(df, x_vars=["TV", "Radio", "Newspaper"], y_vars="Sales", height=4, kind="reg",
             plot_kws={"line_kws": {"color": "red"}, "scatter_kws": {"alpha": 0.5}})
plt.savefig("scatter_vs_sales.png", dpi=110); plt.close()
plt.figure(figsize=(6, 5)); sns.heatmap(df.corr(), annot=True, cmap="YlGnBu")
plt.title("Correlation matrix"); plt.tight_layout(); plt.savefig("correlation.png", dpi=110); plt.close()

# ---------------------------------------------------------------- 3. Split
X, y = df[["TV", "Radio", "Newspaper"]], df["Sales"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RS)

# ---------------------------------------------------------------- 4. Statistical view (which channels matter?)
ols = sm.OLS(y_train, sm.add_constant(X_train)).fit()
print("\nOLS coefficients (training data):")
print(pd.DataFrame({"coef": ols.params, "p-value": ols.pvalues}).round(4))
print("Train R2: %.3f" % ols.rsquared)

# ---------------------------------------------------------------- 5. Compare models
def pipe(model): return Pipeline([("sc", StandardScaler()), ("m", model)])
models = {
    "Simple LR (TV only)": (LinearRegression(), ["TV"]),
    "Multiple LR (all 3)": (LinearRegression(), ["TV", "Radio", "Newspaper"]),
    "Multiple LR (TV+Radio)": (LinearRegression(), ["TV", "Radio"]),
    "Polynomial deg 2 (TV+Radio)": (Pipeline([("poly", PolynomialFeatures(2, include_bias=False)),
                                              ("sc", StandardScaler()), ("lr", Ridge(alpha=1))]), ["TV", "Radio"]),
    "Random Forest (all 3)": (RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=RS), ["TV", "Radio", "Newspaper"]),
    "Gradient Boosting (all 3)": (GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=2, random_state=RS), ["TV", "Radio", "Newspaper"]),
}
cv = KFold(5, shuffle=True, random_state=RS)
print("\n5-fold CV R2 on training data:")
cv_scores = {}
for name, (m, cols) in models.items():
    s = cross_val_score(pipe(m) if "Polynomial" not in name else m, X_train[cols], y_train, cv=cv, scoring="r2")
    cv_scores[name] = s.mean()
    print(f"  {name:30s} {s.mean():.4f} (+/- {s.std():.4f})")

# ---------------------------------------------------------------- 6. Test-set evaluation
print("\n=== Test set results (40 rows) ===")
rows, fitted = [], {}
for name, (m, cols) in models.items():
    p = (pipe(m) if "Polynomial" not in name else m).fit(X_train[cols], y_train)
    pred = p.predict(X_test[cols]); fitted[name] = (p, cols, pred)
    rows.append((name, mean_absolute_error(y_test, pred), np.sqrt(mean_squared_error(y_test, pred)), r2_score(y_test, pred)))
res = pd.DataFrame(rows, columns=["Model", "MAE", "RMSE", "R2"]).round(3)
print(res.to_string(index=False))

best = max(cv_scores, key=cv_scores.get)
print("\nBest model by cross-validation:", best)
_, _, pred = fitted[best]
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].scatter(y_test, pred, alpha=0.7); ax[0].plot([y.min(), y.max()], [y.min(), y.max()], "r--")
ax[0].set(xlabel="Actual sales", ylabel="Predicted sales", title=f"Predicted vs actual: {best}")
sns.histplot(y_test - pred, bins=15, kde=True, ax=ax[1]); ax[1].set_title("Residuals (test)")
plt.tight_layout(); plt.savefig("predictions.png", dpi=110); plt.close()

# ---------------------------------------------------------------- 7. Example prediction
new = pd.DataFrame([[150, 25, 30]], columns=["TV", "Radio", "Newspaper"])
p, cols, _ = fitted[best]
print("\nExample: TV=150, Radio=25, Newspaper=30 ->", round(float(p.predict(new[cols])[0]), 2), "units sold")
