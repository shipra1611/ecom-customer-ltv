"""6-month forward revenue (LTV) model.
Features: history up to LTV_CUTOFF. Target: revenue in the following 6 months (0 if the customer did not return).
Evaluation: 5-fold CV, every customer scored by a model that never saw them (out-of-fold). Hyperparameters were
chosen by comparing a handful of configs with this same CV, so numbers are mildly optimistic for XGBoost;
the config list is in the README."""
import sqlite3, numpy as np, pandas as pd, xgboost as xgb, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
from config import *

con = sqlite3.connect(DB)
df = pd.read_sql((SQL / "04_ltv_features.sql").read_text(), con, params={"cutoff": LTV_CUTOFF, "end": LTV_END})
feats = [c for c in df.columns if c not in ("customer_id", "future_rev")]
y = df.future_rev.to_numpy()
XGB = dict(objective="reg:tweedie", tweedie_variance_power=1.3, max_depth=3, n_estimators=300, learning_rate=0.03,
           min_child_weight=20, reg_lambda=10, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1)
RFM = ["recency", "frequency", "monetary"]

oof = {k: np.zeros(len(df)) for k in ["Mean baseline", "Linear (RFM)", "XGBoost"]}
for tr, te in KFold(5, shuffle=True, random_state=SEED).split(df):
    a, b = df.iloc[tr], df.iloc[te]
    oof["Mean baseline"][te] = a.future_rev.mean()
    oof["Linear (RFM)"][te] = np.clip(LinearRegression().fit(a[RFM], a.future_rev).predict(b[RFM]), 0, None)
    oof["XGBoost"][te] = xgb.XGBRegressor(**XGB).fit(a[feats], a.future_rev).predict(b[feats])

def top_share(p):
    d = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False)
    return y[d.to_numpy() == 9].sum() / y.sum()
rows = []
for k, p in oof.items():
    rows.append({"model": k, "RMSE": np.sqrt(mean_squared_error(y, p)), "MAE": mean_absolute_error(y, p),
                 "R2": r2_score(y, p), "spearman": spearmanr(y, p)[0] if k != "Mean baseline" else 0.0,
                 "top_decile_revenue_share": top_share(p) if k != "Mean baseline" else 0.10})
res = pd.DataFrame(rows).round(3)
print(f"customers={len(df):,} | zero future revenue: {(y==0).mean():.1%} | target mean {y.mean():.0f} median {np.median(y):.0f} max {y.max():,.0f}")
print(f"top 10 customers = {np.sort(y)[-10:].sum()/y.sum():.1%} of target revenue")
print(res.to_string(index=False)); res.to_csv(OUT / "ltv_metrics.csv", index=False)

# paired bootstrap over customers: XGBoost vs linear RFM
rng = np.random.default_rng(SEED); n = len(df); d_mae, d_top, d_rmse = [], [], []
ex, el = oof["XGBoost"], oof["Linear (RFM)"]
for _ in range(1000):
    i = rng.integers(0, n, n); yy = y[i]
    d_mae.append(np.abs(yy - ex[i]).mean() - np.abs(yy - el[i]).mean())
    d_rmse.append(np.sqrt(((yy - ex[i])**2).mean()) - np.sqrt(((yy - el[i])**2).mean()))
ci = lambda v: np.percentile(v, [2.5, 97.5])
mae_gain = 1 - mean_absolute_error(y, ex) / mean_absolute_error(y, el)
boot = pd.Series({"mae_xgb_minus_lin_mean": np.mean(d_mae), "mae_ci_lo": ci(d_mae)[0], "mae_ci_hi": ci(d_mae)[1],
                  "rmse_xgb_minus_lin_mean": np.mean(d_rmse), "rmse_ci_lo": ci(d_rmse)[0], "rmse_ci_hi": ci(d_rmse)[1],
                  "mae_improvement_pct": 100 * mae_gain})
print(f"\nXGBoost vs Linear RFM (xgb - linear, 95% bootstrap CI):\n  MAE  {boot.mae_xgb_minus_lin_mean:+.1f} [{boot.mae_ci_lo:+.1f}, {boot.mae_ci_hi:+.1f}]  ({mae_gain:.1%} lower MAE)"
      f"\n  RMSE {boot.rmse_xgb_minus_lin_mean:+.1f} [{boot.rmse_ci_lo:+.1f}, {boot.rmse_ci_hi:+.1f}]")
boot.to_csv(OUT / "ltv_bootstrap_xgb_vs_linear.csv")

df["pred_ltv_oof"] = oof["XGBoost"]
df["decile"] = pd.qcut(df.pred_ltv_oof.rank(method="first"), 10, labels=False) + 1
cap = df.groupby("decile").future_rev.sum(); print("revenue share by predicted decile:", (cap / cap.sum()).round(3).to_dict())
df[["customer_id", "pred_ltv_oof", "decile", "future_rev"] + feats].to_csv(TAB / "ltv_predictions.csv", index=False)
final = xgb.XGBRegressor(**XGB).fit(df[feats], y)
imp = pd.Series(final.feature_importances_, feats).sort_values(); imp.to_csv(OUT / "ltv_feature_importance.csv")
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
imp.plot.barh(ax=ax[0]); ax[0].set_title("Feature importance")
ax[1].scatter(df.pred_ltv_oof + 1, df.future_rev + 1, s=3, alpha=.3); ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("out-of-fold predicted (GBP, +1)"); ax[1].set_ylabel("actual 6-month revenue (GBP, +1)"); ax[1].set_title("Predicted vs actual")
fig.tight_layout(); fig.savefig(OUT / "ltv_model.png", dpi=150)
