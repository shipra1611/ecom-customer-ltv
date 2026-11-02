import sqlite3, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from config import *
con = sqlite3.connect(DB)
c = pd.read_sql((SQL / "03_cohort.sql").read_text(), con, params={"end": COHORT_END})
size = c[c.month_offset == 0].set_index("cohort")["active_customers"]
c["cohort_size"] = c["cohort"].map(size)
c["retention"] = c["active_customers"] / c["cohort_size"]
c.to_csv(TAB / "cohort_retention.csv", index=False)
piv = c.pivot(index="cohort", columns="month_offset", values="retention")
print("cohorts:", piv.shape[0], "(2009-12 is the start-of-data cohort: it holds every already-active customer, so it is excluded from summary stats)")

# customer-level view: month 0 = calendar month of first order. Only cohorts whose M2 is fully observed.
a = pd.read_sql("SELECT DISTINCT customer_id, strftime('%Y-%m', order_date) AS ym FROM transactions WHERE order_date <= :e",
                con, params={"e": COHORT_END})
a["mi"] = a.ym.str[:4].astype(int) * 12 + a.ym.str[5:].astype(int)
first = a.groupby("customer_id").mi.min()
a["off"] = a.mi - a.customer_id.map(first)
act = a.groupby("customer_id").off.apply(set)
last_m0 = pd.Period(COHORT_END, "M").year * 12 + pd.Period(COHORT_END, "M").month - 2   # M2 must be <= last full month
d = pd.DataFrame({"first": first, "m1": act.map(lambda s: 1 in s), "m2": act.map(lambda s: 2 in s)})
d["cohort"] = d["first"].map(lambda m: f"{(m-1)//12}-{(m-1)%12+1:02d}")
d = d[(d["first"] <= last_m0) & (d.cohort != "2009-12")]
by = d.groupby("cohort").agg(n=("m1", "size"), m1=("m1", "mean"), m2=("m2", "mean"))
by["rel_fall_m1_to_m2"] = (by.m1 - by.m2) / by.m1
by["cond_drop_m1_to_m2"] = d.groupby("cohort").apply(lambda x: 1 - (x.m1 & x.m2).sum() / max(x.m1.sum(), 1), include_groups=False)
out = pd.Series({
  "customers_in_scope": len(d), "cohorts_in_scope": len(by),
  "no_return_in_month_1_pct": 100 * (1 - d.m1.mean()),
  "no_return_in_months_1_or_2_pct": 100 * (1 - (d.m1 | d.m2).mean()),
  "active_m1_but_not_m2_pct_pooled": 100 * (1 - (d.m1 & d.m2).sum() / d.m1.sum()),
  "active_m1_but_not_m2_pct_cohort_mean": 100 * by.cond_drop_m1_to_m2.mean(),
  "relative_fall_m1_to_m2_pct_cohort_mean": 100 * by.rel_fall_m1_to_m2.mean(),
  "m1_retention_pct": 100 * d.m1.mean(), "m2_retention_pct": 100 * d.m2.mean()}).round(1)
print(out.to_string()); out.to_csv(OUT / "month2_dropoff.csv", header=["value"])
by.round(4).to_csv(OUT / "cohort_m1_m2_by_cohort.csv")

fig, ax = plt.subplots(figsize=(8, 5.5))
im = ax.imshow(piv.iloc[:, :13].values, cmap="Blues", aspect="auto", vmin=0, vmax=.5)
ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index, fontsize=7); ax.set_xlabel("Months since first order")
fig.colorbar(im, label="share of cohort active"); fig.tight_layout(); fig.savefig(OUT / "cohort_heatmap.png", dpi=150)
