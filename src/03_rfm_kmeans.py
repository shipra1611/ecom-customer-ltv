import sqlite3, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from config import *

con = sqlite3.connect(DB)
rfm = pd.read_sql((SQL / "02_rfm.sql").read_text(), con, params={"snap": SNAPSHOT})
X = StandardScaler().fit_transform(np.log1p(rfm[["recency","frequency","monetary"]]))

# k selection on a sample
samp = X[np.random.default_rng(SEED).choice(len(X), min(len(X), 5000), replace=False)]
sel = []
for k in range(3, 9):
    km = KMeans(k, n_init=10, random_state=SEED).fit(samp)
    sel.append((k, km.inertia_, silhouette_score(samp, km.labels_)))
sel = pd.DataFrame(sel, columns=["k","inertia","silhouette"]); print(sel.round(3).to_string(index=False))
sel.to_csv(OUT / "kmeans_selection.csv", index=False)

km = KMeans(5, n_init=20, random_state=SEED).fit(X)
rfm["cluster"] = km.labels_
cent = pd.DataFrame(km.cluster_centers_, columns=["R","F","M"])
cent["R"] = -cent["R"]  # higher = more recent
score = cent["R"] + cent["F"] + cent["M"]
names, left = {}, set(range(5))
def take(label, key):
    i = max(left, key=key); names[i] = label; left.remove(i)
take("Champions", lambda i: score[i])
take("Lapsed One-Timers", lambda i: -score[i])
take("Loyal Regulars", lambda i: cent.F[i] + cent.R[i])
take("Recent / Developing", lambda i: cent.R[i])
take("Fading Mid-Value", lambda i: 0)
rfm["persona"] = rfm["cluster"].map(names)

prof = rfm.groupby("persona").agg(customers=("customer_id","count"), recency=("recency","median"),
        frequency=("frequency","mean"), monetary=("monetary","mean"), revenue=("monetary","sum")).round(1)
prof["rev_share_pct"] = (100 * prof["revenue"] / prof["revenue"].sum()).round(1)
print(prof.sort_values("revenue", ascending=False)); prof.to_csv(OUT / "persona_summary.csv")
rfm.to_csv(TAB / "customer_segments.csv", index=False)

fig, ax = plt.subplots(figsize=(6,4.5))
for p, g in rfm.sample(min(len(rfm), 4000), random_state=1).groupby("persona"):
    ax.scatter(np.log1p(g.recency), np.log1p(g.monetary), s=4, alpha=.5, label=p)
ax.set_xlabel("log(1+recency days)"); ax.set_ylabel("log(1+monetary)"); ax.legend(fontsize=7, markerscale=3)
fig.tight_layout(); fig.savefig(OUT / "personas_scatter.png", dpi=150)
