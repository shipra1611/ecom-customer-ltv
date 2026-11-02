"""Synthetic e-commerce data generator (500K+ line items, 18 monthly cohorts).
Latent customer types drive behaviour; the analysis never sees them."""
import numpy as np, pandas as pd
from config import *

rng = np.random.default_rng(SEED)
N_CUST = 41000

# type: (share, p1, floor, decay, extra_orders, aov, disc_prob)
T = {
 "champion": (0.08, 0.85, 0.70, 0.30, 1.00, 1800, 0.10),
 "loyal":    (0.22, 0.55, 0.35, 0.40, 0.30,  900, 0.20),
 "bigbasket":(0.15, 0.35, 0.20, 0.50, 0.10, 2600, 0.15),
 "discount": (0.25, 0.40, 0.10, 0.90, 0.20,  600, 0.65),
 "onetime":  (0.30, 0.15, 0.02, 1.00, 0.00,  700, 0.30),
}
names = list(T)
share = np.array([T[n][0] for n in names])

w = np.linspace(1, 2.2, N_MONTHS); w /= w.sum()
cust = pd.DataFrame({
    "customer_id": np.arange(1, N_CUST + 1),
    "ltype": rng.choice(names, N_CUST, p=share),
    "cohort": rng.choice(N_MONTHS, N_CUST, p=w),
    "aov_mult": rng.lognormal(0, 0.35, N_CUST),
    "channel": rng.choice(["organic","paid","referral","email"], N_CUST, p=[.35,.35,.15,.15]),
})
par = lambda i: cust["ltype"].map({n: T[n][i] for n in names}).to_numpy()
p1, floor, decay, extra, aov, dprob = (par(i) for i in range(1, 7))

months = pd.period_range(START, periods=N_MONTHS, freq="M")
rows = []
for m in range(N_MONTHS):
    k = m - cust["cohort"].to_numpy()
    valid = k >= 0
    p = floor + (p1 - floor) * np.exp(-decay * np.clip(k - 1, 0, None))
    if months[m].month in (10, 11):
        p = np.minimum(p * 1.15, 0.98)
    active = valid & ((k == 0) | (rng.random(N_CUST) < p))
    n_ord = np.where(active, 1 + rng.poisson(extra), 0)
    idx = np.repeat(np.arange(N_CUST), n_ord)
    if len(idx) == 0: continue
    day = rng.integers(1, months[m].days_in_month + 1, len(idx))
    rows.append(pd.DataFrame({"ci": idx, "date": pd.to_datetime(
        {"year": months[m].year, "month": months[m].month, "day": day})}))
orders = pd.concat(rows, ignore_index=True)
orders["order_id"] = np.arange(1, len(orders) + 1)
ci = orders["ci"].to_numpy()
orders["order_value"] = aov[ci] * cust["aov_mult"].to_numpy()[ci] * rng.lognormal(0, 0.3, len(orders))
orders["n_lines"] = 1 + rng.poisson(2.4, len(orders))
orders["disc_order"] = rng.random(len(orders)) < dprob[ci]

li = np.repeat(np.arange(len(orders)), orders["n_lines"].to_numpy())
lines = orders.iloc[li].reset_index(drop=True)
cats = np.array(["Skincare","Haircare","Makeup","Wellness","Fragrance","Accessories"])
lines["category"] = rng.choice(cats, len(lines), p=[.28,.2,.18,.14,.12,.08])
lines["sku_id"] = [f"{c[:3].upper()}-{s:03d}" for c, s in zip(lines["category"], rng.integers(1, 80, len(lines)))]
lines["quantity"] = rng.choice([1,2,3], len(lines), p=[.78,.17,.05])
lines["discount_pct"] = np.where(lines["disc_order"], rng.choice([10,15,20,30], len(lines)), 0)
lines["line_amount"] = (lines["order_value"] / lines["n_lines"] * rng.lognormal(0, 0.25, len(lines))).round(2)
lines["unit_price"] = (lines["line_amount"] / lines["quantity"]).round(2)
lines["customer_id"] = cust["customer_id"].to_numpy()[lines["ci"].to_numpy()]
lines["order_date"] = lines["date"].dt.strftime("%Y-%m-%d")
tx = lines[["order_id","customer_id","order_date","sku_id","category","quantity",
            "unit_price","discount_pct","line_amount"]]
tx.to_parquet(DATA / "transactions.parquet", index=False)

# campaign log: 2026-05-15, 50/50 generic vs persona-tailored, customers active in prior 180d
CONV = {"champion": (.22,1.15), "loyal": (.14,1.35), "bigbasket": (.08,1.40), "discount": (.10,1.55), "onetime": (.03,1.30)}
snap = pd.Timestamp("2026-05-15")
last = tx.assign(d=pd.to_datetime(tx["order_date"])).groupby("customer_id")["d"].max()
elig = last[(snap - last).dt.days <= 180].index
c = cust.set_index("customer_id").loc[elig].reset_index()
c["variant"] = rng.choice(["generic","persona"], len(c))
base = c["ltype"].map({k: v[0] for k, v in CONV.items()}).to_numpy()
mult = np.where(c["variant"] == "persona", c["ltype"].map({k: v[1] for k, v in CONV.items()}), 1.0)
c["converted"] = (rng.random(len(c)) < base * mult).astype(int)
c["sent_date"] = "2026-05-15"
c[["customer_id","variant","sent_date","converted"]].to_parquet(DATA / "campaign_log.parquet", index=False)
cust[["customer_id","channel"]].to_parquet(DATA / "customers.parquet", index=False)
print(f"customers={N_CUST:,} orders={len(orders):,} line_items={len(tx):,} campaign_rows={len(c):,}")
