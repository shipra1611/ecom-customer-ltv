"""Clean UCI Online Retail II into the `transactions` schema used by the rest of the pipeline.

Usage: python 01_prepare_real_data.py [path/to/online_retail_II.csv]
Default input: data/raw/online_retail_II.csv

Cleaning rules (all counted and printed):
  - drop rows with no Customer ID (guest checkouts, cannot be tracked)
  - drop cancellations (Invoice starting with 'C') and non-positive quantity or price
  - drop non-product stock codes (POST, D, M, BANK CHARGES, ADJUST, ...): keep 5-digit codes + optional letter suffix
  - drop 'A...' invoices (bad-debt adjustments)
Returns are removed, not netted, so monetary values are gross purchases.
"""
import sys, pandas as pd
from config import *

src = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA / "raw" / "online_retail_II.csv"
d = pd.read_csv(src, dtype={"Invoice": str, "StockCode": str}, parse_dates=["InvoiceDate"])
n0 = len(d); log = [("raw rows", n0)]
def step(label, mask):
    global d
    d = d[mask]; log.append((label, len(d)))

step("has Customer ID", d["Customer ID"].notna())
step("not a cancellation", ~d["Invoice"].str.startswith("C"))
step("numeric invoice (excl. 'A' adjustments)", d["Invoice"].str.fullmatch(r"\d+"))
step("quantity > 0 and price > 0", (d["Quantity"] > 0) & (d["Price"] > 0))
step("product stock codes only", d["StockCode"].str.fullmatch(r"\d{5}[A-Za-z]*"))
for l, n in log: print(f"{l:45s} {n:>10,}")

tx = pd.DataFrame({
    "order_id": d["Invoice"].astype("int64"),
    "customer_id": d["Customer ID"].astype("int64"),
    "order_date": d["InvoiceDate"].dt.strftime("%Y-%m-%d"),
    "sku_id": d["StockCode"].str.upper(),
    # no category field in this dataset: first 3 digits of the stock code act as a product-family proxy
    "category": d["StockCode"].str[:3],
    "quantity": d["Quantity"].astype("int64"),
    "unit_price": d["Price"].astype(float),
    "line_amount": (d["Quantity"] * d["Price"]).round(2),
})
tx.to_parquet(DATA / "transactions.parquet", index=False)
print(f"\nline_items={len(tx):,} orders={tx.order_id.nunique():,} customers={tx.customer_id.nunique():,} "
      f"| {tx.order_date.min()} to {tx.order_date.max()} | revenue GBP {tx.line_amount.sum():,.0f}")
