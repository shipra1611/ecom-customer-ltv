import sqlite3, pandas as pd
from config import *
if DB.exists(): DB.unlink()
con = sqlite3.connect(DB)
con.executescript((SQL / "01_schema.sql").read_text())
tx = pd.read_parquet(DATA / "transactions.parquet")
tx.to_sql("transactions", con, if_exists="append", index=False, chunksize=100000)
n = con.execute("SELECT COUNT(*), COUNT(DISTINCT customer_id), COUNT(DISTINCT order_id), ROUND(SUM(line_amount)) FROM transactions").fetchone()
print(f"loaded: {n[0]:,} lines | {n[1]:,} customers | {n[2]:,} orders | revenue {n[3]:,.0f}")
con.commit(); con.close()
