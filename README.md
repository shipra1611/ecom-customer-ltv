# Customer Segmentation, Cohort Retention and LTV Prediction (UCI Online Retail II)

Python, SQL (SQLite), XGBoost, scikit-learn, Tableau-ready CSVs. Real data: UCI Online Retail II, a UK online wholesaler, 2009-12-01 to 2011-12-09.

## Pipeline
| Step | Files | Output |
|---|---|---|
| Clean and load | `src/01_prepare_real_data.py`, `src/02_load_sql.py`, `sql/01_schema.sql` | `data/ecom.db` |
| RFM + K-means (k=5) | `sql/02_rfm.sql`, `src/03_rfm_kmeans.py` | `tableau/customer_segments.csv` |
| Cohort retention | `sql/03_cohort.sql`, `src/04_cohorts.py` | `tableau/cohort_retention.csv` |
| 6-month LTV (XGBoost) | `sql/04_ltv_features.sql`, `src/05_ltv_xgboost.py` | `tableau/ltv_predictions.csv` |

Run: `pip install -r requirements.txt`, put the CSV at `data/raw/online_retail_II.csv`, then `./run_all.sh`.

## Data after cleaning
1,067,371 raw rows -> **802,632 line items, 36,594 orders, 5,852 customers, GBP 17.4M gross revenue.**
Dropped: rows with no Customer ID (243K, guest checkouts), cancellations/returns (not netted, so revenue is gross purchases), bad-debt adjustment invoices, non-positive quantity or price, and non-product codes (POST, BANK CHARGES, etc.).
There is no category or discount field, so `category` is a product-family proxy (first 3 digits of the stock code) and there is no discount feature.

## Results (all from this run)
**Segmentation.** K-means k=5 on log-scaled RFM. Silhouette is 0.345 for k=5, 0.369 for k=4 and 0.349 for k=3, so k=5 is a usability choice, not the statistical optimum.

| Persona | Customers | Median recency (days) | Avg orders | Revenue share |
|---|---|---|---|---|
| Champions | 453 (7.7%) | 9 | 32.9 | 55.1% |
| Loyal Regulars | 1,273 | 31 | 9.1 | 26.8% |
| Fading Mid-Value | 1,338 | 259 | 3.9 | 11.1% |
| Recent / Developing | 1,137 | 29 | 2.5 | 4.4% |
| Lapsed One-Timers | 1,651 | 416 | 1.3 | 2.6% |

**Retention.** 24 monthly cohorts (Dec 2011 excluded as a partial month; the 2009-12 cohort is excluded from summary stats because it contains every customer already active when the data starts). Among 4,461 customers across 21 cohorts with M2 fully observed:
- 79.6% do not order in the calendar month after their first order (M1 retention 20.4%).
- 66.3% do not order in either of the next two months.
- Of customers who came back in M1, 59.5% did not order in M2.
- The relative fall in retention from M1 to M2 is **-6.8%** (M2 retention 21.5% is slightly higher than M1). Customers here reorder every few months and Q4 is seasonal, so a "Month-2 drop-off" in the usual sense does not exist in this data.

**LTV.** Target: revenue in the 6 months after 2011-06-09, from history up to that date. 4,944 customers, 48% have zero future revenue, 10 customers are 22.7% of target revenue (max GBP 168K). 5-fold cross-validation, every customer scored out-of-fold.

| Model | RMSE | MAE | R2 | Spearman | Top predicted decile's revenue share |
|---|---|---|---|---|---|
| Mean baseline | 5,511 | 1,213 | 0.00 | - | 10% |
| Linear (RFM) | 3,320 | 640 | 0.637 | 0.603 | 60.2% |
| XGBoost (Tweedie) | 3,542 | 600 | 0.587 | 0.629 | 61.5% |

XGBoost vs linear RFM: MAE is 6.2% lower but the 95% bootstrap CI is [-96, +25] GBP, and RMSE is +228 [-570, +1,196]. **It is a statistical tie.** Both explain about 60% of variance because `monetary` (past spend) dominates and a few very large wholesalers drive squared error.
Config selection: 6 XGBoost configs were compared with this same CV (squared-error depth 3/4, Tweedie power 1.3/1.5, depth 2, 600 trees, monotone constraints) and Tweedie 1.3 was kept, so XGBoost's numbers are slightly optimistic.

## Not in this version
- **Campaign lift.** The dataset has no campaign or send log, so persona-vs-generic conversion cannot be measured and is not claimed. The earlier simulated version is archived in `simulated/` and its numbers are properties of the generator.
- **Tableau workbook (.twbx).** Connect to the three CSVs in `tableau/`. Suggested sheets: persona size and revenue share (customer_segments), cohort retention heatmap (cohort_retention, month_offset on columns), predicted-decile vs actual revenue (ltv_predictions).

## Interview notes
- Say "6-month forward revenue", not "lifetime value": the horizon is six months.
- RMSE is in GBP and inflated by outliers; quote R2, MAE and the top-decile capture instead.
- Be ready to explain why XGBoost does not clearly beat a linear RFM model here (small n, heavy tail, spend history already carries most signal).
- Revenue excludes returns, so Champions' share is gross.
