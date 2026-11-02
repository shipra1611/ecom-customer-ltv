-- params: :end (last day of last full month)
WITH first_order AS (
  SELECT customer_id, strftime('%Y-%m', MIN(order_date)) AS cohort
  FROM transactions WHERE order_date <= :end GROUP BY customer_id
), activity AS (
  SELECT DISTINCT customer_id, strftime('%Y-%m', order_date) AS ym FROM transactions WHERE order_date <= :end
)
SELECT f.cohort,
       (CAST(substr(a.ym,1,4) AS INT) - CAST(substr(f.cohort,1,4) AS INT)) * 12
         + CAST(substr(a.ym,6,2) AS INT) - CAST(substr(f.cohort,6,2) AS INT) AS month_offset,
       COUNT(DISTINCT a.customer_id) AS active_customers
FROM first_order f JOIN activity a USING (customer_id)
GROUP BY f.cohort, month_offset
ORDER BY f.cohort, month_offset;
