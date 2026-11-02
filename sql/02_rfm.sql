-- params: :snap
SELECT customer_id,
       CAST(julianday(:snap) - julianday(MAX(order_date)) AS INTEGER) AS recency,
       COUNT(DISTINCT order_id) AS frequency,
       ROUND(SUM(line_amount), 2) AS monetary
FROM transactions
WHERE order_date <= :snap
GROUP BY customer_id;
