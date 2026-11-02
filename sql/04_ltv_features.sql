-- params: :cutoff, :end   (features use history <= cutoff, target is revenue in (cutoff, end])
WITH orders AS (
  SELECT order_id, customer_id, order_date,
         SUM(line_amount) AS order_value,
         COUNT(*) AS n_lines,
         COUNT(DISTINCT category) AS n_cat
  FROM transactions GROUP BY order_id, customer_id, order_date
), hist AS (
  SELECT * FROM orders WHERE order_date <= :cutoff
), gaps AS (
  SELECT customer_id,
         AVG(julianday(order_date) - julianday(prev)) AS avg_gap
  FROM (SELECT customer_id, order_date,
               LAG(order_date) OVER (PARTITION BY customer_id ORDER BY order_date) AS prev
        FROM hist)
  WHERE prev IS NOT NULL GROUP BY customer_id
), feat AS (
  SELECT customer_id,
         CAST(julianday(:cutoff) - julianday(MAX(order_date)) AS INT) AS recency,
         CAST(julianday(:cutoff) - julianday(MIN(order_date)) AS INT) AS tenure,
         COUNT(*) AS frequency,
         SUM(order_value) AS monetary,
         AVG(order_value) AS aov,
         AVG(n_lines) AS avg_lines,
         MAX(n_cat) AS max_cat,
         SUM(CASE WHEN julianday(:cutoff) - julianday(order_date) <= 90 THEN 1 ELSE 0 END) AS orders_90d,
         SUM(CASE WHEN julianday(:cutoff) - julianday(order_date) <= 90 THEN order_value ELSE 0 END) AS spend_90d
  FROM hist GROUP BY customer_id
), target AS (
  SELECT customer_id, SUM(line_amount) AS future_rev
  FROM transactions WHERE order_date > :cutoff AND order_date <= :end GROUP BY customer_id
)
SELECT f.*, COALESCE(g.avg_gap, f.tenure) AS avg_gap, COALESCE(t.future_rev, 0) AS future_rev
FROM feat f LEFT JOIN gaps g USING (customer_id) LEFT JOIN target t USING (customer_id);
