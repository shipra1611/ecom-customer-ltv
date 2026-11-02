DROP TABLE IF EXISTS transactions;
CREATE TABLE transactions (
  order_id INTEGER, customer_id INTEGER, order_date TEXT, sku_id TEXT, category TEXT,
  quantity INTEGER, unit_price REAL, line_amount REAL
);
CREATE INDEX idx_tx_cust ON transactions(customer_id);
CREATE INDEX idx_tx_date ON transactions(order_date);
