-- 1. Tedarikçi Bazında OTIF ve Kalite Metrikleri
SELECT 
    supplier_id,
    supplier_name,
    COUNT(order_id) AS total_orders,
    ROUND(AVG(CASE WHEN actual_delivery_date <= promised_delivery_date AND accepted_quantity >= order_quantity THEN 1.0 ELSE 0.0 END) * 100, 2) AS otif_rate_pct,
    ROUND(AVG(JULIANDAY(actual_delivery_date) - JULIANDAY(promised_delivery_date)), 2) AS avg_delay_days,
    ROUND((SUM(order_quantity - accepted_quantity) * 1.0 / SUM(order_quantity)) * 100, 2) AS defect_rate_pct
FROM orders
GROUP BY supplier_id, supplier_name;

-- 2. Window Functions (Pencere Fonksiyonları) ile Tedarikçi Teslimat Trend Analizi
SELECT 
    supplier_name,
    order_date,
    actual_delivery_date,
    promised_delivery_date,
    (JULIANDAY(actual_delivery_date) - JULIANDAY(promised_delivery_date)) AS delay_days,
    -- Hareketli Ortalama Gecikme (Son 3 siparişin trendi)
    AVG(JULIANDAY(actual_delivery_date) - JULIANDAY(promised_delivery_date)) OVER (
        PARTITION BY supplier_id 
        ORDER BY order_date 
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS rolling_avg_delay
FROM orders;