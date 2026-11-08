import sqlite3
import pandas as pd

# 1. Geçici SQLite veritabanı oluştur ve CSV'yi tabloya aktar
conn = sqlite3.connect(':memory:')
df = pd.read_csv('data/orders.csv')
df.to_sql('orders', conn, index=False, if_exists='replace')

# 2. SQL dosyasını oku ve çalıştır
with open('sql/otif_trend.sql', 'r', encoding='utf-8') as file:
    sql_script = file.read()

queries = sql_script.split(';')

print("=== 1. TEDARİKÇİ OTIF ÖZETİ (SQL) ===")
df_summary = pd.read_sql_query(queries[0], conn)
print(df_summary.to_string(index=False))

print("\n=== 2. HAREKETLİ GECİKME TRENDİ (WINDOW FUNCTIONS) ===")
df_trend = pd.read_sql_query(queries[1], conn)
print(df_trend.to_string(index=False))