import streamlit as st
import pandas as pd
import numpy as np

import sys
import os

# src klasörünü yola ekleme
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.kpi.otif import calculate_kpis
from src.scoring.ahp_weights import calculate_ahp_weights
from src.scoring.topsis import calculate_topsis
from src.risk.delay_model import train_delay_model, predict_delay_risk

st.set_page_config(page_title="Tedarikçi Performans Paneli", layout="wide")

st.title("📊 Tedarikçi Performans ve Risk Skorlama Sistemi")
st.caption("Talep Planlama & Tedarik Zinciri Analitik Portfolyo Projesi")

# Veri Yükleme
@st.cache_data
def load_data():
    df = pd.read_csv("data/orders.csv")
    return df

df = load_data()

# Sol Panel: AHP Ağırlık Seçimleri
st.sidebar.header("⚖️ AHP Kriter Öncelikleri")
w_otif = st.sidebar.slider("OTIF Önemi", 1, 9, 5)
w_delay = st.sidebar.slider("Gecikme Günü Önemi", 1, 9, 3)
w_defect = st.sidebar.slider("Hata Oranı Önemi", 1, 9, 2)
w_price = st.sidebar.slider("Fiyat Önemi", 1, 9, 1)

# KPI Hesaplama
kpi_df = calculate_kpis(df)

# Üst KPI Kartları
st.subheader("📈 Genel Tedarikçi KPI Özeti")
cols = st.columns(len(kpi_df))
for i, row in kpi_df.iterrows():
    with cols[i]:
        st.metric(
            label=row['supplier_name'],
            value=f"OTIF: %{row['otif_rate']:.1f}",
            delta=f"Gecikme: {row['avg_delay_days']:.1f} Gün",
            delta_color="inverse"
        )

st.markdown("---")

# AHP & TOPSIS Skorlama
pairwise_matrix = np.array([
    [1.0,           w_otif/w_delay, w_otif/w_defect, w_otif/w_price],
    [w_delay/w_otif, 1.0,           w_delay/w_defect, w_delay/w_price],
    [w_defect/w_otif, w_defect/w_delay, 1.0,          w_defect/w_price],
    [w_price/w_otif,  w_price/w_delay,  w_price/w_defect,  1.0]
])

weights, cr = calculate_ahp_weights(pairwise_matrix)

# TOPSIS için Veri Hazırlığı
matrix = kpi_df[['otif_rate', 'avg_delay_days', 'defect_rate', 'total_orders']].values
is_benefit = [True, False, False, True]

topsis_scores = calculate_topsis(matrix, weights, is_benefit)
kpi_df['TOPSIS Skor'] = topsis_scores

# Sıralama hesabı (NaN korumalı)
kpi_df['Sıralama'] = kpi_df['TOPSIS Skor'].rank(ascending=False, method='min').fillna(0).astype(int)

summary_df = kpi_df.sort_values(by='TOPSIS Skor', ascending=False)

# Tablo Gösterimi
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🏆 Tedarikçi TOPSIS Sıralaması")
    st.dataframe(
        summary_df[['Sıralama', 'supplier_name', 'TOPSIS Skor', 'otif_rate', 'avg_delay_days', 'defect_rate']],
        use_container_width=True
    )

with col2:
    st.subheader("🎯 AHP Tutarlılık Oranı (CR)")
    st.metric("AHP CR Değeri", f"{cr:.4f}")
    if cr < 0.10:
        st.success("✅ AHP matrisi tutarlı.")
    else:
        st.warning("⚠️ Kriter ağırlıkları tutarsız, lütfen değerleri gözden geçirin.")

st.markdown("---")

# Makine Öğrenmesi Risk Tahmini
st.subheader("🔮 Yeni Sipariş Gecikme Riski Tahmini (Lojistik Regresyon)")

model, scaler = train_delay_model(df)

col_input1, col_input2 = st.columns(2)
with col_input1:
    req_qty = st.number_input("Sipariş Miktarı", min_value=10, max_value=5000, value=250)
with col_input2:
    req_lead = st.number_input("Planlanan Teslimat Süresi (Gün)", min_value=1, max_value=30, value=7)

if st.button("Risk Tahmini Yap"):
    prob = predict_delay_risk(model, scaler, req_qty, req_lead)
    st.write(f"Tahmini Gecikme Olasılığı: **%{prob*100:.1f}**")
    if prob > 0.5:
        st.error("⚠️ Yüksek Gecikme Riski!")
    else:
        st.success("✅ Düşük Gecikme Riski.")