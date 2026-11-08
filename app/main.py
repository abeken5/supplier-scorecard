import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Proje ana dizinini path'e ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.kpi.otif import calculate_kpis
from src.scoring.ahp_weights import calculate_ahp_weights
from src.scoring.topsis import run_topsis

st.set_page_config(page_title="Tedarikçi Performans & Risk Skorlama", layout="wide")

st.title("📊 Tedarikçi Performans ve Risk Skorlama Sistemi")
st.markdown("Talep Planlama & Tedarik Zinciri Analitik Portfolyo Projesi")

# Veri Yükleme
@st.cache_data
def load_data():
    return pd.read_csv("data/orders.csv")

df = load_data()

# Yan Panel: AHP Kriter Ağırlıkları Ayarı
st.sidebar.header("⚖️ AHP Kriter Öncelikleri")
w_otif = st.sidebar.slider("OTIF Önemi", 1, 9, 5)
w_delay = st.sidebar.slider("Gecikme Günü Önemi", 1, 9, 3)
w_defect = st.sidebar.slider("Hata Oranı Önemi", 1, 9, 2)
w_price = st.sidebar.slider("Fiyat Önemi", 1, 9, 4)

# KPI Hesaplama
kpi_df = calculate_kpis(df)

# Birim Fiyat Ortalama
price_df = df.groupby('supplier_name')['unit_price'].mean().reset_index()
summary_df = pd.merge(kpi_df, price_df, on='supplier_name')

# Metrik Kartları
st.header("📈 Genel Tedarikçi KPI Özeti")
cols = st.columns(len(summary_df))
for idx, row in summary_df.iterrows():
    with cols[idx]:
        st.metric(
            label=row['supplier_name'],
            value=f"OTIF: %{row['otif_rate']:.1f}",
            delta=f"Gecikme: {row['avg_delay_days']:.1f} Gün",
            delta_color="inverse"
        )

# AHP & TOPSIS Hesaplama
pairwise_matrix = [
    [1.0, w_otif/w_delay, w_otif/w_defect, w_otif/w_price],
    [w_delay/w_otif, 1.0, w_delay/w_defect, w_delay/w_price],
    [w_defect/w_otif, w_defect/w_delay, 1.0, w_defect/w_price],
    [w_price/w_otif, w_price/w_delay, w_price/w_defect, 1.0]
]

weights, cr = calculate_ahp_weights(pairwise_matrix)

decision_matrix = summary_df[['otif_rate', 'avg_delay_days', 'defect_rate', 'unit_price']].values
is_benefit = [True, False, False, False]

topsis_scores = run_topsis(decision_matrix, weights, is_benefit)
summary_df['TOPSIS Skor'] = topsis_scores
summary_df['Sıralama'] = summary_df['TOPSIS Skor'].rank(ascending=False, method='min').astype(int)

st.divider()

# Sıralama ve Duyarlılık Analizi Tablosu
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("🏆 TOPSIS Çok Kriterli Tedarikçi Sıralaması")
    st.dataframe(
        summary_df[['Sıralama', 'supplier_name', 'TOPSIS Skor', 'otif_rate', 'avg_delay_days', 'defect_rate', 'unit_price']]
        .sort_values('Sıralama'),
        use_container_width=True
    )

with col_right:
    st.subheader("📋 AHP Ağırlık Dağılımı")
    st.write(f"**Tutarlılık Oranı (CR):** `{cr:.4f}`")
    if cr < 0.10:
        st.success("Tutarlı Ağırlıklandırma ✅")
    else:
        st.warning("Tutarsız Ağırlıklandırma ⚠️")
        
    ahp_chart_data = pd.DataFrame({
        'Kriter': ['OTIF', 'Gecikme Günü', 'Hata Oranı', 'Birim Fiyat'],
        'Ağırlık': weights
    })
    st.bar_chart(ahp_chart_data.set_index('Kriter'))