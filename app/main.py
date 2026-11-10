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
from src.analytics.abc_xyz import calculate_abc_xyz
from src.analytics.monte_carlo import run_monte_carlo_stockout
from src.analytics.allocation import calculate_order_allocation

st.set_page_config(page_title="Tedarikçi Performans Paneli", layout="wide")

st.title("📊 Tedarikçi Performans ve Risk Skorlama Sistemi")
st.caption("Talep Planlama & Tedarik Zinciri Analitik Portfolyo Projesi")

# PyArrow Tiplerini Saf Standart Pandas/NumPy Yapma Fonksiyonu
def sanitize_df(df_input):
    df_clean = df_input.copy()
    df_clean.columns = df_clean.columns.astype(str).str.replace('\ufeff', '').str.strip()
    
    for col in df_clean.columns:
        # Metin/String ise virgülleri noktaya çevir
        if df_clean[col].dtype == 'object' or str(df_clean[col].dtype).startswith('string'):
            df_clean[col] = df_clean[col].astype(str).str.replace(',', '.').str.strip()
            
        # Sayısal dönüştürme yapmayı dene (errors='coerce' ile)
        converted = pd.to_numeric(df_clean[col], errors='coerce')
        if not converted.isna().all():
            df_clean[col] = converted.fillna(0)
            
    return df_clean

@st.cache_data
def load_default_data():
    df_raw = pd.read_csv("data/orders.csv")
    return sanitize_df(df_raw)

# Sol Panel: Veri Yükleme Alanı
st.sidebar.header("📁 Veri Kaynağı")
uploaded_file = st.sidebar.file_uploader("Kendi Veri Setini Yükle (CSV)", type=["csv"])

# Örnek Şablon İndirme Alanı
default_df = load_default_data()
csv_template = default_df.to_csv(index=False, sep=';', encoding='utf-8-sig').encode('utf-8-sig')

st.sidebar.download_button(
    label="📥 Örnek Şablonu İndir (CSV)",
    data=csv_template,
    file_name="ornek_tedarikci_verisi.csv",
    mime="text/csv",
    help="Kendi verinizi bu CSV formatına uygun olarak yükleyebilirsiniz."
)

with st.sidebar.expander("ℹ️ Veri Formatı Gereksinimleri"):
    st.markdown("""
    Yükleyeceğiniz CSV dosyasında şu sütunlar bulunmalıdır:
    - **order_id**: Sipariş Numarası
    - **supplier_name**: Tedarikçi Adı
    - **product_name**: Ürün Adı
    - **order_date**: Sipariş Tarihi (YYYY-AA-GG)
    - **promised_date**: Planlanan Teslimat Tarihi
    - **actual_delivery_date**: Gerçekleşen Teslimat Tarihi
    - **order_quantity**: Sipariş Miktarı (Adet)
    - **accepted_quantity**: Kabul Edilen Miktar
    - **unit_price**: Birim Fiyat (TL/$)
    """)

# Veri Durumu Kontrolü
if uploaded_file is not None:
    try:
        raw_uploaded = pd.read_csv(uploaded_file, sep=None, engine='python', encoding='utf-8-sig')
        df = sanitize_df(raw_uploaded)
        st.sidebar.success("✅ Özel Veri Seti Yüklendi!")
    except Exception as e:
        st.sidebar.error(f"⚠️ Dosya okuma hatası: {e}")
        df = default_df
else:
    df = default_df
    st.sidebar.info("ℹ️ Varsayılan demo veri seti aktif.")

st.sidebar.markdown("---")

# Sol Panel: AHP Ağırlık Seçimleri
st.sidebar.header("⚖️ AHP Kriter Öncelikleri")
w_otif = st.sidebar.slider("OTIF Önemi", 1, 9, 5)
w_delay = st.sidebar.slider("Gecikme Günü Önemi", 1, 9, 3)
w_defect = st.sidebar.slider("Hata Oranı Önemi", 1, 9, 2)
w_price = st.sidebar.slider("Fiyat Önemi", 1, 9, 1)

# KPI Hesaplama
kpi_df = calculate_kpis(df)

# AHP & TOPSIS Hesaplaması (Global)
pairwise_matrix = np.array([
    [1.0,           w_otif/w_delay, w_otif/w_defect, w_otif/w_price],
    [w_delay/w_otif, 1.0,           w_delay/w_defect, w_delay/w_price],
    [w_defect/w_otif, w_defect/w_delay, 1.0,          w_defect/w_price],
    [w_price/w_otif,  w_price/w_delay,  w_price/w_defect,  1.0]
])

weights, cr = calculate_ahp_weights(pairwise_matrix)
matrix = kpi_df[['otif_rate', 'avg_delay_days', 'defect_rate', 'total_orders']].values
is_benefit = [True, False, False, True]

topsis_scores = calculate_topsis(matrix, weights, is_benefit)
kpi_df['TOPSIS Skor'] = topsis_scores
kpi_df['Sıralama'] = kpi_df['TOPSIS Skor'].rank(ascending=False, method='min').fillna(0).astype(int)

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

# Sekme Mimarisi
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 TOPSIS & Skorlama", 
    "📦 ABC/XYZ Stok & Talep Analizi", 
    "🔮 Risk Tahmini",
    "🎲 Monte Carlo Stok Simülasyonu",
    "🧩 Sipariş Tahsis Önerisi"
])

with tab1:
    summary_df = kpi_df.sort_values(by='TOPSIS Skor', ascending=False)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Tedarikçi Sıralaması")
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
            st.warning("⚠️ Kriter ağırlıkları tutarsız.")

with tab2:
    st.subheader("📌 Ürün Bazlı ABC/XYZ Matrisi")
    st.caption("A: Yüksek Ciro (%70), B: Orta Ciro (%20), C: Düşük Ciro (%10) | X: Kararlı Talep, Y: Sezonluk, Z: Düşük/Düzensiz Talep")
    
    abc_xyz_df = calculate_abc_xyz(df)
    
    col_abc1, col_abc2 = st.columns([2, 1])
    with col_abc1:
        st.dataframe(
            abc_xyz_df[['product_name', 'total_spend', 'abc_class', 'xyz_class', 'abc_xyz']],
            use_container_width=True
        )
    with col_abc2:
        st.info("💡 **Talep Planlama Notu:** AX sınıfı ürünleri tedarik eden firmaların gecikme yapmaması kritik önem taşır.")

with tab3:
    st.subheader("🔮 Yeni Sipariş Gecikme Riski Tahmini (Lojistik Regresyon)")
    model, scaler = train_delay_model(df)

    col