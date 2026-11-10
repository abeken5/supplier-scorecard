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
from src.finance.tco import calculate_supplier_tco

st.set_page_config(page_title="Tedarikçi Performans Paneli", layout="wide")

st.title("📊 Tedarikçi Performans ve Risk Skorlama Sistemi")
st.caption("Talep Planlama & Tedarik Zinciri Analitik Karar Destek Sistemi (DSS)")

# DataCo & Standart CSV Esnek Temizleme Fonksiyonu
def sanitize_df(df_input):
    df_clean = df_input.copy()
    df_clean.columns = df_clean.columns.astype(str).str.replace('\ufeff', '').str.strip()
    
    # 1. DataCo Supply Chain Dataset Otomatik Sütun Adaptörü
    dataco_mapping = {
        'Shipping Mode': 'supplier_name',
        'Category Name': 'product_name',
        'Order Item Quantity': 'order_quantity',
        'Sales per customer': 'unit_price',
        'Days for shipping (real)': 'actual_days',
        'Days for shipment (scheduled)': 'scheduled_days',
        'Late_risk': 'is_delayed_flag'
    }
    
    # Eğer DataCo sütunları tespit edilirse otomatik dönüştür
    if 'Shipping Mode' in df_clean.columns or 'Days for shipping (real)' in df_clean.columns:
        df_clean = df_clean.rename(columns=dataco_mapping)
        # DataCo için sentetik tarih türetme (gün farkları üzerinden)
        df_clean['order_date'] = pd.to_datetime('2026-01-01')
        df_clean['promised_delivery_date'] = df_clean['order_date'] + pd.to_timedelta(df_clean.get('scheduled_days', 5), unit='D')
        df_clean['actual_delivery_date'] = df_clean['order_date'] + pd.to_timedelta(df_clean.get('actual_days', 5), unit='D')
        df_clean['accepted_quantity'] = np.where(df_clean.get('is_delayed_flag', 0) == 1, df_clean['order_quantity'] * 0.95, df_clean['order_quantity'])

    # 2. Standart Sayısal ve Metin Temizliği
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object' or str(df_clean[col].dtype).startswith('string'):
            df_clean[col] = df_clean[col].astype(str).str.replace(',', '.').str.strip()
            
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

with st.sidebar.expander("ℹ️ Veri Formatı Gereksinimleri & DataCo Desteği"):
    st.markdown("""
    **Desteklenen Veri Formatları:**
    1. **Standart CSV Formatı**: `supplier_name`, `product_name`, `order_quantity`, `unit_price`, `promised_delivery_date`, `actual_delivery_date`.
    2. **DataCo Supply Chain Dataset (Kaggle)**: Kaggle açık kaynak veri setlerini yüklediğinizde sütunlar (`Shipping Mode`, `Days for shipping (real)`, `Sales per customer`) otomatik olarak adapte edilir.
    """)

# Veri Durumu ve Fail-Safe Kontrolü
has_financial_columns = False
if uploaded_file is not None:
    try:
        raw_uploaded = pd.read_csv(uploaded_file, sep=None, engine='python', encoding='utf-8-sig')
        df = sanitize_df(raw_uploaded)
        st.sidebar.success("✅ Özel Veri Seti Yüklendi!")
        
        fin_cols = ['freight_cost', 'customs_duty_rate', 'payment_terms_days']
        if any(c in df.columns for c in fin_cols):
            has_financial_columns = True
            st.sidebar.info("💡 Özel Finansal Parametreler Algılandı.")
        else:
            st.sidebar.warning("⚠️ TCO, varsayılan finansal parametrelerle hesaplandı.")
    except Exception as e:
        st.sidebar.error(f"⚠️ Dosya okuma hatası: {e}")
        df = default_df
else:
    df = default_df
    st.sidebar.info("ℹ️ Varsayılan demo veri seti aktif.")

st.sidebar.markdown("---")

# Sol Panel: Finansal Sermaye Maliyeti (WACC) Slider'ı
st.sidebar.header("💰 İşletme Sermayesi Ayarları")
wacc_rate = st.sidebar.slider("Ağırlıklı Ortalama Sermaye Maliyeti (WACC %)", min_value=10, max_value=80, value=40, step=5) / 100.0
st.sidebar.caption("Not: Yüksek enflasyon/faiz ortamında işletme sermayesi finansman maliyetini simüle eder.")

st.sidebar.markdown("---")

# Sol Panel: AHP Ağırlık Seçimleri
st.sidebar.header("⚖️ AHP Kriter Öncelikleri")
w_otif = st.sidebar.slider("OTIF Önemi", 1, 9, 5)
w_delay = st.sidebar.slider("Gecikme Günü Önemi", 1, 9, 3)
w_defect = st.sidebar.slider("Hata Oranı Önemi", 1, 9, 2)
w_price = st.sidebar.slider("Fiyat / TCO Önemi", 1, 9, 1)

# KPI & TCO Hesaplama
kpi_df = calculate_kpis(df)
tco_df = calculate_supplier_tco(df, wacc=wacc_rate)

# KPI ile TCO Verilerini Birleştirme
kpi_tco_df = pd.merge(kpi_df, tco_df, on='supplier_name', how='left')

# AHP & TOPSIS Hesaplaması (Global)
pairwise_matrix = np.array([
    [1.0,           w_otif/w_delay, w_otif/w_defect, w_otif/w_price],
    [w_delay/w_otif, 1.0,           w_delay/w_defect, w_delay/w_price],
    [w_defect/w_otif, w_defect/w_delay, 1.0,          w_defect/w_price],
    [w_price/w_otif,  w_price/w_delay,  w_price/w_defect,  1.0]
])

weights, cr = calculate_ahp_weights(pairwise_matrix)
matrix = np.array(kpi_tco_df[['otif_rate', 'avg_delay_days', 'defect_rate', 'unit_tco']].values, dtype=float)
is_benefit = [True, False, False, False] # Birim TCO düştükçe puan artar (Cost criterion)

topsis_scores = calculate_topsis(matrix, weights, is_benefit)
kpi_tco_df['TOPSIS Skor'] = topsis_scores
kpi_tco_df['TCO_Sıralama'] = kpi_tco_df['TOPSIS Skor'].rank(ascending=False, method='min').fillna(0).astype(int)

# Sadece Birim Fiyata Göre Sıralama (Karşılaştırma İçin)
kpi_tco_df['Fiyat_Sıralaması'] = kpi_tco_df['unit_price_avg'].rank(ascending=True, method='min').astype(int)

# Üst KPI Kartları
st.subheader("📈 Genel Tedarikçi KPI & TCO Özeti")
metric_cols = st.columns(max(1, len(kpi_tco_df)))
for i, row in kpi_tco_df.iterrows():
    with metric_cols[i]:
        st.metric(
            label=f"{row['supplier_name']} ({row.get('financial_health', 'B')})",
            value=f"Birim TCO: {row['unit_tco']:.2f} ₺",
            delta=f"Fiyat Farkı: +%{row['tco_premium_perc']:.1f}",
            delta_color="inverse"
        )

st.markdown("---")

# Sekme Mimarisi
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "💵 Fiyat vs TCO (Aha! Tablosu)",
    "🏆 TOPSIS & Skorlama", 
    "📦 ABC/XYZ Stok & Talep Analizi", 
    "🔮 Risk Tahmini",
    "🎲 Monte Carlo Stok Simülasyonu",
    "🧩 Sipariş Tahsis Önerisi"
])

with tab1:
    st.subheader("💡 Yönetici Özeti: Yalın Birim Fiyat vs. Toplam Sahip Olma Maliyeti (TCO)")
    st.caption("En ucuz teklifi veren tedarikçi; gizli lojistik, gümrük, kalite ve sermaye finansman maliyetleri eklendiğinde en pahalı seçeneğe dönüşebilir.")
    
    comp_df = kpi_tco_df.sort_values(by='unit_price_avg', ascending=True).copy()
    
    col_tco1, col_tco2 = st.columns([2, 1])
    with col_tco1:
        # Karşılaştırmalı Tablo
        display_comp = comp_df[[
            'supplier_name', 'Fiyat_Sıralaması', 'unit_price_avg', 
            'TCO_Sıralama', 'unit_tco', 'tco_premium_perc', 
            'avg_payment_terms', 'financial_health'
        ]].rename(columns={
            'supplier_name': 'Tedarikçi',
            'Fiyat_Sıralaması': 'Fiyat Sırası',
            'unit_price_avg': 'Birim Fiyat (₺)',
            'TCO_Sıralama': 'TCO Sırası',
            'unit_tco': 'Birim TCO (₺)',
            'tco_premium_perc': 'Gizli Maliyet Yükü (%)',
            'avg_payment_terms': 'Vade (Gün)',
            'financial_health': 'Finansal Sağlık'
        })
        st.dataframe(display_comp, use_container_width=True)

    with col_tco2:
        # Otomatik Yönetici Notu Üretimi
        top_price_supp = comp_df.iloc[0]
        top_tco_supp = kpi_tco_df.sort_values(by='unit_tco', ascending=True).iloc[0]
        
        st.info("📌 **Yönetici Çıkarım Raporu:**")
        if top_price_supp['supplier_name'] != top_tco_supp['supplier_name']:
            st.warning(
                f"🚨 **Stratejik Karar Çelişkisi!**\n\n"
                f"Yalın birim fiyata göre 1. sırada yer alan **{top_price_supp['supplier_name']}** ({top_price_supp['unit_price_avg']:.2f} ₺), "
                f"gizli lojistik, kalite ve sermaye maliyetleri eklendiğinde **Birim TCO'da {top_price_supp['TCO_Sıralama']}. sıraya** gerilemiştir.\n\n"
                f"🎯 **Tavsiye:** Gerçek TCO bazında en verimli tedarikçi **{top_tco_supp['supplier_name']}** ({top_tco_supp['unit_tco']:.2f} ₺) firmasıdır."
            )
        else:
            st.success(
                f"✅ **Karar Hizalanması:**\n\n"
                f"**{top_price_supp['supplier_name']}** hem en düşük birim fiyatı ({top_price_supp['unit_price_avg']:.2f} ₺) "
                f"hem de en düşük Birim TCO'yu ({top_price_supp['unit_tco']:.2f} ₺) sunarak her iki sıralamada da 1. sıradadır."
            )

    st.markdown("---")
    st.markdown("""
    > ⚖️ **Muhasebe & Vergi Notu:** Landed cost hesaplamasında indirilebilir KDV maliyet unsuruna dahil edilmemiş, gümrük vergisi ise ithalat maliyetine eklenmiştir.
    > 📑 **Veri Notu:** Demo veri sentetiktir. Parametreler örnek işletme değerleridir.
    """)

with tab2:
    summary_df = kpi_tco_df.sort_values(by='TOPSIS Skor', ascending=False)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Tedarikçi Sıralaması (TOPSIS)")
        st.dataframe(
            summary_df[['TCO_Sıralama', 'supplier_name', 'TOPSIS Skor', 'unit_tco', 'otif_rate', 'avg_delay_days', 'defect_rate']],
            use_container_width=True
        )

    with c2:
        st.subheader("🎯 AHP Tutarlılık Oranı (CR)")
        st.metric("AHP CR Değeri", f"{cr:.4f}")
        if cr < 0.10:
            st.success("✅ AHP matrisi tutarlı.")
        else:
            st.warning("⚠️ Kriter ağırlıkları tutarsız.")

with tab3:
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

with tab4:
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

with tab5:
    st.subheader("🎲 Monte Carlo Stoksuz Kalma (Stockout) Riski Simülasyonu")
    st.caption("1.000 Gelecek Senaryosu Üzerinden Olasılıksal Tedarik Zinciri Simülasyonu")

    col_mc1, col_mc2, col_mc3 = st.columns(3)
    with col_mc1:
        selected_supplier = st.selectbox("Tedarikçi Seçin", df['supplier_name'].unique())
    with col_mc2:
        current_stock = st.number_input("Mevcut Depo Stoğu (Adet)", min_value=10, max_value=5000, value=150)
    with col_mc3:
        daily_demand = st.number_input("Ortalama Günlük Talep (Adet/Gün)", min_value=1, max_value=500, value=15)

    if st.button("Simülasyonu Çalıştır (1.000 Senaryo)"):
        prob, simulated_demands = run_monte_carlo_stockout(df, selected_supplier, current_stock, daily_demand)
        
        col_res1, col_res2 = st.columns([1, 2])
        with col_res1:
            st.metric(label="Stok Kırılma (Stockout) Riski", value=f"%{prob*100:.1f}")
            if prob > 0.30:
                st.error("🚨 YÜKSEK STOK KIRILMA RİSKİ! Emniyet stoğunu artırın veya alternatif tedarikçiden sipariş açın.")
            else:
                st.success("✅ GÜVENLİ STOK SEVİYESİ. Mevcut stok teslimat süresini karşılamak için yeterli.")
        
        with col_res2:
            st.write("📊 **1.000 Senaryodaki Tahmini Toplam Talep Dağılımı**")
            chart_data = pd.DataFrame({'Tahmini Harcanacak Stok': simulated_demands})
            st.bar_chart(chart_data)

with tab6:
    st.subheader("🧩 Sipariş Tahsis (Order Allocation) Öneri Motoru")
    st.caption("TOPSIS skorları ve gecikme riski kısıtlarına göre optimum sipariş dağılımı")

    total_order = st.number_input("Verilecek Toplam Sipariş Miktarı (Adet)", min_value=100, max_value=100000, value=10000, step=500)

    if st.button("Optimum Tahsisi Hesapla"):
        alloc_df = calculate_order_allocation(kpi_tco_df, total_order)
        
        col_al1, col_al2 = st.columns([2, 1])
        with col_al1:
            st.dataframe(
                alloc_df[['supplier_name', 'TOPSIS Skor', 'avg_delay_days', 'Allocated_Ratio', 'Allocated_Qty']].style.format({
                    'TOPSIS Skor': '{:.4f}',
                    'avg_delay_days': '{:.1f}',
                    'Allocated_Ratio': '%{:.1%}'
                }),
                use_container_width=True
            )
        with col_al2:
            st.write("📊 **Sipariş Paylaşım Grafiği**")
            chart_alloc = alloc_df.set_index('supplier_name')['Allocated_Qty']
            st.bar_chart(chart_alloc)