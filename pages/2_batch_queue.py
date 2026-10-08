"""Page 2: Batch Churn Analysis & Work Queue Priority."""

from pathlib import Path
import pandas as pd
import streamlit as st

from src.ui.common import load_services, render_sidebar
from src.ui.components import render_metric_card
from src.ui.styles import apply_custom_css

# Apply custom styling
apply_custom_css()

# Render shared sidebar & load services
custom_threshold, _, _ = render_sidebar()
inf_svc, _, _ = load_services()

st.subheader("📁 Batch Churn Analysis & Work Queue Priority Table")
st.write("Unggah file CSV pelanggan atau gunakan dataset bawaan untuk menghasilkan skor risiko massal dan menyusun antrean kerja retensi.")

col_btn1, col_btn2 = st.columns([1, 2])
with col_btn1:
    use_sample = st.button("📥 Muat Sampel Dataset (Top 50 Pelanggan)", key="btn_load_sample")

uploaded_file = st.file_uploader("Atau unggah file CSV pelanggan:", type=["csv"], key="batch_file_uploader")

df_to_process = None
if uploaded_file is not None:
    try:
        df_to_process = pd.read_csv(uploaded_file)
        st.success(f"Berhasil memuat file dengan {len(df_to_process)} baris.")
    except Exception as e:
        st.error(f"Gagal membaca file CSV: {e}")
elif use_sample:
    raw_csv_path = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    if raw_csv_path.exists():
        df_to_process = pd.read_csv(raw_csv_path).head(50)
        st.info("Memuat 50 baris pertama dari dataset Telco Churn.")
    else:
        st.error("File data sampel tidak ditemukan di path data/WA_Fn-UseC_-Telco-Customer-Churn.csv.")

if df_to_process is not None:
    with st.spinner("Menjalankan inferensi LightGBM batch & menyusun antrean prioritas..."):
        scored_df = inf_svc.predict_batch(df_to_process, custom_threshold=custom_threshold)

    # Summary Metrics
    b1, b2, b3, b4 = st.columns(4)
    total_cust = len(scored_df)
    high_risk_cnt = (scored_df["decision"] == "INTERVENE").sum()
    total_loss_at_risk = scored_df[scored_df["decision"] == "INTERVENE"]["expected_loss"].sum()

    with b1:
        render_metric_card(
            "👥 Total Pelanggan",
            f"{total_cust:,}",
            "Batch Terproses",
            color_class="metric-value-info",
        )
    with b2:
        pct_intervene = high_risk_cnt / max(total_cust, 1) * 100
        render_metric_card(
            "🚨 Perlu Intervensi",
            f"{high_risk_cnt:,}",
            f"{pct_intervene:.1f}% dari Total Batch",
            color_class="metric-value-danger",
        )
    with b3:
        render_metric_card(
            "💸 Total Loss at Risk",
            f"${total_loss_at_risk:,.2f}",
            "Potensi Kerugian Akun Kritis",
            color_class="metric-value-warning",
        )
    with b4:
        render_metric_card(
            "🎯 Decision Threshold",
            f"{custom_threshold:.2f}",
            "Ambang Kalibrasi",
            color_class="metric-value-info",
        )

    # Filter option
    risk_filter = st.selectbox(
        "Filter berdasarkan Tingkat Risiko:",
        ["SEMUA", "HIGH_RISK", "MODERATE_RISK", "LOW_RISK"],
        key="batch_risk_filter",
    )
    if risk_filter != "SEMUA":
        display_df = scored_df[scored_df["risk_level"] == risk_filter]
    else:
        display_df = scored_df

    # Columns to display
    show_cols = [
        c
        for c in [
            "customerID",
            "customer_id",
            "tenure",
            "Contract",
            "MonthlyCharges",
            "churn_probability",
            "decision",
            "risk_level",
            "expected_loss",
        ]
        if c in display_df.columns
    ]
    st.dataframe(display_df[show_cols], width="stretch")

    # Download Enriched CSV
    csv_data = scored_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="💾 Unduh Antrean Kerja Retensi (CSV)",
        data=csv_data,
        file_name="telco_retention_work_queue.csv",
        mime="text/csv",
        key="btn_download_batch_csv",
    )
