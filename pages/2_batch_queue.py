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
    use_sample = st.button("Muat sampel dataset (Top 50 pelanggan)", icon=":material/download:", key="btn_load_sample")

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

    # Customer Prioritization Mechanism (Poin 8)
    # Assign Action Priority Tier based on combination of Risk & Business Value (Expected Loss)
    def assign_priority(row):
        if row["decision"] == "INTERVENE" and row["expected_loss"] >= 500:
            return "P1 - Kritis (High Value)"
        elif row["decision"] == "INTERVENE":
            return "P2 - Prioritas Tinggi"
        elif row["risk_level"] == "MODERATE_RISK":
            return "P3 - Waspada"
        else:
            return "P4 - Monitoring Rutin"

    scored_df["priority_tier"] = scored_df.apply(assign_priority, axis=1)

    # Sort & Filter Controls
    col_filter, col_sort = st.columns([1, 1])
    with col_filter:
        risk_filter = st.selectbox(
            "Filter berdasarkan Kategori Urgensi:",
            ["SEMUA", "HIGH_RISK", "MODERATE_RISK", "LOW_RISK"],
            key="batch_risk_filter",
            help="Saring daftar akun berdasarkan tingkat risiko churn yang diprediksi model.",
        )
    with col_sort:
        sort_by = st.selectbox(
            "Urutkan Antrean Berdasarkan:",
            ["Potensi Kerugian (Expected Loss)", "Probabilitas Churn", "Tagihan Bulanan (Monthly Charges)"],
            index=0,
            key="batch_sort_by",
            help="Prioritaskan akun bernilai tinggi untuk memaksimalkan ROI penyelamatan retensi.",
        )

    # Apply Filtering
    if risk_filter != "SEMUA":
        display_df = scored_df[scored_df["risk_level"] == risk_filter].copy()
    else:
        display_df = scored_df.copy()

    # Apply Sorting
    if sort_by == "Potensi Kerugian (Expected Loss)":
        display_df = display_df.sort_values(by="expected_loss", ascending=False)
    elif sort_by == "Probabilitas Churn":
        display_df = display_df.sort_values(by="churn_probability", ascending=False)
    else:
        display_df = display_df.sort_values(by="MonthlyCharges", ascending=False)

    # Columns to display
    display_cols = [
        c
        for c in [
            "priority_tier",
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

    st.markdown("#### 📋 Antrean Kerja Tindakan Retensi")
    st.caption("Daftar terurut secara otomatis memprioritaskan akun dengan potensi kerugian bisnis terbesar.")
    
    st.dataframe(
        display_df[display_cols],
        column_config={
            "priority_tier": st.column_config.TextColumn("Prioritas Tindakan"),
            "customerID": st.column_config.TextColumn("Customer ID"),
            "customer_id": st.column_config.TextColumn("Customer ID"),
            "tenure": st.column_config.NumberColumn("Tenure (Bln)"),
            "Contract": st.column_config.TextColumn("Tipe Kontrak"),
            "MonthlyCharges": st.column_config.NumberColumn("Biaya Bulanan ($)", format="$%.2f"),
            "churn_probability": st.column_config.ProgressColumn(
                "Probabilitas Churn",
                help="Prediksi peluang pelanggan berhenti berlangganan",
                format="%.1f%%",
                min_value=0.0,
                max_value=1.0,
            ),
            "decision": st.column_config.TextColumn("Keputusan"),
            "risk_level": st.column_config.TextColumn("Tingkat Risiko"),
            "expected_loss": st.column_config.NumberColumn("Potensi Rugi/Thn", format="$%.2f"),
        },
        width="stretch",
        hide_index=True,
    )

    # Download Enriched CSV
    csv_data = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Unduh antrean kerja retensi terurut (CSV)",
        data=csv_data,
        file_name="telco_retention_work_queue.csv",
        mime="text/csv",
        icon=":material/download:",
        key="btn_download_batch_csv",
        help="Unduh data antrean kerja ini untuk ditugaskan langsung ke tim Customer Success.",
    )
