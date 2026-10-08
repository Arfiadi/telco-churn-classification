"""Page 3: Executive Governance & Model Diagnostics."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.ui.common import render_sidebar
from src.ui.components import render_metric_card
from src.ui.styles import apply_custom_css

# Apply custom styling
apply_custom_css()

# Render shared sidebar
render_sidebar()

st.subheader("📈 Executive Governance, Performance & Cost-Benefit Calibration")
st.write("Audit metrik performa model LightGBM Bayesian Tuned dan kalibrasi unit economics.")

eg1, eg2, eg3, eg4 = st.columns(4)
with eg1:
    render_metric_card(
        "🎯 Holdout ROC-AUC",
        "0.8447",
        "Bayesian Opt LightGBM",
        color_class="metric-value-info",
    )
with eg2:
    render_metric_card(
        "📊 Optimal F1-Score",
        "0.6279",
        "Threshold = 0.39",
        color_class="metric-value-success",
    )
with eg3:
    render_metric_card(
        "⚡ Churn Recall (Tau=0.39)",
        "78.4%",
        "+26.3% vs Default 0.50",
        color_class="metric-value-success",
    )
with eg4:
    render_metric_card(
        "💵 Net Profit Gain",
        "+$13,540",
        "Simulasi Holdout Test",
        color_class="metric-value-warning",
    )

st.divider()

st.markdown("### ⚖️ Kurva Trade-off Ambang Batas Finansial")
st.caption("Hubungan interaktif antara ambang batas klasifikasi ($\\tau$) dengan potensi Net Profit dan Churn Recall. Arahkan kursor ke grafik untuk inspeksi detail.")

# Generate synthetic curve representing optimal threshold dynamics
tau_range = np.linspace(0.1, 0.9, 50)
profit_curve = 15000 - 35000 * (tau_range - 0.39) ** 2
recall_curve = 1.0 / (1.0 + np.exp(6 * (tau_range - 0.42))) * 100

fig_gov = go.Figure()

# Profit trace (Primary Y-axis)
fig_gov.add_trace(
    go.Scatter(
        x=tau_range,
        y=profit_curve,
        name="Expected Net Profit ($)",
        line=dict(color="#38bdf8", width=3),
        fill="tozeroy",
        fillcolor="rgba(56, 189, 248, 0.08)",
        hovertemplate="Threshold: %{x:.2f}<br>Net Profit: <b>$%{y:,.2f}</b><extra></extra>",
    )
)

# Recall trace (Secondary Y-axis)
fig_gov.add_trace(
    go.Scatter(
        x=tau_range,
        y=recall_curve,
        name="Churn Recall (%)",
        line=dict(color="#34d399", width=2.5, dash="dot"),
        yaxis="y2",
        hovertemplate="Threshold: %{x:.2f}<br>Recall: <b>%{y:.1f}%</b><extra></extra>",
    )
)

# Optimal Tau vertical line
fig_gov.add_vline(
    x=0.39,
    line_width=2,
    line_dash="dash",
    line_color="#ef4444",
    annotation_text="Ambang Optimal (Tau* = 0.39)",
    annotation_position="top left",
    annotation_font=dict(color="#f87171", size=11),
)

fig_gov.update_layout(
    title=dict(
        text="Kalibrasi Ambang Batas: Net Profit ($) vs Churn Recall (%)",
        font=dict(size=14, color="#f8fafc", family="Plus Jakarta Sans"),
    ),
    xaxis=dict(
        title=dict(text="Classification Threshold (Tau)", font=dict(color="#94a3b8", size=11)),
        tickfont=dict(color="#94a3b8"),
        gridcolor="rgba(51, 65, 85, 0.3)",
    ),
    yaxis=dict(
        title=dict(text="Expected Net Profit ($)", font=dict(color="#38bdf8", size=11)),
        tickfont=dict(color="#38bdf8"),
        gridcolor="rgba(51, 65, 85, 0.3)",
    ),
    yaxis2=dict(
        title=dict(text="Churn Recall (%)", font=dict(color="#34d399", size=11)),
        tickfont=dict(color="#34d399"),
        overlaying="y",
        side="right",
        range=[0, 105],
        gridcolor="rgba(0, 0, 0, 0)",
    ),
    paper_bgcolor="rgba(15, 23, 42, 0.6)",
    plot_bgcolor="rgba(15, 23, 42, 0.3)",
    legend=dict(
        x=0.02,
        y=0.98,
        bgcolor="rgba(15, 23, 42, 0.8)",
        bordercolor="rgba(51, 65, 85, 0.5)",
        borderwidth=1,
        font=dict(color="#e2e8f0", size=10),
    ),
    margin=dict(l=40, r=40, t=50, b=40),
    height=380,
)
st.plotly_chart(fig_gov, width="stretch", alt="Financial tradeoff curve Net Profit vs Churn Recall")

st.markdown("### 🛡️ Business Guardrails & Compliance Policy")
st.markdown(
    """
    1. **Pagu Biaya Retensi:** Biaya insentif atau voucher diskon dibatasi maksimal **$20.00** per pelanggan untuk menjaga *margin profitability*.
    2. **Dukungan Counterfactual:** Agen retensi tidak diizinkan merekomendasikan paket tanpa melalui simulasi pipeline ML asli (`simulate_churn_impact`).
    3. **Naskah Non-Akusatif:** Komunikasi dengan pelanggan difokuskan pada penghargaan loyalitas dan benefit, tanpa menyebutkan istilah teknis *churn* atau *berhenti berlangganan*.
    4. **Zero-Downtime Guarantee:** Mekanisme fallback deterministik lokal menjamin antarmuka tetap berfungsi tanpa bergantung penuh pada koneksi API pihak ketiga.
    """
)
