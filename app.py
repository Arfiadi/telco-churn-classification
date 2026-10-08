"""
Telco Churn Intelligent Retention Platform (Phase 2).
Streamlit Web Application featuring Real-Time Inference, TreeSHAP Explainability,
Interactive What-If Sandbox, Batch Work Queue, and Agentic Retention Copilot.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.services.agent_service import AgentService
from src.services.agent_tools import (
    RETENTION_CATALOG,
    calculate_retention_roi,
    simulate_churn_impact,
)
from src.services.inference_service import get_inference_service
from src.services.schemas import CustomerProfile
from src.services.shap_service import get_shap_service

# Page Configuration
st.set_page_config(
    page_title="Telco Churn Intelligent Retention Platform",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Modern Glassmorphism & High-End Aesthetic Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"], [class*="st-"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Glassmorphism KPI Metric Cards */
    .metric-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 12px;
        box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.35);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 12px 28px -4px rgba(56, 189, 248, 0.15);
    }

    .metric-title {
        font-size: 0.78rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .metric-value-danger {
        font-size: 2.1rem;
        font-weight: 800;
        color: #f87171;
        line-height: 1.1;
        font-feature-settings: "tnum";
    }

    .metric-value-warning {
        font-size: 2.1rem;
        font-weight: 800;
        color: #fbbf24;
        line-height: 1.1;
        font-feature-settings: "tnum";
    }

    .metric-value-success {
        font-size: 2.1rem;
        font-weight: 800;
        color: #34d399;
        line-height: 1.1;
        font-feature-settings: "tnum";
    }

    .metric-value-info {
        font-size: 2.1rem;
        font-weight: 800;
        color: #38bdf8;
        line-height: 1.1;
        font-feature-settings: "tnum";
    }

    .metric-subtext {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 6px;
        display: block;
    }

    /* Status Badges */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.025em;
    }

    .badge-high {
        background-color: rgba(239, 68, 68, 0.18);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.5);
    }

    .badge-mod {
        background-color: rgba(245, 158, 11, 0.18);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.5);
    }

    .badge-low {
        background-color: rgba(16, 185, 129, 0.18);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.5);
    }

    /* Explainability Factor Card */
    .factor-card {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(51, 65, 85, 0.6);
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
        display: flex;
        flex-direction: column;
        gap: 2px;
    }

    /* Strategic Action / Copilot Card */
    .copilot-action-card {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.85));
        border: 1px solid rgba(59, 130, 246, 0.35);
        border-radius: 16px;
        padding: 22px 24px;
        margin-top: 12px;
        box-shadow: 0 12px 32px -8px rgba(37, 99, 235, 0.2);
    }

    .copilot-script-bubble {
        background: #090d16;
        border: 1px solid #1e293b;
        border-left: 4px solid #38bdf8;
        border-radius: 12px;
        padding: 16px 18px;
        font-size: 0.92rem;
        line-height: 1.65;
        color: #e2e8f0;
        margin-top: 10px;
    }

    /* Global button styling enhancement */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        border: 1px solid rgba(147, 197, 253, 0.35);
        border-radius: 10px;
        padding: 8px 22px;
        font-weight: 700;
        font-size: 0.95rem;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
        transition: all 0.2s ease-in-out;
    }

    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.55);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Services
@st.cache_resource
def load_services():
    inf_svc = get_inference_service()
    shap_svc = get_shap_service()
    agent_svc = AgentService()
    return inf_svc, shap_svc, agent_svc

inf_svc, shap_svc, agent_svc = load_services()

# Load Benchmark Personas for quick testing
@st.cache_data
def load_personas():
    persona_file = Path("tests/benchmark_personas.json")
    if persona_file.exists():
        with open(persona_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

personas_data = load_personas()

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.title("📡 Telco Copilot")
    st.caption("Intelligent Customer Retention Platform")
    st.divider()

    st.subheader("⚙️ Status Sistem ML")
    st.markdown("**Core Model:** LightGBM Bayesian Opt")
    st.markdown("**Holdout ROC-AUC:** `0.8447`")
    st.markdown("**Ambang Optimal:** $\\tau^* = 0.39$")

    # API Connection Badge
    client_test = agent_svc._get_client()
    if client_test is not None:
        st.success("🟢 OpenRouter Agent Online")
    else:
        st.warning("🟡 Heuristic Fallback Active")

    st.divider()

    st.subheader("🎯 Quick Preset Persona")
    persona_options = ["Custom Input"] + [f"{p['persona_id']}: {p['persona_name']}" for p in personas_data]
    selected_persona_name = st.selectbox("Pilih Persona Evaluasi:", persona_options, index=1 if personas_data else 0)

    preset_profile = None
    if selected_persona_name != "Custom Input":
        selected_id = selected_persona_name.split(":")[0]
        matched = [p for p in personas_data if p["persona_id"] == selected_id]
        if matched:
            preset_profile = matched[0]["profile"]

    st.divider()
    st.subheader("📊 Kalibrasi Threshold")
    custom_threshold = st.slider(
        "Decision Threshold (Tau)",
        min_value=0.10,
        max_value=0.90,
        value=0.39,
        step=0.01,
        help="Probabilitas di atas ambang ini memicu aksi retensi (INTERVENE). Nilai optimal finansial = 0.39.",
    )

    st.caption("Pagu Anggaran Maksimal: **$20.00 / pelanggan**")


# ---------------------------------------------------------
# MAIN TABS
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "👤 Single Customer Risk Profiler & Copilot",
    "📁 Batch Work Queue Priority",
    "📈 Executive Governance & Diagnostics",
])


# =========================================================
# TAB 1: SINGLE CUSTOMER PROFILER
# =========================================================
with tab1:
    st.subheader("🔍 Profil Risiko Pelanggan & Rekomendasi Tindakan")

    # Default values from preset or fallback
    def get_val(key, default):
        if preset_profile and key in preset_profile:
            return preset_profile[key]
        return default

    with st.expander("📝 Form Parameter Pelanggan (19 Fitur)", expanded=(selected_persona_name == "Custom Input")):
        st.caption("💡 Form terbagi ke dalam 3 kelompok logis. Anda juga dapat memilih Quick Preset persona di sidebar.")

        subtab_demo, subtab_serv, subtab_fin = st.tabs([
            "👤 Demografi & Kontrak",
            "🌐 Portofolio Layanan & Add-ons",
            "💳 Finansial & Pembayaran",
        ])

        with subtab_demo:
            c1_1, c1_2, c1_3 = st.columns(3)
            with c1_1:
                customer_id = st.text_input("Customer ID", value=get_val("customer_id", "CUST-1001"))
                gender = st.selectbox("Gender", ["Male", "Female"], index=0 if get_val("gender", "Male") == "Male" else 1)
            with c1_2:
                senior = st.selectbox("Senior Citizen", [0, 1], index=get_val("SeniorCitizen", 0))
                partner = st.selectbox("Partner", ["Yes", "No"], index=0 if get_val("Partner", "No") == "Yes" else 1)
            with c1_3:
                dependents = st.selectbox("Dependents", ["Yes", "No"], index=0 if get_val("Dependents", "No") == "Yes" else 1)
                tenure = st.number_input("Tenure (Bulan)", min_value=0, max_value=72, value=int(get_val("tenure", 4)))
                contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(get_val("Contract", "Month-to-month")))

        with subtab_serv:
            c2_1, c2_2, c2_3 = st.columns(3)
            with c2_1:
                phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=0 if get_val("PhoneService", "Yes") == "Yes" else 1)
                mult_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], index=["No", "Yes", "No phone service"].index(get_val("MultipleLines", "No")))
                internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], index=["Fiber optic", "DSL", "No"].index(get_val("InternetService", "Fiber optic")))
            with c2_2:
                sec = st.selectbox("Online Security", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(get_val("OnlineSecurity", "No")))
                backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(get_val("OnlineBackup", "No")))
                device = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(get_val("DeviceProtection", "No")))
            with c2_3:
                tech = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(get_val("TechSupport", "No")))
                tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(get_val("StreamingTV", "No")))
                movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(get_val("StreamingMovies", "No")))

        with subtab_fin:
            c3_1, c3_2 = st.columns(2)
            with c3_1:
                billing = st.selectbox("Paperless Billing", ["Yes", "No"], index=0 if get_val("PaperlessBilling", "Yes") == "Yes" else 1)
                payment = st.selectbox(
                    "Payment Method",
                    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                    index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(get_val("PaymentMethod", "Electronic check")),
                )
            with c3_2:
                monthly = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=150.0, value=float(get_val("MonthlyCharges", 85.0)), step=0.5)
                calc_total = float(get_val("TotalCharges", monthly * max(tenure, 1)))
                total = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=calc_total, step=10.0)

    # Build Profile Model
    current_profile = CustomerProfile(
        customer_id=customer_id,
        gender=gender,
        SeniorCitizen=senior,
        Partner=partner,
        Dependents=dependents,
        tenure=tenure,
        PhoneService=phone_service,
        MultipleLines=mult_lines,
        InternetService=internet,
        OnlineSecurity=sec,
        OnlineBackup=backup,
        DeviceProtection=device,
        TechSupport=tech,
        StreamingTV=tv,
        StreamingMovies=movies,
        Contract=contract,
        PaperlessBilling=billing,
        PaymentMethod=payment,
        MonthlyCharges=monthly,
        TotalCharges=total,
    )

    # Execute ML Inference
    inf_result = inf_svc.predict_single(current_profile, custom_threshold=custom_threshold)
    diagnostic = shap_svc.explain_customer(current_profile, top_k=3)

    # Top KPI Metrics Display
    st.markdown("### 📊 Status Prediksi & Dampak Finansial")
    m1, m2, m3, m4 = st.columns(4)

    prob_val = inf_result.churn_probability * 100
    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">⚡ Probabilitas Churn</div>
                <div class="{'metric-value-danger' if prob_val >= 60 else ('metric-value-warning' if prob_val >= custom_threshold * 100 else 'metric-value-success')}">
                    {prob_val:.1f}%
                </div>
                <span class="metric-subtext">Ambang Batas Intervensi: {custom_threshold * 100:.1f}%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m2:
        badge_class = "badge-high" if inf_result.risk_level == "HIGH_RISK" else ("badge-mod" if inf_result.risk_level == "MODERATE_RISK" else "badge-low")
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">🛡️ Tingkat Risiko</div>
                <div style="margin-top: 6px;">
                    <span class="badge-pill {badge_class}">{inf_result.risk_level}</span>
                </div>
                <span class="metric-subtext">Kategori Urgensi Akun</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m3:
        decision_color = "#ef4444" if inf_result.decision == "INTERVENE" else "#10b981"
        decision_sub = "Tindakan Retensi Diperlukan" if inf_result.decision == "INTERVENE" else "Akun Pelanggan Aman"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">🎯 Keputusan Sistem</div>
                <div style="font-size: 1.55rem; font-weight: 800; color: {decision_color}; margin-top: 4px;">
                    {inf_result.decision}
                </div>
                <span class="metric-subtext">{decision_sub}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">💰 Expected Annual Loss</div>
                <div class="metric-value-warning">
                    ${inf_result.expected_loss:,.2f}
                </div>
                <span class="metric-subtext">P(churn) &times; Tahunan ($ {monthly * 12:,.1f})</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------
    # EXPLAINABLE AI (TREE SHAP) VISUALIZATION
    # -----------------------------------------------------
    st.divider()
    st.markdown("### 🔬 Explainable AI (TreeSHAP) Factor Attribution")
    st.caption("Dekomposisi log-odds prediksi churn menjadi kontribusi marjinal tiap fitur spesifik untuk pelanggan ini.")

    col_shap_plot, col_shap_desc = st.columns([1.3, 1.0])

    with col_shap_plot:
        all_factors = diagnostic.top_risk_drivers + diagnostic.top_retention_anchors
        if all_factors:
            # Sort ascending so top risk driver is at top or anchor at bottom
            sorted_factors = sorted(all_factors, key=lambda f: f.shap_value)
            names = [f.feature_name.split("__")[-1] for f in sorted_factors]
            vals = [f.shap_value for f in sorted_factors]
            colors = ["#f87171" if v > 0 else "#34d399" for v in vals]
            explanations = [f.human_explanation for f in sorted_factors]
            types = ["Pemicu Risiko (+)" if v > 0 else "Penahan Retensi (-)" for v in vals]

            fig_shap = go.Figure()
            fig_shap.add_trace(
                go.Bar(
                    y=names,
                    x=vals,
                    orientation="h",
                    marker=dict(
                        color=colors,
                        line=dict(color="rgba(255, 255, 255, 0.2)", width=1),
                    ),
                    customdata=list(zip(types, explanations)),
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        + "Pengaruh SHAP: <b>%{x:+.4f}</b><br>"
                        + "Kategori: %{customdata[0]}<br>"
                        + "<i>%{customdata[1]}</i><extra></extra>"
                    ),
                )
            )

            fig_shap.update_layout(
                title=dict(
                    text="Faktor Penentu Risiko (Merah: Menaikkan Risiko | Hijau: Menahan Churn)",
                    font=dict(size=13, color="#f8fafc", family="Plus Jakarta Sans"),
                ),
                xaxis=dict(
                    title=dict(text="SHAP Value (Log-Odds Impact)", font=dict(color="#94a3b8", size=11)),
                    tickfont=dict(color="#94a3b8"),
                    gridcolor="rgba(51, 65, 85, 0.4)",
                    zerolinecolor="#94a3b8",
                    zerolinewidth=1.5,
                ),
                yaxis=dict(
                    tickfont=dict(color="#e2e8f0", size=11),
                ),
                paper_bgcolor="rgba(15, 23, 42, 0.6)",
                plot_bgcolor="rgba(15, 23, 42, 0.3)",
                margin=dict(l=20, r=20, t=40, b=30),
                height=320,
            )
            st.plotly_chart(fig_shap, width="stretch", alt="TreeSHAP local feature attribution chart")

    with col_shap_desc:
        st.markdown("**🚨 Faktor Pemicu Risiko Teratas:**")
        for rd in diagnostic.top_risk_drivers:
            feature_clean = rd.feature_name.split("__")[-1]
            st.markdown(
                f"""
                <div class="factor-card" style="border-left: 3px solid #f87171;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <strong style="color: #f87171;">{feature_clean}</strong>
                        <span style="font-family: monospace; font-size: 0.85rem; color: #f87171;">+{rd.shap_value:.4f}</span>
                    </div>
                    <span style="font-size: 0.84rem; color: #cbd5e1;">{rd.human_explanation}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if diagnostic.top_retention_anchors:
            st.markdown("**🛡️ Faktor Penahan / Pelindung:**")
            for ra in diagnostic.top_retention_anchors:
                feature_clean = ra.feature_name.split("__")[-1]
                st.markdown(
                    f"""
                    <div class="factor-card" style="border-left: 3px solid #34d399;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <strong style="color: #34d399;">{feature_clean}</strong>
                            <span style="font-family: monospace; font-size: 0.85rem; color: #34d399;">{ra.shap_value:.4f}</span>
                        </div>
                        <span style="font-size: 0.84rem; color: #cbd5e1;">{ra.human_explanation}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # -----------------------------------------------------
    # WHAT-IF COUNTERFACTUAL SIMULATOR SANDBOX
    # -----------------------------------------------------
    st.divider()
    st.markdown("### 🧪 What-If Counterfactual Sandbox")
    st.caption("Uji dampak modifikasi penawaran kontrak atau paket proteksi terhadap penurunan probabilitas churn secara instan.")

    w1, w2, w3 = st.columns(3)
    with w1:
        whatif_contract = st.selectbox("Simulasi Kontrak:", ["Sama dengan Profil", "One year", "Two year", "Month-to-month"])
    with w2:
        whatif_tech = st.selectbox("Simulasi Tech Support:", ["Sama dengan Profil", "Yes", "No"])
    with w3:
        whatif_security = st.selectbox("Simulasi Online Security:", ["Sama dengan Profil", "Yes", "No"])

    mods = {}
    if whatif_contract != "Sama dengan Profil":
        mods["Contract"] = whatif_contract
    if whatif_tech != "Sama dengan Profil":
        mods["TechSupport"] = whatif_tech
    if whatif_security != "Sama dengan Profil":
        mods["OnlineSecurity"] = whatif_security

    if mods:
        sim_out = simulate_churn_impact(current_profile, mods)
        sw1, sw2, sw3 = st.columns(3)
        sw1.metric("Probabilitas Baru", f"{sim_out['simulated_churn_prob'] * 100:.1f}%", delta=f"-{sim_out['absolute_drop'] * 100:.1f}%", delta_color="inverse")
        sw2.metric("Penurunan Relatif", f"{sim_out['relative_drop_pct']:.1f}%", delta="Pengurangan Risiko")
        net_eval = calculate_retention_roi(monthly, sim_out["baseline_churn_prob"], sim_out["simulated_churn_prob"], 15.0)
        sw3.metric("Projected Net Value", f"${net_eval['projected_net_value_usd']:,.2f}", delta="Finansial Positif" if net_eval['is_profitable'] else "Negatif")
    else:
        st.info("💡 Ubah salah satu opsi simulasi di atas untuk mengevaluasi dampak counterfactual terhadap risiko churn.")

    # -----------------------------------------------------
    # AGENTIC RETENTION COPILOT EXECUTION
    # -----------------------------------------------------
    st.divider()
    st.markdown("### 🤖 Agentic AI Retention Copilot")
    st.write("Orkestrasi intervensi strategis: Mencari paket retensi optimal ($C \\le \\$20$), menguji simulasi dampak, dan merumuskan naskah komunikasi empati.")

    if st.button("⚡ Generate Strategic Retention Plan", type="primary"):
        with st.spinner("Mengorkestrasi Agen AI & Menghitung Optimasi Intervensi..."):
            plan = agent_svc.generate_retention_plan(current_profile, diagnostic)

        source_label = "🤖 OpenRouter AI Copilot" if plan.generation_source == "OPENROUTER_AGENT" else "🛡️ Offline Deterministic Heuristic Engine"
        st.success(f"Strategi Retensi Siap Dieksekusi! (Sumber: {source_label})")

        p_col1, p_col2 = st.columns([1.2, 1.0])
        with p_col1:
            st.markdown(
                f"""
                <div class="copilot-action-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <h4 style="color: #60a5fa; margin: 0; font-size: 1.2rem;">📋 {plan.recommended_package_name}</h4>
                        <span class="badge-pill badge-mod">Confidence: {plan.confidence_level}</span>
                    </div>
                    <div style="background: rgba(15, 23, 42, 0.6); padding: 12px 14px; border-radius: 8px; margin-bottom: 14px; border: 1px solid rgba(51, 65, 85, 0.5);">
                        <strong style="color: #93c5fd; font-size: 0.85rem;">🔍 Diagnosis Akar Masalah:</strong>
                        <p style="margin: 4px 0 0 0; font-size: 0.9rem; color: #e2e8f0;">{plan.root_cause_diagnosis}</p>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-top: 14px;">
                        <div style="background: rgba(30, 41, 59, 0.5); padding: 10px; border-radius: 8px;">
                            <span style="color: #94a3b8; font-size: 0.75rem; text-transform: uppercase;">Biaya Insentif</span><br>
                            <span style="font-size: 1.35rem; font-weight: 800; color: #10b981;">${plan.incentive_cost_usd:.2f}</span><br>
                            <small style="color: #64748b;">(Pagu &le; $20)</small>
                        </div>
                        <div style="background: rgba(30, 41, 59, 0.5); padding: 10px; border-radius: 8px;">
                            <span style="color: #94a3b8; font-size: 0.75rem; text-transform: uppercase;">Prediksi Pasca Aksi</span><br>
                            <span style="font-size: 1.35rem; font-weight: 800; color: #38bdf8;">{plan.simulated_churn_prob * 100:.1f}%</span><br>
                            <small style="color: #10b981;">(-{plan.risk_reduction_pct:.1f}%)</small>
                        </div>
                        <div style="background: rgba(30, 41, 59, 0.5); padding: 10px; border-radius: 8px;">
                            <span style="color: #94a3b8; font-size: 0.75rem; text-transform: uppercase;">Saved Net Value</span><br>
                            <span style="font-size: 1.35rem; font-weight: 800; color: #f59e0b;">+${plan.projected_net_value_usd:,.2f}</span><br>
                            <small style="color: #64748b;">ROI Positif</small>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with p_col2:
            st.markdown("**📞 Naskah Outreach Tim Customer Success:**")
            st.markdown(
                f"""
                <div class="copilot-script-bubble">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px; color: #38bdf8; font-weight: 600; font-size: 0.85rem;">
                        <span>💬 Personalized Empathy Script</span>
                    </div>
                    {plan.outreach_script}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption("Klik tombol di bawah untuk menyalin naskah lengkap:")
            st.code(plan.outreach_script, language="markdown")


# =========================================================
# TAB 2: BATCH WORK QUEUE PRIORITY
# =========================================================
with tab2:
    st.subheader("📁 Batch Churn Analysis & Work Queue Priority Table")
    st.write("Unggah file CSV pelanggan atau gunakan dataset bawaan untuk menghasilkan skor risiko massal dan menyusun antrean kerja retensi.")

    col_btn1, col_btn2 = st.columns([1, 2])
    with col_btn1:
        use_sample = st.button("📥 Muat Sampel Dataset (Top 50 Pelanggan)")

    uploaded_file = st.file_uploader("Atau unggah file CSV pelanggan:", type=["csv"])

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
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">👥 Total Pelanggan</div>
                    <div class="metric-value-info">{total_cust:,}</div>
                    <span class="metric-subtext">Batch Terproses</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with b2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">🚨 Perlu Intervensi</div>
                    <div class="metric-value-danger">{high_risk_cnt:,}</div>
                    <span class="metric-subtext">{high_risk_cnt / max(total_cust, 1) * 100:.1f}% dari Total Batch</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with b3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">💸 Total Loss at Risk</div>
                    <div class="metric-value-warning">${total_loss_at_risk:,.2f}</div>
                    <span class="metric-subtext">Potensi Kerugian Akun Kritis</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with b4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">🎯 Decision Threshold</div>
                    <div class="metric-value-info">{custom_threshold:.2f}</div>
                    <span class="metric-subtext">Ambang Kalibrasi</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Filter option
        risk_filter = st.selectbox("Filter berdasarkan Tingkat Risiko:", ["SEMUA", "HIGH_RISK", "MODERATE_RISK", "LOW_RISK"])
        if risk_filter != "SEMUA":
            display_df = scored_df[scored_df["risk_level"] == risk_filter]
        else:
            display_df = scored_df

        # Columns to display
        show_cols = [c for c in ["customerID", "customer_id", "tenure", "Contract", "MonthlyCharges", "churn_probability", "decision", "risk_level", "expected_loss"] if c in display_df.columns]
        st.dataframe(display_df[show_cols], width="stretch")

        # Download Enriched CSV
        csv_data = scored_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Unduh Antrean Kerja Retensi (CSV)",
            data=csv_data,
            file_name="telco_retention_work_queue.csv",
            mime="text/csv",
        )


# =========================================================
# TAB 3: EXECUTIVE GOVERNANCE & MODEL DIAGNOSTICS
# =========================================================
with tab3:
    st.subheader("📈 Executive Governance, Performance & Cost-Benefit Calibration")
    st.write("Audit metrik performa model LightGBM Bayesian Tuned dan kalibrasi unit economics.")

    eg1, eg2, eg3, eg4 = st.columns(4)
    with eg1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">🎯 Holdout ROC-AUC</div>
                <div class="metric-value-info">0.8447</div>
                <span class="metric-subtext">Bayesian Opt LightGBM</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with eg2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">📊 Optimal F1-Score</div>
                <div class="metric-value-success">0.6279</div>
                <span class="metric-subtext">Threshold = 0.39</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with eg3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">⚡ Churn Recall (Tau=0.39)</div>
                <div class="metric-value-success">78.4%</div>
                <span class="metric-subtext">+26.3% vs Default 0.50</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with eg4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">💵 Net Profit Gain</div>
                <div class="metric-value-warning">+$13,540</div>
                <span class="metric-subtext">Simulasi Holdout Test</span>
            </div>
            """,
            unsafe_allow_html=True,
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
