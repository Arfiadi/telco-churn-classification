"""Page 1: Single Customer Risk Profiler & Retention Copilot."""

import plotly.graph_objects as go
import streamlit as st

from src.services.agent_tools import (
    calculate_retention_roi,
    simulate_churn_impact,
)
from src.services.schemas import CustomerProfile
from src.ui.common import load_services, render_sidebar
from src.ui.components import (
    render_badge_card,
    render_decision_card,
    render_factor_card,
    render_metric_card,
    render_risk_gauge,
)
from src.ui.styles import apply_custom_css

# Apply custom styling
apply_custom_css()

# Render shared sidebar & load services
custom_threshold, preset_profile, selected_persona_name = render_sidebar()
inf_svc, shap_svc, agent_svc = load_services()

st.subheader("🔍 Profil Risiko Pelanggan & Rekomendasi Tindakan")


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
            contract = st.selectbox(
                "Contract Type",
                ["Month-to-month", "One year", "Two year"],
                index=["Month-to-month", "One year", "Two year"].index(get_val("Contract", "Month-to-month")),
            )

    with subtab_serv:
        c2_1, c2_2, c2_3 = st.columns(3)
        with c2_1:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=0 if get_val("PhoneService", "Yes") == "Yes" else 1)
            mult_lines = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"],
                index=["No", "Yes", "No phone service"].index(get_val("MultipleLines", "No")),
            )
            internet = st.selectbox(
                "Internet Service",
                ["Fiber optic", "DSL", "No"],
                index=["Fiber optic", "DSL", "No"].index(get_val("InternetService", "Fiber optic")),
            )
        with c2_2:
            sec = st.selectbox(
                "Online Security",
                ["No", "Yes", "No internet service"],
                index=["No", "Yes", "No internet service"].index(get_val("OnlineSecurity", "No")),
            )
            backup = st.selectbox(
                "Online Backup",
                ["No", "Yes", "No internet service"],
                index=["No", "Yes", "No internet service"].index(get_val("OnlineBackup", "No")),
            )
            device = st.selectbox(
                "Device Protection",
                ["No", "Yes", "No internet service"],
                index=["No", "Yes", "No internet service"].index(get_val("DeviceProtection", "No")),
            )
        with c2_3:
            tech = st.selectbox(
                "Tech Support",
                ["No", "Yes", "No internet service"],
                index=["No", "Yes", "No internet service"].index(get_val("TechSupport", "No")),
            )
            tv = st.selectbox(
                "Streaming TV",
                ["No", "Yes", "No internet service"],
                index=["No", "Yes", "No internet service"].index(get_val("StreamingTV", "No")),
            )
            movies = st.selectbox(
                "Streaming Movies",
                ["No", "Yes", "No internet service"],
                index=["No", "Yes", "No internet service"].index(get_val("StreamingMovies", "No")),
            )

    with subtab_fin:
        c3_1, c3_2 = st.columns(2)
        with c3_1:
            billing = st.selectbox("Paperless Billing", ["Yes", "No"], index=0 if get_val("PaperlessBilling", "Yes") == "Yes" else 1)
            payment = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(
                    get_val("PaymentMethod", "Electronic check")
                ),
            )
        with c3_2:
            monthly = st.number_input(
                "Monthly Charges ($)",
                min_value=18.0,
                max_value=150.0,
                value=float(get_val("MonthlyCharges", 85.0)),
                step=0.5,
            )
            calc_total = float(get_val("TotalCharges", monthly * max(tenure, 1)))
            total = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=calc_total, step=10.0)

# Build CustomerProfile
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

# Execute ML Inference & SHAP explanation
inf_result = inf_svc.predict_single(current_profile, custom_threshold=custom_threshold)
diagnostic = shap_svc.explain_customer(current_profile, top_k=3)

# Top KPI Metrics & Risk Indicator Display
st.markdown("### 📊 Status Risiko Pelanggan & Dampak Finansial")
col_gauge, col_kpis = st.columns([1.15, 1.85])

with col_gauge:
    render_risk_gauge(inf_result.churn_probability, threshold=custom_threshold)

with col_kpis:
    k1, k2, k3 = st.columns(3)
    with k1:
        render_badge_card(
            "🛡️ Tingkat Risiko",
            inf_result.risk_level,
            f"Ambang Batas: {custom_threshold * 100:.0f}%",
        )
    with k2:
        decision_sub = "Perlu Tindakan Retensi" if inf_result.decision == "INTERVENE" else "Akun Pelanggan Aman"
        render_decision_card(
            "🎯 Status Rekomendasi",
            inf_result.decision,
            decision_sub,
            is_intervene=(inf_result.decision == "INTERVENE"),
        )
    with k3:
        render_metric_card(
            "💰 Potensi Rugi / Thn",
            f"${inf_result.expected_loss:,.2f}",
            f"Nilai Kontrak: ${monthly * 12:,.1f}/thn",
            color_class="metric-value-warning",
        )

# Explainable AI & Factor Attribution Visualization
st.divider()
st.markdown("### 🔍 Analisis Faktor Pemicu & Penahan Churn")
st.caption("Identifikasi faktor utama yang mendorong pelanggan untuk berhenti berlangganan vs faktor yang menjaga loyalitas akun.")

col_shap_plot, col_shap_desc = st.columns([1.3, 1.0])

with col_shap_plot:
    all_factors = diagnostic.top_risk_drivers + diagnostic.top_retention_anchors
    if all_factors:
        sorted_factors = sorted(all_factors, key=lambda f: f.shap_value)
        names = [f.feature_name.split("__")[-1] for f in sorted_factors]
        vals = [f.shap_value for f in sorted_factors]
        colors = ["#f87171" if v > 0 else "#34d399" for v in vals]
        explanations = [f.human_explanation for f in sorted_factors]
        types = ["Pemicu Risiko (+)" if v > 0 else "Penjaga Loyalitas (-)" for v in vals]

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
                    + "Tingkat Pengaruh: <b>%{x:+.4f}</b><br>"
                    + "Kategori: %{customdata[0]}<br>"
                    + "<i>%{customdata[1]}</i><extra></extra>"
                ),
            )
        )

        fig_shap.update_layout(
            title=dict(
                text="Pengaruh Fitur terhadap Keputusan Pelanggan",
                font=dict(size=13, color="#f8fafc", family="Plus Jakarta Sans"),
            ),
            xaxis=dict(
                title=dict(text="Tingkat Pengaruh (Relatif)", font=dict(color="#94a3b8", size=11)),
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
        st.plotly_chart(fig_shap, width="stretch", alt="Grafik faktor penentu retensi pelanggan")

with col_shap_desc:
    st.markdown("**🚨 Faktor Utama Pemicu Risiko Churn:**")
    for rd in diagnostic.top_risk_drivers:
        feature_clean = rd.feature_name.split("__")[-1]
        render_factor_card(feature_clean, rd.shap_value, rd.human_explanation, is_risk=True)

    if diagnostic.top_retention_anchors:
        st.markdown("**🛡️ Faktor Utama Penjaga Loyalitas Pelanggan:**")
        for ra in diagnostic.top_retention_anchors:
            feature_clean = ra.feature_name.split("__")[-1]
            render_factor_card(feature_clean, ra.shap_value, ra.human_explanation, is_risk=False)


# -----------------------------------------------------
# WHAT-IF COUNTERFACTUAL SANDBOX (ISOLATED FRAGMENT)
# -----------------------------------------------------
st.divider()
st.markdown("### 🧪 Simulasi Intervensi Penawaran (What-If)")
st.caption("Uji coba dampak modifikasi penawaran kontrak atau paket proteksi terhadap penurunan risiko churn secara instan.")


@st.fragment
def render_whatif_sandbox(profile: CustomerProfile, baseline_monthly: float):
    w1, w2, w3 = st.columns(3)
    with w1:
        whatif_contract = st.selectbox("Simulasi Kontrak:", ["Sama dengan Profil", "One year", "Two year", "Month-to-month"], key="whatif_contract")
    with w2:
        whatif_tech = st.selectbox("Simulasi Tech Support:", ["Sama dengan Profil", "Yes", "No"], key="whatif_tech")
    with w3:
        whatif_security = st.selectbox("Simulasi Online Security:", ["Sama dengan Profil", "Yes", "No"], key="whatif_security")

    mods = {}
    if whatif_contract != "Sama dengan Profil":
        mods["Contract"] = whatif_contract
    if whatif_tech != "Sama dengan Profil":
        mods["TechSupport"] = whatif_tech
    if whatif_security != "Sama dengan Profil":
        mods["OnlineSecurity"] = whatif_security

    if mods:
        sim_out = simulate_churn_impact(profile, mods)
        sw1, sw2, sw3 = st.columns(3)
        sw1.metric(
            "Probabilitas Baru",
            f"{sim_out['simulated_churn_prob'] * 100:.1f}%",
            delta=f"-{sim_out['absolute_drop'] * 100:.1f}%",
            delta_color="inverse",
        )
        sw2.metric(
            "Penurunan Relatif",
            f"{sim_out['relative_drop_pct']:.1f}%",
            delta="Pengurangan Risiko",
        )
        net_eval = calculate_retention_roi(
            baseline_monthly,
            sim_out["baseline_churn_prob"],
            sim_out["simulated_churn_prob"],
            15.0,
        )
        sw3.metric(
            "Projected Net Value",
            f"${net_eval['projected_net_value_usd']:,.2f}",
            delta="Finansial Positif" if net_eval["is_profitable"] else "Negatif",
        )
    else:
        st.info("💡 Ubah salah satu opsi simulasi di atas untuk mengevaluasi dampak intervensi terhadap penurunan risiko churn.")


render_whatif_sandbox(current_profile, monthly)


# -----------------------------------------------------
# AGENTIC RETENTION COPILOT (ISOLATED FRAGMENT)
# -----------------------------------------------------
st.divider()
st.markdown("### 🤖 Asisten Rekomendasi Tindakan Retensi (AI Copilot)")
st.write("Rekomendasi paket retensi otomatis dengan efisiensi biaya intervensi optimal (pagu $\\le \\$20$), proyeksi penghematan (ROI), dan naskah komunikasi ramah pelanggan.")


@st.fragment
def render_copilot_section(profile: CustomerProfile, diag):
    if st.button("⚡ Buat Rencana Retensi Strategis", type="primary", key="btn_gen_plan"):
        with st.spinner("Mengorkestrasi Agen AI & Menghitung Optimasi Intervensi..."):
            plan = agent_svc.generate_retention_plan(profile, diag)

        source_label = (
            "🤖 OpenRouter AI Copilot"
            if plan.generation_source == "OPENROUTER_AGENT"
            else "🛡️ Offline Deterministic Heuristic Engine"
        )
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


render_copilot_section(current_profile, diagnostic)
