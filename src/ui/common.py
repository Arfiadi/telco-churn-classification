"""Shared services, state management, and sidebar rendering for Streamlit pages."""

import json
from pathlib import Path
import streamlit as st

from src.services.agent_service import AgentService
from src.services.inference_service import get_inference_service
from src.services.shap_service import get_shap_service


@st.cache_resource
def _load_ml_resources():
    """Load core ML inference and SHAP explainability models (heavy resources)."""
    return get_inference_service(), get_shap_service()


def load_services():
    """Load core ML inference, SHAP, and dynamic agent services."""
    inf_svc, shap_svc = _load_ml_resources()
    agent_svc = AgentService()
    custom_key = st.session_state.get("custom_user_api_key")
    if custom_key and hasattr(agent_svc, "refresh_keys"):
        agent_svc.refresh_keys(api_key=custom_key)
    return inf_svc, shap_svc, agent_svc


@st.cache_data
def load_personas():
    """Load benchmark personas for quick evaluation."""
    persona_file = Path("tests/benchmark_personas.json")
    if persona_file.exists():
        with open(persona_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def render_sidebar():
    """Render the standard sidebar navigation and configuration widgets."""
    _, _, agent_svc = load_services()
    personas_data = load_personas()

    with st.sidebar:
        st.title("📡 Telco Copilot")
        st.caption("Intelligent Customer Retention Platform")
        st.divider()

        st.subheader("⚙️ Status Sistem ML")
        # API Connection Badge
        client_test = agent_svc._get_client()
        if client_test is not None:
            if getattr(agent_svc, "api_provider", "") == "gemini":
                st.success("🟢 Google Gemini Agent Online")
            else:
                st.success("🟢 OpenRouter Agent Online")
        else:
            st.warning("🟡 Heuristic Fallback Active")

        # LLM Key Configuration expander
        with st.expander("🔑 Konfigurasi API Key LLM", expanded=(client_test is None)):
            if client_test is not None:
                st.caption(
                    f"Provider aktif: **{agent_svc.api_provider.upper()}** "
                    f"(`{agent_svc.model_ladder[0]}`)"
                )
            else:
                st.caption(
                    "API Key belum terdeteksi di server/secrets. "
                    "Anda dapat memasukkan API Key di bawah untuk mengaktifkan AI Retensi:"
                )

            current_session_key = st.session_state.get("custom_user_api_key", "")
            user_input_key = st.text_input(
                "Gemini / OpenRouter API Key:",
                value=current_session_key,
                type="password",
                key="ui_key_input",
                placeholder="AQ... / AIza... / sk-or-...",
                help="Key disimpan secara aman hanya di memori sesi browser Anda.",
            )

            col_btn1, col_btn2 = st.columns([1, 1])
            with col_btn1:
                if st.button("Aktifkan AI", use_container_width=True, type="primary"):
                    if user_input_key.strip():
                        st.session_state["custom_user_api_key"] = user_input_key.strip()
                        if hasattr(agent_svc, "refresh_keys"):
                            agent_svc.refresh_keys(api_key=user_input_key.strip())
                        st.rerun()
            with col_btn2:
                if st.button("Reset Key", use_container_width=True):
                    st.session_state.pop("custom_user_api_key", None)
                    if hasattr(agent_svc, "refresh_keys"):
                        agent_svc.refresh_keys()
                    st.rerun()

            if client_test is None:
                st.markdown(
                    """
                    ---
                    💡 **Untuk Admin (Streamlit Cloud):**
                    Buka dashboard di [share.streamlit.io](https://share.streamlit.io) > **App Settings > Secrets**, tambahkan:
                    ```toml
                    GEMINI_API_KEY = "kunci_api_anda"
                    ```
                    Lalu save untuk mengaktifkan AI permanen tanpa input manual.
                    """
                )

        st.divider()

        st.subheader("🎯 Quick Preset Persona")
        persona_options = ["Custom Input"] + [
            f"{p['persona_id']}: {p['persona_name']}" for p in personas_data
        ]

        default_index = 1 if personas_data else 0

        selected_persona = st.selectbox(
            "Pilih Persona Evaluasi:",
            persona_options,
            index=default_index,
            key="selected_persona_name",
        )

        st.caption("Pagu Anggaran Maksimal: **$20.00 / pelanggan**")

        st.divider()

        with st.expander("⚙️ Parameter Teknis & Governance ML", expanded=False):
            st.markdown("**Core Model:** LightGBM Bayesian Opt")
            st.markdown("**Holdout ROC-AUC:** `0.8447`")
            st.markdown("**Ambang Finansial:** $\\tau^* = 0.39$")
            custom_threshold = st.slider(
                "Decision Threshold (Tau)",
                min_value=0.10,
                max_value=0.90,
                value=0.39,
                step=0.01,
                key="custom_threshold",
                help="Probabilitas di atas ambang ini memicu aksi retensi (INTERVENE). Nilai optimal finansial = 0.39.",
            )

    # Match selected profile if persona chosen
    preset_profile = None
    if selected_persona != "Custom Input":
        selected_id = selected_persona.split(":")[0]
        matched = [p for p in personas_data if p["persona_id"] == selected_id]
        if matched:
            preset_profile = matched[0]["profile"]

    return custom_threshold, preset_profile, selected_persona


def humanize_feature_name(raw_name: str) -> str:
    """Translates raw pipeline feature encoding names into clean, readable business labels."""
    name = raw_name.split("__")[-1].strip()
    mapping = {
        "Contract_Month-to-month": "Kontrak Bulanan",
        "Contract_One year": "Kontrak 1 Tahun",
        "Contract_Two year": "Kontrak 2 Tahun",
        "tenure": "Masa Langganan (Tenure)",
        "MonthlyCharges": "Tagihan Bulanan",
        "TotalCharges": "Total Pengeluaran",
        "OnlineSecurity_No": "Tanpa Online Security",
        "OnlineSecurity_Yes": "Online Security Aktif",
        "OnlineSecurity_No internet service": "Tanpa Layanan Internet",
        "TechSupport_No": "Tanpa Tech Support",
        "TechSupport_Yes": "Tech Support Aktif",
        "TechSupport_No internet service": "Tanpa Layanan Internet",
        "InternetService_Fiber optic": "Internet Fiber Optic",
        "InternetService_DSL": "Internet DSL",
        "InternetService_No": "Tanpa Layanan Internet",
        "PaymentMethod_Electronic check": "Bayar via Electronic Check",
        "PaymentMethod_Mailed check": "Bayar via Mailed Check",
        "PaymentMethod_Bank transfer (automatic)": "Transfer Bank Otomatis",
        "PaymentMethod_Credit card (automatic)": "Kartu Kredit Otomatis",
        "PaperlessBilling_Yes": "Tagihan Digital (Paperless)",
        "PaperlessBilling_No": "Tagihan Manual (Kertas)",
        "MultipleLines_Yes": "Multiple Lines Aktif",
        "MultipleLines_No": "Tanpa Multiple Lines",
        "MultipleLines_No phone service": "Tanpa Saluran Telepon",
        "OnlineBackup_No": "Tanpa Online Backup",
        "OnlineBackup_Yes": "Online Backup Aktif",
        "DeviceProtection_No": "Tanpa Proteksi Perangkat",
        "DeviceProtection_Yes": "Proteksi Perangkat Aktif",
        "StreamingTV_Yes": "Streaming TV Aktif",
        "StreamingTV_No": "Tanpa Streaming TV",
        "StreamingMovies_Yes": "Streaming Movies Aktif",
        "StreamingMovies_No": "Tanpa Streaming Movies",
        "SeniorCitizen_0": "Bukan Lansia (<65 thn)",
        "SeniorCitizen_1": "Warga Senior (Lansia)",
        "Partner_No": "Belum Berkeluarga",
        "Partner_Yes": "Memiliki Pasangan",
        "Dependents_No": "Tanpa Tanggungan",
        "Dependents_Yes": "Memiliki Tanggungan",
        "PhoneService_Yes": "Layanan Telepon Aktif",
        "PhoneService_No": "Tanpa Layanan Telepon",
    }
    return mapping.get(name, name.replace("_", " "))
