"""Shared services, state management, and sidebar rendering for Streamlit pages."""

import json
from pathlib import Path
import streamlit as st

from src.services.agent_service import AgentService
from src.services.inference_service import get_inference_service
from src.services.shap_service import get_shap_service


@st.cache_resource
def load_services():
    """Load core ML inference, SHAP, and agent services."""
    inf_svc = get_inference_service()
    shap_svc = get_shap_service()
    agent_svc = AgentService()
    return inf_svc, shap_svc, agent_svc


@st.cache_data
def load_personas():
    """Load benchmark personas for quick evaluation."""
    persona_file = Path("tests/benchmark_personas.json")
    if persona_file.exists():
        with open(persona_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def init_session_state():
    """Initialize default session state keys."""
    if "custom_threshold" not in st.session_state:
        st.session_state.custom_threshold = 0.39
    if "selected_persona_name" not in st.session_state:
        st.session_state.selected_persona_name = "Custom Input"


def render_sidebar():
    """Render the standard sidebar navigation and configuration widgets."""
    init_session_state()
    _, _, agent_svc = load_services()
    personas_data = load_personas()

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
        persona_options = ["Custom Input"] + [
            f"{p['persona_id']}: {p['persona_name']}" for p in personas_data
        ]
        
        default_index = 0
        if st.session_state.selected_persona_name in persona_options:
            default_index = persona_options.index(st.session_state.selected_persona_name)
        elif personas_data:
            default_index = 1

        selected_persona = st.selectbox(
            "Pilih Persona Evaluasi:",
            persona_options,
            index=default_index,
            key="selected_persona_name",
        )

        st.divider()
        st.subheader("📊 Kalibrasi Threshold")
        st.slider(
            "Decision Threshold (Tau)",
            min_value=0.10,
            max_value=0.90,
            value=st.session_state.custom_threshold,
            step=0.01,
            key="custom_threshold",
            help="Probabilitas di atas ambang ini memicu aksi retensi (INTERVENE). Nilai optimal finansial = 0.39.",
        )

        st.caption("Pagu Anggaran Maksimal: **$20.00 / pelanggan**")

    # Match selected profile if persona chosen
    preset_profile = None
    if selected_persona != "Custom Input":
        selected_id = selected_persona.split(":")[0]
        matched = [p for p in personas_data if p["persona_id"] == selected_id]
        if matched:
            preset_profile = matched[0]["profile"]

    return st.session_state.custom_threshold, preset_profile, selected_persona
