"""Streamlit App Automated UI & Interaction Tests.

Built using Streamlit's first-party in-process testing framework: `st.testing.v1.AppTest`.
Tests cover navigation, customer profiling, what-if counterfactual simulation,
AI retention copilot execution, batch processing work queue, and governance diagnostics.
"""

import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture
def app_runner():
    """Initializes and runs the main Streamlit application entrypoint."""
    at = AppTest.from_file("../app.py", default_timeout=25).run()
    assert not at.exception, f"App crashed on startup: {at.exception}"
    return at


def test_main_app_navigation_and_default_page(app_runner):
    """Test that main navigation loads and defaults to the Single Customer Profiler page."""
    at = app_runner
    assert not at.exception

    # Verify Sidebar components
    sidebar_subheaders = [s.value for s in at.sidebar.subheader]
    assert any("Status Sistem ML" in s for s in sidebar_subheaders)
    assert any("Quick Preset Persona" in s for s in sidebar_subheaders)

    # Verify Profiler subheader on default page
    subheaders = [s.value for s in at.subheader]
    assert any("Profil Risiko Pelanggan" in s for s in subheaders)


def test_profiler_persona_selection(app_runner):
    """Test switching customer personas from the sidebar and re-running inference."""
    at = app_runner
    persona_select = at.sidebar.selectbox(key="selected_persona_name")
    assert persona_select is not None

    # Select the first available benchmark persona if available
    if len(persona_select.options) > 1:
        target_persona = persona_select.options[1]
        persona_select.select(target_persona).run()
        assert not at.exception


def test_profiler_whatif_simulation(app_runner):
    """Test the counterfactual What-If sandbox fragment interaction."""
    at = app_runner

    # Interact with What-If contract simulation
    contract_sim = at.selectbox(key="whatif_contract")
    if contract_sim is not None and "Two year" in contract_sim.options:
        contract_sim.select("Two year").run()
        assert not at.exception


def test_profiler_copilot_generation(app_runner, monkeypatch):
    """Test clicking the Agentic AI Retention Copilot button with deterministic plan generation."""
    from src.services.agent_service import AgentService
    from src.services.schemas import RetentionPlan

    dummy_plan = RetentionPlan(
        root_cause_diagnosis="Kontrak bulanan berisiko tinggi tanpa proteksi online.",
        recommended_package_name="Annual Security Retention Bundle",
        incentive_cost_usd=15.0,
        simulated_churn_prob=0.25,
        risk_reduction_pct=55.0,
        projected_net_value_usd=120.0,
        outreach_script="Halo, kami menghargai loyalitas Anda...",
        confidence_level="HIGH",
    )
    monkeypatch.setattr(AgentService, "generate_retention_plan", lambda self, p, d: dummy_plan)

    at = app_runner
    gen_btn = at.button(key="btn_gen_plan")
    assert gen_btn is not None
    gen_btn.click().run()
    assert not at.exception


def test_batch_queue_page_sample_processing(app_runner):
    """Test navigating to Page 2 (Batch Queue) and running sample batch inference."""
    at = app_runner
    at.switch_page("pages/2_batch_queue.py").run()
    assert not at.exception

    # Subheader check
    subheaders = [s.value for s in at.subheader]
    assert any("Batch Churn Analysis" in s for s in subheaders)

    # Click sample loading button
    btn_sample = at.button(key="btn_load_sample")
    assert btn_sample is not None
    btn_sample.click().run()
    assert not at.exception

    # Check dataframe rendered
    assert len(at.dataframe) > 0

    # Test filtering
    filter_box = at.selectbox(key="batch_risk_filter")
    if filter_box is not None:
        filter_box.select("HIGH_RISK").run()
        assert not at.exception


def test_governance_page_rendering(app_runner):
    """Test navigating to Page 3 (Executive Governance) and verifying diagnostics."""
    at = app_runner
    at.switch_page("pages/3_governance.py").run()
    assert not at.exception

    subheaders = [s.value for s in at.subheader]
    assert any("Executive Governance" in s for s in subheaders)
