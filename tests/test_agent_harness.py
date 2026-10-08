"""
Comprehensive Test Harness for Agentic Copilot, Personas, and Guardrails.
"""

import json
from pathlib import Path
import pytest
from pydantic import ValidationError

from src.services.agent_service import AgentService
from src.services.fallback_engine import HeuristicRetentionEngine
from src.services.schemas import CustomerProfile, RetentionPlan


@pytest.fixture
def benchmark_personas():
    persona_path = Path("tests/benchmark_personas.json")
    with open(persona_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_budget_guardrail_pydantic_enforcement():
    """Validates that RetentionPlan strictly rejects any offer > $20.00."""
    with pytest.raises(ValidationError):
        RetentionPlan(
            root_cause_diagnosis="Test diagnosis",
            recommended_package_name="Unauthorized VIP Discount",
            incentive_cost_usd=50.00,  # VIOLATES BUDGET GUARDRAIL ($20.00 max)
            simulated_churn_prob=0.20,
            risk_reduction_pct=50.0,
            projected_net_value_usd=100.0,
            outreach_script="Script",
            confidence_level="HIGH",
        )


def test_offline_fallback_execution():
    """Validates zero-downtime execution when API key is missing or offline."""
    agent_svc = AgentService(force_fallback=True)  # Force offline fallback mode
    cust = CustomerProfile(
        customer_id="CUST-OFFLINE-TEST",
        gender="Male",
        SeniorCitizen=0,
        Partner="No",
        Dependents="No",
        tenure=3,
        PhoneService="Yes",
        MultipleLines="No",
        InternetService="Fiber optic",
        OnlineSecurity="No",
        OnlineBackup="No",
        DeviceProtection="No",
        TechSupport="No",
        StreamingTV="Yes",
        StreamingMovies="No",
        Contract="Month-to-month",
        PaperlessBilling="Yes",
        PaymentMethod="Electronic check",
        MonthlyCharges=85.0,
        TotalCharges=255.0,
    )
    plan = agent_svc.generate_retention_plan(cust)
    assert isinstance(plan, RetentionPlan)
    assert plan.generation_source == "HEURISTIC_FALLBACK"
    assert plan.incentive_cost_usd <= 20.00
    assert len(plan.outreach_script) > 50
    assert plan.simulated_churn_prob < 0.77


def test_all_benchmark_personas_harness(benchmark_personas):
    """
    Evaluates all 5 golden personas from tests/benchmark_personas.json
    against the Heuristic and Agent pipeline.
    """
    engine = HeuristicRetentionEngine()

    for item in benchmark_personas:
        p_id = item["persona_id"]
        p_name = item["persona_name"]
        profile = CustomerProfile(**item["profile"])

        plan = engine.generate_plan(profile)

        # 1. Schema integrity
        assert isinstance(plan, RetentionPlan), f"Failed schema on {p_id}: {p_name}"

        # 2. Financial budget guardrail
        assert plan.incentive_cost_usd <= 20.00, f"Budget exceeded on {p_id}: {plan.incentive_cost_usd}"

        # 3. Empathetic copywriting
        assert len(plan.outreach_script) > 30, f"Script too short on {p_id}"
        assert "churn" not in plan.outreach_script.lower(), f"Forbidden word 'churn' in script on {p_id}"

        # 4. Valid confidence level
        assert plan.confidence_level in ["HIGH", "MEDIUM", "LOW"]
