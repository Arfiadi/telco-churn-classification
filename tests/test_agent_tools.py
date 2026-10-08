"""
Unit tests for deterministic Agent Tools (Simulation, ROI, Perk Catalog).
"""

import pytest
from src.services.agent_tools import (
    calculate_retention_roi,
    get_eligible_retention_offers,
    simulate_churn_impact,
)
from src.services.schemas import CustomerProfile


@pytest.fixture
def profile():
    return CustomerProfile(
        customer_id="CUST-TOOL-TEST",
        gender="Male",
        SeniorCitizen=0,
        Partner="No",
        Dependents="No",
        tenure=5,
        PhoneService="Yes",
        MultipleLines="No",
        InternetService="Fiber optic",
        OnlineSecurity="No",
        OnlineBackup="No",
        DeviceProtection="No",
        TechSupport="No",
        StreamingTV="No",
        StreamingMovies="No",
        Contract="Month-to-month",
        PaperlessBilling="Yes",
        PaymentMethod="Electronic check",
        MonthlyCharges=75.0,
        TotalCharges=375.0,
    )


def test_simulate_churn_impact_reduces_risk(profile: CustomerProfile):
    # Changing Month-to-month to One year should decrease churn probability
    sim = simulate_churn_impact(profile, {"Contract": "One year"})
    assert sim["baseline_churn_prob"] > sim["simulated_churn_prob"]
    assert sim["absolute_drop"] > 0
    assert sim["relative_drop_pct"] > 0


def test_calculate_retention_roi_profitability():
    # Baseline prob = 0.70, New prob = 0.35, Monthly charges = $80, Cost = $15
    res = calculate_retention_roi(
        monthly_charges=80.0,
        current_prob=0.70,
        new_prob=0.35,
        incentive_cost=15.0,
    )
    # CLV = 80 * 12 = 960. Delta P = 0.35. Saved = 336. Net profit = 336 - 15 = 321
    assert res["annual_clv"] == 960.0
    assert res["saved_clv"] == 336.0
    assert res["projected_net_value_usd"] == 321.0
    assert res["is_profitable"] is True


def test_calculate_retention_roi_budget_guardrail_exception():
    # Attempting incentive cost > $20.00 must raise ValueError
    with pytest.raises(ValueError, match="exceeds company budget guardrail"):
        calculate_retention_roi(
            monthly_charges=80.0,
            current_prob=0.70,
            new_prob=0.35,
            incentive_cost=25.0,  # Exceeds $20
        )


def test_get_eligible_retention_offers(profile: CustomerProfile):
    offers = get_eligible_retention_offers(profile)
    assert len(offers) > 0
    for offer in offers:
        assert offer.cost_usd <= 20.00
