"""
Unit tests for SHAPService (TreeExplainer and Business Explanations).
"""

import pytest
from src.services.schemas import CustomerProfile
from src.services.shap_service import SHAPService, get_shap_service


@pytest.fixture
def shap_svc():
    return get_shap_service()


@pytest.fixture
def sample_customer():
    return CustomerProfile(
        customer_id="CUST-SHAP-TEST",
        gender="Female",
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
        StreamingMovies="Yes",
        Contract="Month-to-month",
        PaperlessBilling="Yes",
        PaymentMethod="Electronic check",
        MonthlyCharges=90.0,
        TotalCharges=270.0,
    )


def test_shap_service_initialization(shap_svc: SHAPService):
    assert shap_svc.explainer is not None
    assert len(shap_svc.feature_names) == 53


def test_explain_customer_returns_diagnostic(shap_svc: SHAPService, sample_customer: CustomerProfile):
    diagnostic = shap_svc.explain_customer(sample_customer, top_k=3)
    assert diagnostic.customer.customer_id == "CUST-SHAP-TEST"
    assert diagnostic.is_at_risk is True
    assert len(diagnostic.top_risk_drivers) <= 3
    assert len(diagnostic.top_retention_anchors) <= 3

    # All risk drivers have positive SHAP
    for rd in diagnostic.top_risk_drivers:
        assert rd.shap_value > 0
        assert rd.impact_type == "RISK_DRIVER"
        assert len(rd.human_explanation) > 10

    # All anchors have negative SHAP
    for anchor in diagnostic.top_retention_anchors:
        assert anchor.shap_value < 0
        assert anchor.impact_type == "RETENTION_ANCHOR"
        assert len(anchor.human_explanation) > 10
