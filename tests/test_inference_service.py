"""
Unit tests for InferenceService (LightGBM Pipeline Runner & Decision Logic).
"""

import pandas as pd
import pytest
from src.services.inference_service import InferenceService, get_inference_service
from src.services.schemas import CustomerProfile


@pytest.fixture
def inference_svc():
    return get_inference_service()


@pytest.fixture
def high_risk_profile():
    return CustomerProfile(
        customer_id="CUST-TEST-HIGH",
        gender="Female",
        SeniorCitizen=0,
        Partner="No",
        Dependents="No",
        tenure=2,
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
        MonthlyCharges=89.5,
        TotalCharges=179.0,
    )


@pytest.fixture
def low_risk_profile():
    return CustomerProfile(
        customer_id="CUST-TEST-LOW",
        gender="Male",
        SeniorCitizen=0,
        Partner="Yes",
        Dependents="Yes",
        tenure=65,
        PhoneService="Yes",
        MultipleLines="Yes",
        InternetService="DSL",
        OnlineSecurity="Yes",
        OnlineBackup="Yes",
        DeviceProtection="Yes",
        TechSupport="Yes",
        StreamingTV="Yes",
        StreamingMovies="Yes",
        Contract="Two year",
        PaperlessBilling="No",
        PaymentMethod="Credit card (automatic)",
        MonthlyCharges=65.0,
        TotalCharges=4225.0,
    )


def test_model_loaded(inference_svc: InferenceService):
    assert inference_svc.pipeline is not None
    assert inference_svc.optimal_threshold == 0.39
    assert "fe" in inference_svc.pipeline.named_steps
    assert "preprocessor" in inference_svc.pipeline.named_steps
    assert "classifier" in inference_svc.pipeline.named_steps


def test_predict_single_high_risk(inference_svc: InferenceService, high_risk_profile: CustomerProfile):
    res = inference_svc.predict_single(high_risk_profile)
    assert res.customer_id == "CUST-TEST-HIGH"
    assert res.churn_probability >= inference_svc.optimal_threshold
    assert res.decision == "INTERVENE"
    assert res.risk_level in ["HIGH_RISK", "MODERATE_RISK"]
    assert res.expected_loss > 0


def test_predict_single_low_risk(inference_svc: InferenceService, low_risk_profile: CustomerProfile):
    res = inference_svc.predict_single(low_risk_profile)
    assert res.customer_id == "CUST-TEST-LOW"
    assert res.churn_probability < inference_svc.optimal_threshold
    assert res.decision == "DO_NOT_INTERVENE"
    assert res.risk_level == "LOW_RISK"


def test_empty_string_total_charges_handled(inference_svc: InferenceService):
    data = {
        "customer_id": "CUST-NEW-ZERO",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 0,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 20.0,
        "TotalCharges": "   ",  # Blank string edge case
    }
    profile = CustomerProfile(**data)
    res = inference_svc.predict_single(profile)
    assert 0.0 <= res.churn_probability <= 1.0


def test_predict_batch(inference_svc: InferenceService, high_risk_profile: CustomerProfile, low_risk_profile: CustomerProfile):
    df = pd.DataFrame([high_risk_profile.model_dump(), low_risk_profile.model_dump()])
    res_df = inference_svc.predict_batch(df)
    assert len(res_df) == 2
    assert "churn_probability" in res_df.columns
    assert "decision" in res_df.columns
    assert "expected_loss" in res_df.columns
    # Prioritized: higher expected loss first
    assert res_df.iloc[0]["expected_loss"] >= res_df.iloc[1]["expected_loss"]
