"""
Deterministic Tools for Retention Agent & Heuristic Engine.
Provides what-if counterfactual churn simulation, unit economics ROI calculations,
and pre-approved retention package catalog queries.
"""

from typing import Any, Optional
import numpy as np

from src.services.inference_service import get_inference_service
from src.services.schemas import CustomerProfile, RetentionPerk

# Pre-approved Corporate Retention Catalog ($C <= $20.00)
RETENTION_CATALOG: list[RetentionPerk] = [
    RetentionPerk(
        perk_id="PERK-CONTRACT-MIGRATE",
        name="Contract Migration Shield",
        description="Diskon tagihan $10/bulan selama 2 bulan (Total: $20) untuk komitmen kontrak 1 tahun.",
        cost_usd=20.00,
        feature_patch={"Contract": "One year"},
        target_drivers=["Contract_Month-to-month", "Contract", "tenure"],
    ),
    RetentionPerk(
        perk_id="PERK-TECH-SEC-BUNDLE",
        name="Tech Support & Security Bundle",
        description="Gratis fasilitas Tech Support dan Online Security selama 3 bulan (Biaya: $15).",
        cost_usd=15.00,
        feature_patch={"TechSupport": "Yes", "OnlineSecurity": "Yes"},
        target_drivers=["TechSupport_No", "OnlineSecurity_No", "InternetService_Fiber optic"],
    ),
    RetentionPerk(
        perk_id="PERK-AUTOPAY-CASHBACK",
        name="Auto-Pay Loyalty Cashback",
        description="Cashback kredit tagihan $10 untuk migrasi dari tagihan manual ke Bank Transfer otomatis.",
        cost_usd=10.00,
        feature_patch={"PaymentMethod": "Bank transfer (automatic)"},
        target_drivers=["PaymentMethod_Electronic check", "PaymentMethod"],
    ),
    RetentionPerk(
        perk_id="PERK-DEVICE-PROTECT",
        name="Device Protection Shield",
        description="Perlindungan ganti rugi perangkat keras gratis selama 2 bulan (Biaya: $10).",
        cost_usd=10.00,
        feature_patch={"DeviceProtection": "Yes"},
        target_drivers=["DeviceProtection_No", "DeviceProtection"],
    ),
    RetentionPerk(
        perk_id="PERK-LOYALTY-VIP",
        name="VIP Loyalty Appreciation",
        description="Apresiasi pelanggan setia: Kupon streaming gratis dan prioritas antrean CS (Biaya: $0).",
        cost_usd=0.00,
        feature_patch={},
        target_drivers=["Contract_Two year", "Contract_One year", "tenure"],
    ),
]


def get_eligible_retention_offers(
    customer_profile: CustomerProfile,
    max_budget_usd: float = 20.00,
) -> list[RetentionPerk]:
    """
    Returns pre-approved retention offers whose cost is within company budget constraints
    and relevant to the customer's active features.
    """
    eligible = []
    cust_dict = customer_profile.model_dump()
    for perk in RETENTION_CATALOG:
        if perk.cost_usd > max_budget_usd:
            continue

        # Prevent contract downgrade: If customer already has 'One year' or 'Two year',
        # contract migration to 'One year' is not an eligible retention offer.
        if perk.perk_id == "PERK-CONTRACT-MIGRATE" and cust_dict.get("Contract") != "Month-to-month":
            continue

        # For perks with empty feature patches (e.g. VIP loyalty appreciation),
        # only offer to contracted or long-tenure customers
        if not perk.feature_patch:
            if cust_dict.get("Contract") in ["One year", "Two year"] or int(cust_dict.get("tenure", 0)) >= 12:
                eligible.append(perk)
            continue

        # Check if perk applies a real modification to the customer
        is_relevant = False
        for k, v in perk.feature_patch.items():
            if cust_dict.get(k) != v:
                is_relevant = True
                break

        if is_relevant:
            eligible.append(perk)

    return eligible


def simulate_churn_impact(
    customer_profile: CustomerProfile,
    feature_modifications: dict[str, Any],
) -> dict[str, float]:
    """
    Simulates a counterfactual what-if scenario by applying feature modifications
    to the customer profile and scoring through the LightGBM pipeline.

    Returns:
        baseline_churn_prob: Float original churn probability.
        new_churn_prob: Float simulated churn probability.
        absolute_drop: Absolute probability reduction.
        relative_drop_pct: Relative percentage drop ((P_old - P_new) / P_old) * 100%.
    """
    inference_svc = get_inference_service()

    # Score baseline
    base_res = inference_svc.predict_single(customer_profile)
    p_baseline = base_res.churn_probability

    # Create modified profile
    modified_data = customer_profile.model_dump()
    for key, value in feature_modifications.items():
        if key in modified_data:
            modified_data[key] = value

    modified_profile = CustomerProfile(**modified_data)
    new_res = inference_svc.predict_single(modified_profile)
    p_new = new_res.churn_probability

    abs_drop = float(np.round(p_baseline - p_new, 4))
    rel_drop_pct = float(np.round((abs_drop / max(p_baseline, 1e-6)) * 100.0, 2))

    return {
        "baseline_churn_prob": p_baseline,
        "simulated_churn_prob": p_new,
        "absolute_drop": abs_drop,
        "relative_drop_pct": rel_drop_pct,
    }


def calculate_retention_roi(
    monthly_charges: float,
    current_prob: float,
    new_prob: float,
    incentive_cost: float,
) -> dict[str, float]:
    """
    Calculates unit economic financial impact of a proposed retention intervention:
    Expected Net Profit = (Delta P(churn) * CLV) - Incentive Cost
    where CLV = MonthlyCharges * 12 months.
    """
    if incentive_cost > 20.00:
        raise ValueError(f"Incentive cost ${incentive_cost:.2f} exceeds company budget guardrail ($20.00).")

    annual_clv = float(monthly_charges * 12.0)
    delta_prob = float(max(current_prob - new_prob, 0.0))
    saved_clv = float(np.round(delta_prob * annual_clv, 2))
    net_profit = float(np.round(saved_clv - incentive_cost, 2))
    is_profitable = net_profit > 0.0
    roi_pct = float(np.round((net_profit / max(incentive_cost, 1.0)) * 100.0, 2))

    return {
        "annual_clv": annual_clv,
        "saved_clv": saved_clv,
        "incentive_cost": incentive_cost,
        "projected_net_value_usd": net_profit,
        "is_profitable": is_profitable,
        "roi_pct": roi_pct,
    }
