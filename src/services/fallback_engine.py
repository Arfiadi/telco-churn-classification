"""
Offline Heuristic Retention Rule Engine.
Provides deterministic, zero-downtime fallback when external LLM APIs are unreachable or throttled.
"""

from typing import Optional
from src.services.agent_tools import (
    RETENTION_CATALOG,
    calculate_retention_roi,
    get_eligible_retention_offers,
    simulate_churn_impact,
)
from src.services.schemas import CustomerDiagnostic, CustomerProfile, RetentionPerk, RetentionPlan
from src.services.shap_service import get_shap_service


class HeuristicRetentionEngine:
    """
    Deterministic rule-based decision engine that matches customer SHAP risk factors
    to the optimal retention offer, computes simulated ML drop and ROI,
    and produces an empathetic outreach script.
    """

    def __init__(self):
        self.shap_service = get_shap_service()

    def generate_plan(
        self,
        customer: CustomerProfile,
        diagnostic: Optional[CustomerDiagnostic] = None,
    ) -> RetentionPlan:
        """
        Executes heuristic rule-matching and returns a validated RetentionPlan object.
        """
        if diagnostic is None:
            diagnostic = self.shap_service.explain_customer(customer)

        eligible_offers = get_eligible_retention_offers(customer)
        if not eligible_offers:
            # Fallback to contract migration if already contracted, or auto-pay
            eligible_offers = RETENTION_CATALOG

        # Determine the most pressing risk driver from SHAP
        risk_names = [r.feature_name.lower() for r in diagnostic.top_risk_drivers]
        risk_summary_text = " dan ".join([r.feature_name.split("__")[-1] for r in diagnostic.top_risk_drivers[:2]])

        selected_perk: RetentionPerk = eligible_offers[0]

        # Prioritize offer matching against top SHAP risk factors
        if any("contract" in r for r in risk_names) and customer.Contract == "Month-to-month":
            matched = [p for p in eligible_offers if p.perk_id == "PERK-CONTRACT-MIGRATE"]
            if matched:
                selected_perk = matched[0]
        elif (
            any("techsupport" in r or "security" in r or "fiber" in r for r in risk_names)
            and (customer.TechSupport == "No" or customer.OnlineSecurity == "No")
        ):
            matched = [p for p in eligible_offers if p.perk_id == "PERK-TECH-SEC-BUNDLE"]
            if matched:
                selected_perk = matched[0]
        elif any("paymentmethod" in r for r in risk_names) and customer.PaymentMethod == "Electronic check":
            matched = [p for p in eligible_offers if p.perk_id == "PERK-AUTOPAY-CASHBACK"]
            if matched:
                selected_perk = matched[0]
        elif customer.DeviceProtection == "No":
            matched = [p for p in eligible_offers if p.perk_id == "PERK-DEVICE-PROTECT"]
            if matched:
                selected_perk = matched[0]

        # Deterministic ML what-if simulation via local pipeline
        sim_res = simulate_churn_impact(customer, selected_perk.feature_patch)

        # Financial unit economics calculation
        roi_res = calculate_retention_roi(
            monthly_charges=customer.MonthlyCharges,
            current_prob=sim_res["baseline_churn_prob"],
            new_prob=sim_res["simulated_churn_prob"],
            incentive_cost=selected_perk.cost_usd,
        )

        # Root cause 2-sentence synthesis
        root_cause = (
            f"Pelanggan menunjukkan sensitivitas tinggi terhadap faktor {risk_summary_text} dengan probabilitas "
            f"churn baseline sebesar {sim_res['baseline_churn_prob'] * 100:.1f}%. "
            f"Ketiadaan komitmen jangka panjang atau layanan pendukung menjadi pemicu utama potensi penghentian layanan."
        )

        # Empathetic outreach copywriting tailored to selected package
        cust_id = customer.customer_id or "Pelanggan Setia"
        outreach_script = (
            f"Halo Bapak/Ibu {cust_id}, terima kasih atas kepercayaan Anda telah bersama kami selama "
            f"{customer.tenure} bulan. Sebagai bentuk apresiasi loyalitas Anda, kami ingin memberikan penawaran istimewa "
            f"eksklusif '{selected_perk.name}'. Melalui program ini, Anda berhak menikmati {selected_perk.description} "
            f"tanpa ada biaya tambahan di awal. Kami ingin memastikan pengalaman telekomunikasi Anda selalu nyaman dan bebas kendala. "
            f"Boleh kami bantu aktifkan paket apresiasi ini sekarang?"
        )

        confidence = "HIGH" if sim_res["relative_drop_pct"] >= 25.0 else ("MEDIUM" if sim_res["relative_drop_pct"] >= 10.0 else "LOW")

        return RetentionPlan(
            root_cause_diagnosis=root_cause,
            recommended_package_name=selected_perk.name,
            incentive_cost_usd=selected_perk.cost_usd,
            simulated_churn_prob=sim_res["simulated_churn_prob"],
            risk_reduction_pct=sim_res["relative_drop_pct"],
            projected_net_value_usd=roi_res["projected_net_value_usd"],
            outreach_script=outreach_script,
            confidence_level=confidence,
            generation_source="HEURISTIC_FALLBACK",
        )
