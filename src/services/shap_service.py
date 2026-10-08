"""
Explainability (XAI) Service for Telco Churn Platform.
Calculates local feature attributions via TreeSHAP and generates business-readable insights.
"""

from typing import Any, Optional, Union
import numpy as np
import pandas as pd
import shap

from src.services.inference_service import InferenceService, get_inference_service
from src.services.schemas import CustomerDiagnostic, CustomerProfile, SHAPFactor


class SHAPService:
    """
    Manages TreeExplainer caching and local attribution analysis for LightGBM.
    """

    def __init__(self, inference_service: Optional[InferenceService] = None):
        self.inference_service = inference_service or get_inference_service()
        self.pipeline = self.inference_service.pipeline
        self.fe_step = self.pipeline.named_steps["fe"]
        self.preprocessor_step = self.pipeline.named_steps["preprocessor"]
        self.classifier_step = self.pipeline.named_steps["classifier"]

        # Cache TreeExplainer on classifier step
        self.explainer = shap.TreeExplainer(self.classifier_step)
        self.feature_names = list(self.preprocessor_step.get_feature_names_out())

    def _transform_customer_features(self, df_clean: pd.DataFrame) -> np.ndarray:
        """Runs the feature engineer and preprocessor transforms."""
        X_fe = self.fe_step.transform(df_clean)
        X_trans = self.preprocessor_step.transform(X_fe)
        return X_trans

    def _generate_business_explanation(self, feat_name: str, feat_val: Any, shap_val: float) -> str:
        """
        Translates raw feature encoding names into human-readable business context.
        """
        is_risk = shap_val > 0
        name_lower = feat_name.lower()

        if "contract_month-to-month" in name_lower or "contract" in name_lower and "month" in str(feat_val).lower():
            return "Kontrak bulanan (Month-to-month) tanpa komitmen jangka panjang mempermudah perpindahan ke kompetitor."
        elif "contract_two year" in name_lower or "contract_one year" in name_lower:
            return "Komitmen kontrak jangka panjang menjadi jangkar protektif yang menahan churn."
        elif "techsupport_no" in name_lower:
            return "Ketiadaan layanan prioritas Tech Support meningkatkan friksi pelanggan saat terjadi kendala teknis."
        elif "onlinesecurity_no" in name_lower:
            return "Tidak berlangganan Online Security menurunkan hambatan switching ke provider lain."
        elif "internetservice_fiber optic" in name_lower:
            return "Layanan Fiber Optic memiliki ekspektasi performa tinggi dengan biaya premium yang sensitif terhadap kepuasan."
        elif "paymentmethod_electronic check" in name_lower:
            return "Metode Electronic Check memerlukan tindakan pembayaran manual tiap bulan dan memiliki churn rate tertinggi."
        elif "paymentmethod" in name_lower and "automatic" in name_lower:
            return "Metode pembayaran otomatis (Bank/Credit Card) menciptakan kebiasaan bayar pasif yang menekan churn."
        elif "tenure" in name_lower:
            if is_risk:
                return f"Masa langganan relatif baru ({feat_val} bulan), loyalitas merek belum terbentuk kuat."
            return f"Masa langganan lama ({feat_val} bulan) menjadi pilar loyalitas yang memperkuat retensi."
        elif "totalservices" in name_lower:
            if is_risk:
                return f"Sedikitnya add-on layanan aktif ({feat_val} layanan) membuat switching barrier sangat rendah."
            return f"Tingginya variasi layanan terintegrasi ({feat_val} layanan) menciptakan ekosistem penggunaan yang lengket."
        elif "monthlycharges" in name_lower:
            if is_risk:
                return f"Besaran tagihan (${feat_val}/bln) dirasa berat tanpa bundling keuntungan yang sepadan."
            return f"Tingkat tagihan (${feat_val}/bln) dinilai terjangkau dan seimbang oleh pelanggan."
        elif "paperlessbilling_yes" in name_lower:
            return "Penerimaan tagihan digital secara berkala meningkatkan kesadaran pelanggan terhadap fluktuasi biaya."
        else:
            direction = "meningkatkan risiko churn" if is_risk else "memperkuat retensi pelanggan"
            return f"Fitur {feat_name} bernilai '{feat_val}' secara signifikan {direction}."

    def explain_customer(
        self,
        customer: Union[CustomerProfile, dict[str, Any]],
        top_k: int = 3,
    ) -> CustomerDiagnostic:
        """
        Computes TreeSHAP attributions and returns top risk drivers and retention anchors.
        """
        # Ensure profile schema validation
        if isinstance(customer, dict):
            profile = CustomerProfile(**customer)
        elif type(customer).__name__ == "CustomerProfile" or hasattr(customer, "model_dump") or hasattr(customer, "dict"):
            profile = customer
        else:
            try:
                profile = CustomerProfile(**dict(customer))
            except Exception:
                profile = customer

        df_clean = self.inference_service._prepare_dataframe(profile)
        X_trans = self._transform_customer_features(df_clean)

        # Baseline prediction
        inf_result = self.inference_service.predict_single(profile)

        # Compute SHAP values
        shap_res = self.explainer(X_trans)
        vals = shap_res.values[0]
        # In binary classification, LightGBM TreeExplainer output may have shape (53, 2)
        if vals.ndim == 2:
            churn_shap = vals[:, 1]
        else:
            churn_shap = vals

        # Combine feature names, values, and shap scores
        feature_contributions = []
        for i, name in enumerate(self.feature_names):
            val_in_matrix = X_trans[0, i] if hasattr(X_trans, "ndim") else X_trans[i]
            s_val = float(churn_shap[i])

            feature_contributions.append({
                "feature_name": name,
                "feature_value": str(np.round(val_in_matrix, 2)),
                "shap_value": s_val,
            })

        # Sort for Risk Drivers (highest positive SHAP)
        sorted_risk = sorted(feature_contributions, key=lambda x: x["shap_value"], reverse=True)
        top_risks = [f for f in sorted_risk if f["shap_value"] > 0][:top_k]

        # Sort for Retention Anchors (lowest negative SHAP)
        sorted_anchors = sorted(feature_contributions, key=lambda x: x["shap_value"])
        top_anchors = [f for f in sorted_anchors if f["shap_value"] < 0][:top_k]

        risk_factors: list[SHAPFactor] = [
            SHAPFactor(
                feature_name=item["feature_name"],
                feature_value=item["feature_value"],
                shap_value=float(np.round(item["shap_value"], 4)),
                impact_type="RISK_DRIVER",
                human_explanation=self._generate_business_explanation(
                    item["feature_name"], item["feature_value"], item["shap_value"]
                ),
            )
            for item in top_risks
        ]

        anchor_factors: list[SHAPFactor] = [
            SHAPFactor(
                feature_name=item["feature_name"],
                feature_value=item["feature_value"],
                shap_value=float(np.round(item["shap_value"], 4)),
                impact_type="RETENTION_ANCHOR",
                human_explanation=self._generate_business_explanation(
                    item["feature_name"], item["feature_value"], item["shap_value"]
                ),
            )
            for item in top_anchors
        ]

        return CustomerDiagnostic(
            customer=profile,
            baseline_churn_prob=inf_result.churn_probability,
            decision_threshold=inf_result.optimal_threshold,
            is_at_risk=(inf_result.decision == "INTERVENE"),
            top_risk_drivers=risk_factors,
            top_retention_anchors=anchor_factors,
        )


_shap_service: Optional[SHAPService] = None


def get_shap_service() -> SHAPService:
    global _shap_service
    if _shap_service is None:
        _shap_service = SHAPService()
    return _shap_service
