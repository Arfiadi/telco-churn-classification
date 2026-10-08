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

    def _generate_business_explanation(
        self,
        feat_name: str,
        feat_val: Any,
        shap_val: float,
        profile: Optional[CustomerProfile] = None,
    ) -> str:
        """
        Translates raw feature encoding names into human-readable business context.
        Uses comprehensive mapping for all pipeline-encoded feature names.
        """
        is_risk = shap_val > 0
        name_lower = feat_name.lower()
        name_suffix = feat_name.split("__")[-1].strip()

        # --- Contract features ---
        if "contract_month-to-month" in name_lower:
            if is_risk:
                return "Kontrak bulanan (Month-to-month) tanpa komitmen jangka panjang mempermudah perpindahan ke kompetitor."
            return "Ketiadaan kontrak bulanan (memiliki komitmen kontrak jangka panjang) memperkuat stabilitas retensi."
        if "contract_two year" in name_lower:
            if is_risk:
                return "Masa kontrak 2 tahun mendekati akhir evaluasi layanan oleh pelanggan."
            return "Komitmen kontrak 2 tahun menjadi jangkar protektif yang sangat kuat menahan churn."
        if "contract_one year" in name_lower:
            if is_risk:
                return "Mendekati akhir siklus kontrak 1 tahun meningkatkan potensi pertimbangan provider lain."
            return "Komitmen kontrak 1 tahun memberikan hambatan berpindah yang moderat bagi pelanggan."

        # --- Tenure (recover real month value) ---
        if "tenure" in name_lower:
            real_months = profile.tenure if profile is not None else None
            if real_months is None:
                try:
                    real_months = int(round(float(feat_val)))
                    if real_months < 0:
                        real_months = abs(real_months)
                except (ValueError, TypeError):
                    real_months = feat_val
            if is_risk:
                return f"Masa langganan baru ({real_months} bulan) menandakan loyalitas merek belum terbentuk kuat."
            return f"Masa langganan lama ({real_months} bulan) menjadi pilar loyalitas yang memperkuat retensi."

        # --- Tech Support ---
        if "techsupport_no" in name_lower:
            return "Tidak berlangganan layanan Tech Support meningkatkan friksi saat pelanggan mengalami kendala teknis."
        if "techsupport_yes" in name_lower:
            return "Layanan Tech Support aktif memberikan rasa aman dan mengurangi alasan pelanggan untuk berpindah."

        # --- Online Security ---
        if "onlinesecurity_no" in name_lower:
            return "Tanpa perlindungan Online Security, pelanggan memiliki hambatan berpindah (switching barrier) yang rendah."
        if "onlinesecurity_yes" in name_lower:
            return "Perlindungan Online Security aktif menciptakan ketergantungan layanan yang memperkuat retensi."

        # --- Internet Service ---
        if "internetservice_fiber optic" in name_lower or "internetservice_fiber" in name_lower:
            return "Layanan Fiber Optic memiliki ekspektasi performa tinggi dan biaya premium yang sensitif terhadap kepuasan."
        if "internetservice_dsl" in name_lower:
            return "Layanan DSL dengan biaya lebih rendah cenderung memiliki tingkat loyalitas pelanggan yang lebih stabil."
        if "internetservice_no" in name_lower:
            return "Pelanggan tanpa layanan internet hanya menggunakan layanan telepon dasar."

        # --- Payment Method ---
        if "paymentmethod_electronic check" in name_lower or "electronic check" in name_lower:
            return "Pembayaran via Electronic Check bersifat manual tiap bulan dan memiliki korelasi churn tertinggi."
        if "paymentmethod" in name_lower and "automatic" in name_lower:
            return "Metode pembayaran otomatis (Bank/Kartu Kredit) menciptakan kebiasaan pasif yang menekan churn."
        if "paymentmethod_mailed check" in name_lower:
            return "Pembayaran via cek pos menunjukkan preferensi tradisional dengan tingkat churn sedang."

        # --- Multiple Lines ---
        if "multiplelines_no" in name_lower:
            return "Pelanggan hanya menggunakan satu saluran telepon, menandakan utilisasi layanan yang minimal."
        if "multiplelines_yes" in name_lower:
            return "Penggunaan beberapa saluran telepon menunjukkan ketergantungan lebih tinggi pada layanan provider."
        if "multiplelines_no phone" in name_lower:
            return "Pelanggan tidak memiliki layanan telepon sama sekali."

        # --- Paperless Billing ---
        if "paperlessbilling_yes" in name_lower:
            return "Tagihan digital (paperless) meningkatkan kesadaran pelanggan terhadap fluktuasi biaya setiap bulan."
        if "paperlessbilling_no" in name_lower:
            return "Tagihan fisik (kertas) cenderung mengurangi perhatian aktif pelanggan terhadap perubahan biaya."

        # --- Online Backup ---
        if "onlinebackup_no" in name_lower:
            return "Tanpa layanan Online Backup, pelanggan tidak memiliki data terikat yang menghambat perpindahan."
        if "onlinebackup_yes" in name_lower:
            return "Layanan Online Backup aktif menciptakan ketergantungan data yang memperkuat retensi."

        # --- Device Protection ---
        if "deviceprotection_no" in name_lower:
            return "Tidak berlangganan proteksi perangkat mengurangi nilai tambah yang dirasakan pelanggan."
        if "deviceprotection_yes" in name_lower:
            return "Proteksi perangkat aktif memberikan rasa aman dan menambah alasan pelanggan untuk tetap berlangganan."

        # --- Streaming TV / Movies ---
        if "streamingtv_yes" in name_lower:
            return "Layanan Streaming TV aktif menambah ekosistem hiburan yang meningkatkan switching barrier."
        if "streamingtv_no" in name_lower:
            return "Tanpa layanan Streaming TV, pelanggan memiliki sedikit insentif non-konektivitas untuk bertahan."
        if "streamingmovies_yes" in name_lower:
            return "Layanan Streaming Movies aktif memperkaya bundling layanan yang menahan pelanggan."
        if "streamingmovies_no" in name_lower:
            return "Tanpa Streaming Movies, variasi layanan pelanggan terbatas sehingga loyalitas rendah."

        # --- Senior Citizen ---
        if "seniorcitizen_1" in name_lower or "seniorcitizen" in name_lower and str(feat_val) == "1":
            direction = "lebih sensitif terhadap biaya dan cenderung churn" if is_risk else "yang loyal cenderung bertahan"
            return f"Pelanggan warga senior (≥65 tahun) {direction}."
        if "seniorcitizen_0" in name_lower:
            direction = "meningkatkan risiko churn" if is_risk else "memperkuat retensi pelanggan"
            return f"Status bukan warga senior secara statistik {direction}."

        # --- Demographics ---
        if "partner_yes" in name_lower:
            return "Memiliki pasangan cenderung meningkatkan stabilitas langganan rumah tangga."
        if "partner_no" in name_lower:
            return "Pelanggan tanpa pasangan memiliki fleksibilitas lebih tinggi untuk berpindah provider."
        if "dependents_yes" in name_lower:
            return "Memiliki tanggungan keluarga menciptakan kebutuhan layanan yang stabil dan mengurangi churn."
        if "dependents_no" in name_lower:
            return "Tanpa tanggungan, pelanggan lebih leluasa mengambil keputusan berpindah."

        # --- Monthly/Total Charges ---
        if "monthlychargesratio" in name_lower:
            direction = "menambah beban biaya relatif bulanan" if is_risk else "menunjukkan proporsi biaya bulanan yang stabil"
            return f"Rasio tagihan bulanan terhadap total pengeluaran {direction} bagi pelanggan."
        if "monthlycharges" in name_lower:
            m_val = f"{profile.MonthlyCharges:.2f}" if profile is not None else str(feat_val)
            if is_risk:
                return f"Tagihan bulanan (${m_val}/bln) dirasa berat tanpa bundling keuntungan yang sepadan."
            return f"Tagihan bulanan (${m_val}/bln) dinilai terjangkau dan seimbang oleh pelanggan."
        if "totalcharges" in name_lower:
            t_val = f"{profile.TotalCharges:,.2f}" if profile is not None else str(feat_val)
            if is_risk:
                return f"Total pengeluaran kumulatif (${t_val}) relatif rendah, menandakan masa langganan singkat."
            return f"Total pengeluaran kumulatif (${t_val}) mencerminkan hubungan pelanggan jangka panjang."

        # --- Total Services ---
        if "totalservices" in name_lower:
            if profile is not None:
                srv_fields = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
                s_val = sum(1 for f in srv_fields if getattr(profile, f, "No") in ["Yes"] or (f == "MultipleLines" and getattr(profile, f, "") == "Yes"))
            else:
                s_val = feat_val
            if is_risk:
                return f"Sedikitnya layanan aktif ({s_val} layanan) membuat switching barrier sangat rendah."
            return f"Banyaknya layanan terintegrasi ({s_val} layanan) menciptakan ekosistem penggunaan yang lengket."

        # --- Phone Service ---
        if "phoneservice_yes" in name_lower:
            return "Layanan telepon aktif menambah satu dimensi konektivitas yang mengikat pelanggan."
        if "phoneservice_no" in name_lower:
            return "Tanpa layanan telepon, pelanggan hanya bergantung pada internet saja."

        # --- Gender ---
        if "gender_male" in name_lower or "gender_female" in name_lower:
            direction = "meningkatkan risiko churn" if is_risk else "memperkuat retensi pelanggan"
            return f"Faktor demografis jenis kelamin secara statistik {direction} pada segmen ini."

        # --- Fallback: humanized generic ---
        direction = "meningkatkan risiko churn" if is_risk else "memperkuat retensi pelanggan"
        clean_name = name_suffix.replace("_", " ")
        return f"Faktor '{clean_name}' secara signifikan {direction}."

    def _get_human_feature_value(self, feat_name: str, val_in_matrix: float, profile: CustomerProfile) -> str:
        """Returns clean human-readable real feature value, avoiding standardized z-score leakage."""
        name_lower = feat_name.lower()
        if "tenure" in name_lower:
            return f"{profile.tenure} bulan"
        if "monthlychargesratio" in name_lower:
            return f"{float(np.round(val_in_matrix, 2)):.2f}"
        if "monthlycharges" in name_lower:
            return f"${profile.MonthlyCharges:.2f}/bln"
        if "totalcharges" in name_lower:
            return f"${profile.TotalCharges:,.2f}"
        if "totalservices" in name_lower:
            srv_fields = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
            cnt = sum(1 for f in srv_fields if getattr(profile, f, "No") in ["Yes"] or (f == "MultipleLines" and getattr(profile, f, "") == "Yes"))
            return f"{cnt} layanan"

        # Check categorical matches from profile attributes
        for attr, val in profile.model_dump().items():
            if attr.lower() in name_lower:
                return str(val)

        return str(np.round(val_in_matrix, 2))

    def explain_customer(
        self,
        customer: Union[CustomerProfile, dict[str, Any]],
        top_k: int = 3,
    ) -> CustomerDiagnostic:
        """
        Computes TreeSHAP attributions and returns top risk drivers and retention anchors.
        """
        # Ensure profile schema validation
        if hasattr(customer, "model_dump"):
            profile_data = customer.model_dump()
        elif isinstance(customer, dict):
            profile_data = customer
        elif isinstance(customer, CustomerProfile):
            profile_data = customer.model_dump()
        else:
            profile_data = dict(customer)
        profile = CustomerProfile(**profile_data)

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
            human_val = self._get_human_feature_value(name, val_in_matrix, profile)

            feature_contributions.append({
                "feature_name": name,
                "feature_value": human_val,
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
                    item["feature_name"], item["feature_value"], item["shap_value"], profile=profile
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
                    item["feature_name"], item["feature_value"], item["shap_value"], profile=profile
                ),
            )
            for item in top_anchors
        ]

        return CustomerDiagnostic(
            customer=profile.model_dump(),
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
