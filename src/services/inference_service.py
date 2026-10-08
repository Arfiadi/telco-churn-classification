"""
Inference Service for Telco Churn Machine Learning Pipeline.
Handles model loading, validation, real-time prediction, and business thresholding.
"""

from pathlib import Path
from typing import Any, Optional, Union
import joblib
import numpy as np
import pandas as pd

from src.services.schemas import CustomerProfile, InferenceResult
import src.transformers  # Ensure custom transformer is registered for unpickling


class InferenceService:
    """
    Manages loading and executing the LightGBM Telco Churn Pipeline.
    Thread-safe and cached execution on CPU.
    """

    def __init__(self, model_path: Union[str, Path] = "models/telco_churn_lgbm_pipeline.joblib"):
        self.model_path = Path(model_path)
        self.pipeline = None
        self.optimal_threshold = 0.39
        self.metadata = {}
        self._load_model()

    def _load_model(self) -> None:
        """Loads serialized model and extracts pipeline and threshold metadata."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {self.model_path.resolve()}")

        loaded = joblib.load(self.model_path)
        if isinstance(loaded, dict):
            self.pipeline = loaded["pipeline"]
            self.optimal_threshold = float(loaded.get("optimal_threshold", 0.39))
            self.metadata = loaded.get("metadata", {})
        else:
            self.pipeline = loaded
            self.optimal_threshold = 0.39
            self.metadata = {"model_name": "Standard Scikit-learn Pipeline"}

    def _prepare_dataframe(self, data: Any) -> pd.DataFrame:
        """
        Converts customer input into standard cleaned DataFrame for pipeline ingestion.
        Guaranteed to handle CustomerProfile instances, reloaded Pydantic models, dicts,
        DataFrames, Series, or any custom object.
        """
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, pd.Series):
            df = pd.DataFrame([data.to_dict()])
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        elif hasattr(data, "model_dump"):
            try:
                dump = data.model_dump()
                df = pd.DataFrame([dump])
            except Exception:
                try:
                    df = pd.DataFrame([data.__dict__])
                except Exception:
                    df = pd.DataFrame([dict(data)])
        elif hasattr(data, "dict") and callable(getattr(data, "dict")):
            df = pd.DataFrame([data.dict()])
        elif type(data).__name__ == "CustomerProfile" or hasattr(data, "__dict__"):
            if hasattr(data, "model_fields"):
                d = {k: getattr(data, k) for k in data.model_fields.keys() if hasattr(data, k)}
                df = pd.DataFrame([d])
            else:
                clean_dict = {k: v for k, v in data.__dict__.items() if not k.startswith("_")}
                df = pd.DataFrame([clean_dict])
        else:
            try:
                df = pd.DataFrame([dict(data)])
            except Exception:
                raise ValueError(f"Unsupported data type for inference: {type(data)}")

        # Exclude non-feature identification columns if present
        cols_to_drop = [c for c in ["customer_id", "customerID", "Churn", "churn"] if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        # Sanitize TotalCharges (handle whitespace strings from tenure=0)
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            # For 0 tenure or missing TotalCharges, default to 0.0 or MonthlyCharges
            df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

        # Ensure numeric columns are cast
        for col in ["tenure", "SeniorCitizen"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        if "MonthlyCharges" in df.columns:
            df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0.0)

        return df

    def predict_single(
        self,
        customer: Union[CustomerProfile, dict[str, Any]],
        custom_threshold: Optional[float] = None,
    ) -> InferenceResult:
        """
        Performs single customer churn inference with profit threshold evaluation.
        """
        customer_id = getattr(customer, "customer_id", None) or (
            customer.get("customer_id") if isinstance(customer, dict) else "CUST-UNKNOWN"
        )
        monthly_charges = float(
            getattr(customer, "MonthlyCharges", None)
            or (customer.get("MonthlyCharges") if isinstance(customer, dict) else 0.0)
        )

        df = self._prepare_dataframe(customer)
        proba_matrix = self.pipeline.predict_proba(df)
        churn_prob = float(proba_matrix[0, 1])

        active_threshold = custom_threshold if custom_threshold is not None else self.optimal_threshold
        decision = "INTERVENE" if churn_prob >= active_threshold else "DO_NOT_INTERVENE"

        # Annualized Customer Lifetime Value estimation (12 months of charges)
        estimated_clv = monthly_charges * 12.0
        expected_loss = float(np.round(churn_prob * estimated_clv, 2))

        # Risk level tiers
        if churn_prob >= 0.60:
            risk_level = "HIGH_RISK"
        elif churn_prob >= active_threshold:
            risk_level = "MODERATE_RISK"
        else:
            risk_level = "LOW_RISK"

        return InferenceResult(
            customer_id=customer_id,
            churn_probability=float(np.round(churn_prob, 4)),
            optimal_threshold=float(np.round(active_threshold, 4)),
            decision=decision,
            expected_loss=expected_loss,
            risk_level=risk_level,
        )

    def predict_batch(
        self,
        df: pd.DataFrame,
        custom_threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Runs vectorized batch predictions and sorts customers by priority (Expected Loss).
        """
        active_threshold = custom_threshold if custom_threshold is not None else self.optimal_threshold
        df_clean = self._prepare_dataframe(df)
        proba_matrix = self.pipeline.predict_proba(df_clean)
        churn_probs = proba_matrix[:, 1]

        result_df = df.copy()
        result_df["churn_probability"] = np.round(churn_probs, 4)
        result_df["optimal_threshold"] = active_threshold
        result_df["decision"] = np.where(churn_probs >= active_threshold, "INTERVENE", "DO_NOT_INTERVENE")

        # Compute Expected Loss = churn_probability * (MonthlyCharges * 12)
        m_charges = pd.to_numeric(result_df["MonthlyCharges"], errors="coerce").fillna(0.0)
        result_df["expected_loss"] = np.round(churn_probs * (m_charges * 12.0), 2)

        # Risk tiering
        conditions = [
            churn_probs >= 0.60,
            churn_probs >= active_threshold,
        ]
        choices = ["HIGH_RISK", "MODERATE_RISK"]
        result_df["risk_level"] = np.select(conditions, choices, default="LOW_RISK")

        # Sort descending by expected loss to prioritize intervention queue
        result_df = result_df.sort_values(by="expected_loss", ascending=False).reset_index(drop=True)
        return result_df


# Module-level singleton instance for convenient access
_inference_service: Optional[InferenceService] = None


def get_inference_service(model_path: Union[str, Path] = "models/telco_churn_lgbm_pipeline.joblib") -> InferenceService:
    global _inference_service
    if _inference_service is None:
        _inference_service = InferenceService(model_path=model_path)
    return _inference_service
