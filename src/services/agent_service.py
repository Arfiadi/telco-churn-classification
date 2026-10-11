"""
Agent Service Orchestrator for Telco Churn Intelligent Retention Platform.
Orchestrates PydanticAI Agent with deterministic tool calling and offline heuristic fallback.
"""

from dataclasses import dataclass
import logging
import os
from typing import Any, Optional
from dotenv import load_dotenv

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider

from src.services.agent_tools import (
    calculate_retention_roi as tools_calculate_retention_roi,
    get_eligible_retention_offers as tools_get_eligible_retention_offers,
    simulate_churn_impact as tools_simulate_churn_impact,
)
from src.services.fallback_engine import HeuristicRetentionEngine
from src.services.schemas import CustomerDiagnostic, CustomerProfile, RetentionPlan
from src.services.shap_service import get_shap_service

load_dotenv()
logger = logging.getLogger(__name__)


def _resolve_secret(key_name: str, passed_val: Optional[str] = None) -> Optional[str]:
    """
    Safely resolves API credentials from explicit argument, OS environment,
    or Streamlit Cloud secrets (st.secrets).
    """
    if passed_val and not str(passed_val).startswith("your_"):
        return str(passed_val).strip()

    # 1. Check os.environ
    env_val = os.getenv(key_name)
    if env_val and not str(env_val).startswith("your_"):
        return str(env_val).strip()

    # 2. Check Streamlit Cloud secrets (st.secrets)
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            try:
                # Direct match: st.secrets["GEMINI_API_KEY"]
                if key_name in st.secrets:
                    val = str(st.secrets[key_name]).strip()
                    if val and not val.startswith("your_"):
                        os.environ[key_name] = val
                        return val

                # Lowercase match: st.secrets["gemini_api_key"]
                key_lower = key_name.lower()
                if key_lower in st.secrets:
                    val = str(st.secrets[key_lower]).strip()
                    if val and not val.startswith("your_"):
                        os.environ[key_name] = val
                        return val

                # Section match: st.secrets["gemini"]["api_key"]
                section = "gemini" if "GEMINI" in key_name else "openrouter"
                if section in st.secrets and isinstance(st.secrets[section], dict):
                    val = str(st.secrets[section].get("api_key", "")).strip()
                    if val and not val.startswith("your_"):
                        os.environ[key_name] = val
                        return val
            except Exception:
                pass
    except Exception:
        pass

    return None


# Model Fallback Hierarchy (Free Tier on OpenRouter)
DEFAULT_MODEL_LADDER = [
    os.getenv("OPENROUTER_MODEL", "openrouter/free"),
    "openrouter/free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
]

# Google Gemini Active Models Ladder (Production-ready)
DEFAULT_GEMINI_LADDER = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-flash-latest",
]

SYSTEM_PROMPT = """Anda adalah "Telco Retention Copilot", asisten AI strategis tingkat eksekutif untuk tim Customer Success PT Telekomunikasi.
Tugas Anda adalah merumuskan rencana retensi presisi untuk pelanggan yang teridentifikasi berisiko churn.

ATURAN BISNIS & GUARDRAILS WAJIB (PELANGGARAN AKAN MENGAKIBATKAN REJEKSI SISTEM):
1. ANGGARAN KETAT: Total biaya penawaran (incentive_cost_usd) TIDAK BOLEH melebihi $20.00. Anda dilarang memberikan diskon atau insentif yang melebihi pagu ini.
2. DETERMINISTIK & DATA-DRIVEN: Anda WAJIB memanggil tool `simulate_churn_impact` untuk menguji penurunan probabilitas churn sebelum menyimpulkan rekomendasi paket. JANGAN MENEBAK probabilitas!
3. TARGET SHAP DRIVER: Rencana intervensi harus secara langsung mengatasi faktor risiko teratas (top risk drivers) dari analisis SHAP pelanggan. JANGAN PERNAH menawarkan migrasi atau downgrade kontrak kepada pelanggan yang sudah terikat kontrak 1 tahun atau 2 tahun.
4. NASKAH KOMUNIKASI & BAHASA: Tuliskan `outreach_script` dan `root_cause_diagnosis` seluruhnya dalam Bahasa Indonesia yang santun, personal, dan empatik, mengakui nilai hubungan pelanggan tanpa menyebutkan kata "kami mendeteksi Anda akan churn".
5. OUTPUT TERSTRUKTUR: Kembalikan respon akhir yang mematuhi skema RetentionPlan secara eksak.
"""


@dataclass
class RetentionDeps:
    """
    Dependency injection container for PydanticAI Agent.
    Provides customer context and diagnostic explainability factors to tools via RunContext.
    """
    customer: CustomerProfile
    diagnostic: CustomerDiagnostic


def create_retention_agent(model: Any = None) -> Agent[RetentionDeps, RetentionPlan]:
    """
    Factory function creating a PydanticAI Agent configured with deterministic
    retention engineering tools and typed RetentionPlan output.
    """
    agent: Agent[RetentionDeps, RetentionPlan] = Agent(
        model=model,
        output_type=RetentionPlan,
        deps_type=RetentionDeps,
        system_prompt=SYSTEM_PROMPT,
    )

    @agent.tool
    def get_eligible_retention_offers(ctx: RunContext[RetentionDeps]) -> list[dict[str, Any]]:
        """
        Mengambil katalog penawaran retensi yang telah disetujui perusahaan dengan pagu biaya <= $20.00.
        Menyaring penawaran yang relevan dengan profil pelanggan saat ini.
        """
        offers = tools_get_eligible_retention_offers(ctx.deps.customer)
        return [o.model_dump() for o in offers]

    @agent.tool
    def simulate_churn_impact(
        ctx: RunContext[RetentionDeps],
        feature_modifications: dict[str, Any],
    ) -> dict[str, float]:
        """
        Menjalankan pipeline Machine Learning LightGBM asli untuk menghitung probabilitas churn baru
        jika fitur pelanggan dimodifikasi (what-if analysis).
        """
        return tools_simulate_churn_impact(ctx.deps.customer, feature_modifications)

    @agent.tool
    def calculate_retention_roi(
        ctx: RunContext[RetentionDeps],
        incentive_cost: float,
        new_prob: float,
        monthly_charges: Optional[float] = None,
        current_prob: Optional[float] = None,
    ) -> dict[str, Any]:
        """
        Menghitung return-on-investment finansial retensi (Expected Net Profit) berdasarkan unit economics telco.
        Total biaya penawaran insentif (incentive_cost) WAJIB <= $20.00.
        """
        m_charges = monthly_charges if monthly_charges is not None else ctx.deps.customer.MonthlyCharges
        c_prob = current_prob if current_prob is not None else ctx.deps.diagnostic.baseline_churn_prob
        safe_cost = min(float(incentive_cost), 20.00)
        return tools_calculate_retention_roi(
            monthly_charges=m_charges,
            current_prob=c_prob,
            new_prob=float(new_prob),
            incentive_cost=safe_cost,
        )

    return agent


class AgentService:
    """
    Manages PydanticAI multi-tier LLM interaction with cross-provider resilience
    and zero-downtime heuristic fallback.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_ladder: Optional[list[str]] = None,
        force_fallback: bool = False,
    ):
        self.force_fallback = force_fallback
        self.model_ladder_override = model_ladder
        self.heuristic_engine = HeuristicRetentionEngine()
        self.shap_service = get_shap_service()
        self.agent = create_retention_agent()
        self._refresh_keys(api_key=api_key, model_ladder=model_ladder)

    def _refresh_keys(
        self,
        api_key: Optional[str] = None,
        model_ladder: Optional[list[str]] = None,
    ) -> None:
        """Dynamically re-reads environment or secrets for live key activation."""
        if self.force_fallback:
            self.api_provider = "none"
            self.api_key = None
            self.gemini_api_key = None
            self.openrouter_api_key = None
            self.model_ladder = []
            return

        self.gemini_api_key = _resolve_secret("GEMINI_API_KEY")
        self.openrouter_api_key = _resolve_secret("OPENROUTER_API_KEY", api_key)

        if api_key and not api_key.startswith("your_"):
            if api_key.startswith("sk-or-") or api_key.startswith("sk-"):
                self.openrouter_api_key = api_key.strip()
                os.environ["OPENROUTER_API_KEY"] = self.openrouter_api_key
            else:
                self.gemini_api_key = api_key.strip()
                os.environ["GEMINI_API_KEY"] = self.gemini_api_key

        if self.gemini_api_key and not self.gemini_api_key.startswith("your_"):
            self.api_provider = "gemini"
            self.api_key = self.gemini_api_key
            self.model_ladder = model_ladder or self.model_ladder_override or DEFAULT_GEMINI_LADDER
        elif self.openrouter_api_key and not self.openrouter_api_key.startswith("your_"):
            self.api_provider = "openrouter"
            self.api_key = self.openrouter_api_key
            self.model_ladder = model_ladder or self.model_ladder_override or DEFAULT_MODEL_LADDER
        else:
            self.api_provider = "none"
            self.api_key = None
            self.model_ladder = []

    def _get_client(self) -> Optional[Any]:
        """
        Backward-compatible inspection method for UI connection badges.
        Returns the active model instance if an API key is present.
        """
        self._refresh_keys()
        if self.force_fallback or not self.api_key or self.api_key.startswith("your_"):
            return None
        return self._build_model_instance(self.model_ladder[0]) if self.model_ladder else None

    def _build_model_instance(
        self,
        model_name: str,
        provider: Optional[str] = None,
    ) -> Optional[Any]:
        """Creates a provider-backed model instance for PydanticAI."""
        active_provider = provider or self.api_provider
        if self.force_fallback:
            return None

        try:
            if active_provider == "gemini" and self.gemini_api_key:
                p = GoogleProvider(api_key=self.gemini_api_key)
                return GoogleModel(model_name, provider=p)
            elif active_provider == "openrouter" and self.openrouter_api_key:
                p = OpenRouterProvider(api_key=self.openrouter_api_key)
                return OpenRouterModel(model_name, provider=p)
            return None
        except Exception as e:
            logger.warning(f"Failed to create model instance for {model_name} ({active_provider}): {e}")
            return None

    def generate_retention_plan(
        self,
        customer: CustomerProfile,
        diagnostic: Optional[CustomerDiagnostic] = None,
    ) -> RetentionPlan:
        """
        Generates retention plan by querying LLM ladder via PydanticAI Agent with tool calling,
        falling back across providers and seamlessly to HeuristicRetentionEngine if all fail.
        """
        if diagnostic is None:
            diagnostic = self.shap_service.explain_customer(customer)

        self._refresh_keys()
        if self.force_fallback or not self.api_key or self.api_key.startswith("your_"):
            logger.info("API Key not set, invalid, or offline mode forced. Using HeuristicRetentionEngine.")
            return self.heuristic_engine.generate_plan(customer, diagnostic)

        # Build detailed user prompt
        top_risks = [
            f"- {r.feature_name}: {r.human_explanation} (SHAP: +{r.shap_value:.4f})"
            for r in diagnostic.top_risk_drivers
        ]
        top_anchors = [
            f"- {a.feature_name}: {a.human_explanation} (SHAP: {a.shap_value:.4f})"
            for a in diagnostic.top_retention_anchors
        ]

        user_prompt = f"""PROFIL PELANGGAN:
- ID Pelanggan: {customer.customer_id}
- Masa Langganan: {customer.tenure} bulan
- Tagihan Bulanan: ${customer.MonthlyCharges:.2f}
- Jenis Kontrak: {customer.Contract}
- Internet Service: {customer.InternetService}
- Tech Support: {customer.TechSupport}
- Online Security: {customer.OnlineSecurity}
- Metode Pembayaran: {customer.PaymentMethod}

DIAGNOSTIK RISIKO:
- Probabilitas Churn Baseline: {diagnostic.baseline_churn_prob:.4f} (Threshold Bisnis: {diagnostic.decision_threshold:.2f})
- Status Risiko: {'BERISIKO CHURN' if diagnostic.is_at_risk else 'AMAN / AMBANG RENDAH'}

FAKTOR PENDORONG RISIKO UTAMA (TOP SHAP RISK DRIVERS):
{chr(10).join(top_risks)}

FAKTOR PELINDUNG UTAMA (TOP SHAP RETENTION ANCHORS):
{chr(10).join(top_anchors)}

INSTRUKSI EKSEKUTIF:
1. Panggil `get_eligible_retention_offers` untuk melihat opsi paket retensi yang disetujui. JANGAN PERNAH menawarkan migrasi/downgrade kontrak jika pelanggan sudah memiliki kontrak 1 tahun atau 2 tahun.
2. Pilih paket yang paling tepat mengatasi pemicu risiko teratas. Jika pelanggan berisiko rendah, prioritaskan program apresiasi loyalitas.
3. Panggil `simulate_churn_impact` untuk menguji paket tersebut pada pipeline ML.
4. Panggil `calculate_retention_roi` untuk memverifikasi keuntungan finansial.
5. Kembalikan respons terstruktur sesuai skema RetentionPlan. Seluruh diagnosis dan naskah outreach wajib dalam Bahasa Indonesia yang santun dan empatik.
"""

        deps = RetentionDeps(customer=customer, diagnostic=diagnostic)

        # 1. Primary Provider Ladder
        for model_name in self.model_ladder:
            try:
                model_inst = self._build_model_instance(model_name, provider=self.api_provider)
                if model_inst is None:
                    continue

                logger.info(f"Invoking PydanticAI Agent with primary model: {model_name} ({self.api_provider})")
                result = self.agent.run_sync(
                    user_prompt,
                    deps=deps,
                    model=model_inst,
                )

                plan: RetentionPlan = result.output
                source_tag = f"{self.api_provider.upper()}_AGENT"
                if plan.generation_source != source_tag:
                    plan = plan.model_copy(update={"generation_source": source_tag})

                # Validate budget guardrail
                if plan.incentive_cost_usd > 20.00:
                    plan = plan.model_copy(update={"incentive_cost_usd": 20.00})

                return plan

            except Exception as e:
                logger.warning(
                    f"Error running PydanticAI Agent with model '{model_name}': {e}. "
                    "Proceeding to next model in ladder."
                )
                continue

        # 2. Secondary Provider Fallover (Gemini -> OpenRouter)
        if self.api_provider == "gemini" and self.openrouter_api_key and not self.openrouter_api_key.startswith("your_"):
            for or_model in DEFAULT_MODEL_LADDER:
                try:
                    logger.info(f"Failing over to OpenRouter secondary model: {or_model}")
                    or_inst = self._build_model_instance(or_model, provider="openrouter")
                    if or_inst is None:
                        continue
                    result = self.agent.run_sync(user_prompt, deps=deps, model=or_inst)
                    plan = result.output
                    if plan.generation_source != "OPENROUTER_AGENT":
                        plan = plan.model_copy(update={"generation_source": "OPENROUTER_AGENT"})
                    if plan.incentive_cost_usd > 20.00:
                        plan = plan.model_copy(update={"incentive_cost_usd": 20.00})
                    return plan
                except Exception as e_or:
                    logger.warning(f"Error failover OpenRouter '{or_model}': {e_or}")
                    continue

        logger.warning(
            "All configured LLM models failed or were rate-limited. "
            "Falling back seamlessly to HeuristicRetentionEngine."
        )
        return self.heuristic_engine.generate_plan(customer, diagnostic)
