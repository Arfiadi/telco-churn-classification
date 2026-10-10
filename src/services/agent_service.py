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

# Model Fallback Hierarchy (Free Tier on OpenRouter)
DEFAULT_MODEL_LADDER = [
    os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"),
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemini-2.0-flash-exp:free",
    "qwen/qwen-2.5-72b-instruct:free",
    "google/gemma-4-31b-it:free",
    "liquid/lfm-2.5-2.6b:free",
]

DEFAULT_GEMINI_LADDER = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
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
    Manages PydanticAI multi-tier LLM interaction with zero-downtime heuristic fallback.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_ladder: Optional[list[str]] = None,
        force_fallback: bool = False,
    ):
        self.force_fallback = force_fallback
        if force_fallback:
            self.api_provider = "none"
            self.api_key = None
            self.model_ladder = []
        elif os.getenv("GEMINI_API_KEY"):
            self.api_provider = "gemini"
            self.api_key = os.getenv("GEMINI_API_KEY")
            self.model_ladder = model_ladder or DEFAULT_GEMINI_LADDER
        else:
            self.api_provider = "openrouter"
            self.api_key = api_key if api_key is not None else os.getenv("OPENROUTER_API_KEY")
            self.model_ladder = model_ladder or DEFAULT_MODEL_LADDER

        self.heuristic_engine = HeuristicRetentionEngine()
        self.shap_service = get_shap_service()
        self.agent = create_retention_agent()

    def _get_client(self) -> Optional[Any]:
        """
        Backward-compatible inspection method for UI connection badges.
        Returns the active model instance or Provider if an API key is present.
        """
        if self.force_fallback or not self.api_key or self.api_key.startswith("your_"):
            return None
        return self._build_model_instance(self.model_ladder[0]) if self.model_ladder else None

    def _build_model_instance(self, model_name: str) -> Optional[Any]:
        """Creates a provider-backed model instance for PydanticAI."""
        if self.force_fallback or not self.api_key or self.api_key.startswith("your_"):
            return None

        try:
            if self.api_provider == "gemini":
                provider = GoogleProvider(api_key=self.api_key)
                return GoogleModel(model_name, provider=provider)
            else:
                provider = OpenRouterProvider(api_key=self.api_key)
                return OpenRouterModel(model_name, provider=provider)
        except Exception as e:
            logger.warning(f"Failed to create model instance for {model_name}: {e}")
            return None

    def generate_retention_plan(
        self,
        customer: CustomerProfile,
        diagnostic: Optional[CustomerDiagnostic] = None,
    ) -> RetentionPlan:
        """
        Generates retention plan by querying LLM ladder via PydanticAI Agent with tool calling,
        falling back seamlessly to HeuristicRetentionEngine if all models fail.
        """
        if diagnostic is None:
            diagnostic = self.shap_service.explain_customer(customer)

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

        # Iterate through model ladder
        for model_name in self.model_ladder:
            try:
                model_inst = self._build_model_instance(model_name)
                if model_inst is None:
                    continue

                logger.info(f"Invoking PydanticAI Agent with model: {model_name}")
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

        logger.warning(
            "All configured LLM models in ladder failed or were rate-limited. "
            "Falling back seamlessly to HeuristicRetentionEngine."
        )
        return self.heuristic_engine.generate_plan(customer, diagnostic)
