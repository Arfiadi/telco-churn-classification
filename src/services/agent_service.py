"""
Agent Service Orchestrator for Telco Churn Intelligent Retention Platform.
Orchestrates OpenRouter LLM ladder with function/tool calling and offline heuristic fallback.
"""

import json
import logging
import os
from typing import Any, Optional
from dotenv import load_dotenv
from openai import OpenAI

from src.services.agent_tools import (
    calculate_retention_roi,
    get_eligible_retention_offers,
    simulate_churn_impact,
)
from src.services.fallback_engine import HeuristicRetentionEngine
from src.services.schemas import CustomerDiagnostic, CustomerProfile, RetentionPlan
from src.services.shap_service import get_shap_service

load_dotenv()
logger = logging.getLogger(__name__)

# Model Fallback Hierarchy (Free Tier on OpenRouter)
DEFAULT_MODEL_LADDER = [
    os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free"),
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "liquid/lfm-2.5-2.6b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemini-2.0-flash-exp:free",
]

SYSTEM_PROMPT = """Anda adalah "Telco Retention Copilot", asisten AI strategis tingkat eksekutif untuk tim Customer Success PT Telekomunikasi.
Tugas Anda adalah merumuskan rencana retensi presisi untuk pelanggan yang teridentifikasi berisiko churn.

ATURAN BISNIS & GUARDRAILS WAJIB (PELANGGARAN AKAN MENGAKIBATKAN REJEKSI SISTEM):
1. ANGGARAN KETAT: Total biaya penawaran (incentive_cost_usd) TIDAK BOLEH melebihi $20.00. Anda dilarang memberikan diskon atau insentif yang melebihi pagu ini.
2. DETERMINISTIK & DATA-DRIVEN: Anda WAJIB memanggil tool `simulate_churn_impact` untuk menguji penurunan probabilitas churn sebelum menyimpulkan rekomendasi paket. JANGAN MENEBAK probabilitas!
3. TARGET SHAP DRIVER: Rencana intervensi harus secara langsung mengatasi faktor risiko teratas (top risk drivers) dari analisis SHAP pelanggan.
4. NASKAH KOMUNIKASI: Tuliskan `outreach_script` dalam bahasa yang santun, personal, dan empatik, mengakui nilai hubungan pelanggan tanpa menyebutkan kata "kami mendeteksi Anda akan churn".
5. OUTPUT TERSTRUKTUR: Seluruh respon akhir harus berupa JSON valid yang mematuhi skema berikut secara eksak:
```json
{
  "root_cause_diagnosis": "<Ringkasan 2 kalimat akar masalah>",
  "recommended_package_name": "<Nama paket retensi>",
  "incentive_cost_usd": <Biaya <= 20.0>,
  "simulated_churn_prob": <Probabilitas baru hasil simulate_churn_impact>,
  "risk_reduction_pct": <Persentase penurunan risiko>,
  "projected_net_value_usd": <Nilai keuntungan bersih>,
  "outreach_script": "<Naskah percakapan empati>",
  "confidence_level": "HIGH"
}
```
"""

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_eligible_retention_offers",
            "description": "Mengambil katalog penawaran retensi yang telah disetujui perusahaan dengan pagu biaya <= $20.00.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "simulate_churn_impact",
            "description": "Menjalankan pipeline Machine Learning LightGBM asli untuk menghitung probabilitas churn baru jika fitur pelanggan dimodifikasi (what-if analysis).",
            "parameters": {
                "type": "object",
                "properties": {
                    "feature_modifications": {
                        "type": "object",
                        "description": "Dictionary modifikasi fitur, contoh: {'Contract': 'One year'} atau {'TechSupport': 'Yes'}.",
                    }
                },
                "required": ["feature_modifications"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_retention_roi",
            "description": "Menghitung return-on-investment finansial retensi (Expected Net Profit) berdasarkan unit economics telco.",
            "parameters": {
                "type": "object",
                "properties": {
                    "monthly_charges": {"type": "number", "description": "Tagihan bulanan pelanggan saat ini."},
                    "current_prob": {"type": "number", "description": "Probabilitas churn baseline sebelum intervensi."},
                    "new_prob": {"type": "number", "description": "Probabilitas churn baru hasil simulasi."},
                    "incentive_cost": {"type": "number", "description": "Biaya penawaran insentif (wajib <= $20.00)."},
                },
                "required": ["monthly_charges", "current_prob", "new_prob", "incentive_cost"],
            },
        },
    },
]


class AgentService:
    """
    Manages multi-tier LLM interaction via OpenRouter with zero-downtime heuristic fallback.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_ladder: Optional[list[str]] = None,
        force_fallback: bool = False,
    ):
        self.force_fallback = force_fallback
        if force_fallback:
            self.api_key = None
        else:
            self.api_key = api_key if api_key is not None else os.getenv("OPENROUTER_API_KEY")
        self.model_ladder = model_ladder or DEFAULT_MODEL_LADDER
        self.heuristic_engine = HeuristicRetentionEngine()
        self.shap_service = get_shap_service()

    def _get_client(self) -> Optional[OpenAI]:
        """Creates OpenRouter-compatible OpenAI client if API key is present."""
        if self.force_fallback or not self.api_key or self.api_key.startswith("your_"):
            return None
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/telco-churn-retention-copilot",
                "X-Title": "Telco Churn Intelligent Retention Platform",
            },
        )

    def _execute_tool(self, name: str, args: dict[str, Any], customer: CustomerProfile) -> dict[str, Any]:
        """Safely executes a local deterministic tool call requested by the agent."""
        if name == "get_eligible_retention_offers":
            offers = get_eligible_retention_offers(customer)
            return {"offers": [o.model_dump() for o in offers]}
        elif name == "simulate_churn_impact":
            mods = args.get("feature_modifications", {})
            return simulate_churn_impact(customer, mods)
        elif name == "calculate_retention_roi":
            m_charges = float(args.get("monthly_charges", customer.MonthlyCharges))
            c_prob = float(args.get("current_prob", 0.5))
            n_prob = float(args.get("new_prob", 0.3))
            cost = float(args.get("incentive_cost", 15.0))
            return calculate_retention_roi(m_charges, c_prob, n_prob, cost)
        else:
            return {"error": f"Tool '{name}' tidak ditemukan."}

    def generate_retention_plan(
        self,
        customer: CustomerProfile,
        diagnostic: Optional[CustomerDiagnostic] = None,
    ) -> RetentionPlan:
        """
        Generates retention plan by querying OpenRouter LLM ladder with function calling,
        falling back seamlessly to HeuristicRetentionEngine if all models fail.
        """
        if diagnostic is None:
            diagnostic = self.shap_service.explain_customer(customer)

        client = self._get_client()
        if client is None:
            logger.info("OpenRouter API Key not set or invalid. Using HeuristicRetentionEngine.")
            return self.heuristic_engine.generate_plan(customer, diagnostic)

        # Build initial prompt messages
        top_risks = [f"- {r.feature_name}: {r.human_explanation} (SHAP: +{r.shap_value:.4f})" for r in diagnostic.top_risk_drivers]
        top_anchors = [f"- {a.feature_name}: {a.human_explanation} (SHAP: {a.shap_value:.4f})" for a in diagnostic.top_retention_anchors]

        user_content = f"""PROFIL PELANGGAN:
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

INSTRUKSI:
1. Panggil `get_eligible_retention_offers` untuk melihat opsi paket retensi yang disetujui.
2. Pilih paket yang paling tepat mengatasi pemicu risiko teratas.
3. Panggil `simulate_churn_impact` untuk menguji paket tersebut pada pipeline ML.
4. Panggil `calculate_retention_roi` untuk memverifikasi keuntungan finansial.
5. Kembalikan respons akhir dalam format JSON eksak sesuai skema RetentionPlan.
"""

        # Try each model in the fallback ladder
        for model_name in self.model_ladder:
            try:
                logger.info(f"Invoking OpenRouter model: {model_name}")
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ]

                # Interactive tool calling loop (max 4 turns)
                for _ in range(4):
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        tools=TOOL_DEFINITIONS,
                        tool_choice="auto",
                        temperature=0.2,
                    )
                    choice = response.choices[0]
                    message = choice.message

                    if message.tool_calls:
                        messages.append(message)
                        for tool_call in message.tool_calls:
                            func_name = tool_call.function.name
                            func_args = json.loads(tool_call.function.arguments or "{}")
                            tool_result = self._execute_tool(func_name, func_args, customer)
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(tool_result),
                            })
                    else:
                        # Final response received
                        content = message.content or ""
                        # Extract JSON from response
                        plan_dict = self._extract_json(content)
                        if plan_dict:
                            normalized = self._normalize_plan_dict(plan_dict, customer, diagnostic)
                            if normalized:
                                return RetentionPlan(**normalized)
                        break

            except Exception as e:
                logger.warning(f"Error querying model {model_name}: {e}. Trying next model in ladder.")
                continue

        logger.warning("All OpenRouter models in ladder failed or were rate-limited. Falling back to heuristic engine.")
        return self.heuristic_engine.generate_plan(customer, diagnostic)

    def _normalize_plan_dict(
        self,
        plan_dict: dict[str, Any],
        customer: CustomerProfile,
        diagnostic: CustomerDiagnostic,
    ) -> Optional[dict[str, Any]]:
        """
        Defensively normalizes alternate keys and validates numeric fields before Pydantic parsing.
        """
        # Map alternate key names
        key_aliases = {
            "root_cause_diagnosis": ["diagnosis", "root_cause", "summary"],
            "recommended_package_name": ["package_name", "package", "recommendation"],
            "incentive_cost_usd": ["cost", "incentive_cost", "cost_usd"],
            "simulated_churn_prob": ["new_churn_prob", "simulated_prob", "simulated_probability"],
            "risk_reduction_pct": ["risk_reduction", "reduction_pct", "churn_reduction"],
            "projected_net_value_usd": ["net_value", "net_profit", "roi_usd"],
            "outreach_script": ["script", "communication_script", "pitch"],
            "confidence_level": ["confidence", "confidence_score"],
        }

        normalized = dict(plan_dict)
        for target, aliases in key_aliases.items():
            if target not in normalized:
                for alias in aliases:
                    if alias in normalized:
                        normalized[target] = normalized[alias]
                        break

        # Check required string fields or provide defaults from diagnostic
        if "root_cause_diagnosis" not in normalized:
            risk_summary = ", ".join([r.feature_name.split("__")[-1] for r in diagnostic.top_risk_drivers[:2]])
            normalized["root_cause_diagnosis"] = f"Risiko churn dipicu oleh faktor dominan: {risk_summary}."

        if "recommended_package_name" not in normalized:
            normalized["recommended_package_name"] = "Contract Migration Shield"

        if "outreach_script" not in normalized:
            normalized["outreach_script"] = (
                f"Halo Bapak/Ibu, terima kasih telah setia menggunakan layanan kami. "
                f"Kami menyiapkan program apresiasi khusus untuk meningkatkan kenyamanan Anda."
            )

        # Enforce budget guardrails on incentive_cost_usd
        try:
            cost = float(normalized.get("incentive_cost_usd", 15.0))
            normalized["incentive_cost_usd"] = min(cost, 20.00)
        except (ValueError, TypeError):
            normalized["incentive_cost_usd"] = 15.00

        # Numeric conversions
        try:
            normalized["simulated_churn_prob"] = float(normalized.get("simulated_churn_prob", 0.35))
            normalized["risk_reduction_pct"] = float(normalized.get("risk_reduction_pct", 30.0))
            normalized["projected_net_value_usd"] = float(normalized.get("projected_net_value_usd", 50.0))
        except (ValueError, TypeError):
            return None

        # Confidence level regex enforcement
        conf = str(normalized.get("confidence_level", "HIGH")).upper()
        if conf not in ["HIGH", "MEDIUM", "LOW"]:
            conf = "HIGH"
        normalized["confidence_level"] = conf

        normalized["generation_source"] = "OPENROUTER_AGENT"
        return normalized

    def _extract_json(self, text: str) -> Optional[dict[str, Any]]:
        """Extracts JSON object from text containing possible markdown formatting."""
        try:
            return json.loads(text)
        except Exception:
            pass

        # Try to find json code block
        if "```json" in text:
            try:
                extracted = text.split("```json")[1].split("```")[0].strip()
                return json.loads(extracted)
            except Exception:
                pass
        elif "```" in text:
            try:
                extracted = text.split("```")[1].split("```")[0].strip()
                return json.loads(extracted)
            except Exception:
                pass

        # Try finding opening and closing braces
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                pass

        return None
