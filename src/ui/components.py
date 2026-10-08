import plotly.graph_objects as go
import streamlit as st


def render_risk_gauge(churn_prob: float, threshold: float = 0.39):
    """Render a modern glassmorphism KPI gauge card perfectly aligned with the metric card family."""
    prob_pct = max(0.0, min(100.0, churn_prob * 100))
    thresh_pct = max(0.0, min(100.0, threshold * 100))

    if prob_pct < thresh_pct:
        bar_color = "#34d399"  # Emerald Green (Low)
        val_color = "#34d399"
        status_label = "Aman"
        badge_class = "badge-low"
    elif prob_pct < 60:
        bar_color = "#fbbf24"  # Amber Yellow (Moderate)
        val_color = "#fbbf24"
        status_label = "Waspada"
        badge_class = "badge-mod"
    else:
        bar_color = "#f87171"  # Coral Red (High)
        val_color = "#f87171"
        status_label = "Kritis"
        badge_class = "badge-high"

    card_html = (
        f'<div class="metric-card">'
        f'<div class="metric-title">📊 Probabilitas Churn</div>'
        f'<div class="metric-body" style="flex-direction: column; align-items: flex-start; justify-content: center; gap: 8px; width: 100%;">'
        f'<div style="display: flex; align-items: baseline; justify-content: space-between; width: 100%;">'
        f'<span style="font-size: clamp(1.4rem, 1.8vw, 1.9rem); font-weight: 800; color: {val_color}; line-height: 1.1; font-family: \'Plus Jakarta Sans\', sans-serif;">{prob_pct:.1f}%</span>'
        f'<span class="badge-pill {badge_class}" style="font-size: 0.72rem; padding: 2px 8px; margin: 0;">{status_label}</span>'
        f'</div>'
        f'<div style="width: 100%; position: relative; margin-top: 2px; padding-top: 4px;">'
        f'<div style="width: 100%; height: 7px; background: rgba(255, 255, 255, 0.08); border-radius: 999px; overflow: hidden; position: relative;">'
        f'<div style="width: {prob_pct:.1f}%; height: 100%; background: {bar_color}; border-radius: 999px; transition: width 0.3s ease;"></div>'
        f'</div>'
        f'<div style="position: absolute; left: {thresh_pct:.1f}%; top: 0px; bottom: -2px; width: 2px; background: #ef4444; border-radius: 1px; z-index: 2;" title="Ambang Batas: {thresh_pct:.0f}%"></div>'
        f'</div>'
        f'</div>'
        f'<div style="display: flex; justify-content: space-between; align-items: center; width: 100%;">'
        f'<span class="metric-subtext" style="margin: 0;">Ambang: {thresh_pct:.0f}%</span>'
        f'<span style="font-size: 0.72rem; color: #ef4444; font-weight: 600;">Garis Ambang &tau;*</span>'
        f'</div>'
        f'</div>'
    )
    st.html(card_html)


def render_metric_card(title: str, value: str, subtext: str, color_class: str = "metric-value-info"):
    """Render a modern glassmorphism KPI metric card."""
    card_html = (
        f'<div class="metric-card">'
        f'<div class="metric-title">{title}</div>'
        f'<div class="metric-body"><div class="{color_class}">{value}</div></div>'
        f'<span class="metric-subtext">{subtext}</span>'
        f'</div>'
    )
    st.html(card_html)


def render_decision_card(title: str, decision: str, subtext: str, is_intervene: bool):
    """Render a decision card with aligned badge styling matching risk status."""
    if is_intervene:
        badge_class = "badge-high"
        icon = "🚨"
        decision_label = decision.title().replace("_", " ")
    else:
        badge_class = "badge-low"
        icon = "✅"
        decision_label = "Retained (Aman)"

    card_html = (
        f'<div class="metric-card">'
        f'<div class="metric-title">{title}</div>'
        f'<div class="metric-body"><div class="badge-pill {badge_class}"><span>{icon}</span> {decision_label}</div></div>'
        f'<span class="metric-subtext">{subtext}</span>'
        f'</div>'
    )
    st.html(card_html)


def render_badge_card(title: str, risk_level: str, subtext: str):
    """Render a risk level card with prominent colored status badge."""
    risk_clean = risk_level.replace("_", " ").title()
    if risk_level == "HIGH_RISK":
        badge_class = "badge-high"
        icon = "🔴"
    elif risk_level == "MODERATE_RISK":
        badge_class = "badge-mod"
        icon = "🟡"
    else:
        badge_class = "badge-low"
        icon = "🟢"

    card_html = (
        f'<div class="metric-card">'
        f'<div class="metric-title">{title}</div>'
        f'<div class="metric-body"><div class="badge-pill {badge_class}"><span>{icon}</span> {risk_clean}</div></div>'
        f'<span class="metric-subtext">{subtext}</span>'
        f'</div>'
    )
    st.html(card_html)


def render_factor_card(feature_name: str, shap_val: float, explanation: str, is_risk: bool = True):
    """Render a SHAP feature attribution card."""
    border_color = "#f87171" if is_risk else "#34d399"
    val_prefix = "+" if shap_val > 0 else ""
    st.markdown(
        f"""
        <div class="factor-card" style="border-left: 3px solid {border_color};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <strong style="color: {border_color};">{feature_name}</strong>
                <span style="font-family: monospace; font-size: 0.85rem; color: {border_color};">{val_prefix}{shap_val:.4f}</span>
            </div>
            <span style="font-size: 0.84rem; color: #cbd5e1;">{explanation}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
