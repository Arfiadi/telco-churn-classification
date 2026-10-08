"""Reusable UI components for Streamlit."""

import streamlit as st


def render_metric_card(title: str, value: str, subtext: str, color_class: str = "metric-value-info"):
    """Render a modern glassmorphism KPI metric card."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="{color_class}">
                {value}
            </div>
            <span class="metric-subtext">{subtext}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_decision_card(title: str, decision: str, subtext: str, is_intervene: bool):
    """Render a decision card with appropriate color indicator."""
    decision_color = "#ef4444" if is_intervene else "#10b981"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div style="font-size: 1.55rem; font-weight: 800; color: {decision_color}; margin-top: 4px;">
                {decision}
            </div>
            <span class="metric-subtext">{subtext}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_badge_card(title: str, risk_level: str, subtext: str):
    """Render a risk level card with colored badge pill."""
    badge_class = (
        "badge-high"
        if risk_level == "HIGH_RISK"
        else ("badge-mod" if risk_level == "MODERATE_RISK" else "badge-low")
    )
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div style="margin-top: 6px;">
                <span class="badge-pill {badge_class}">{risk_level}</span>
            </div>
            <span class="metric-subtext">{subtext}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
