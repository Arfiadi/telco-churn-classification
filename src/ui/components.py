import plotly.graph_objects as go
import streamlit as st


def render_risk_gauge(churn_prob: float, threshold: float = 0.39):
    """Render a modern Plotly Gauge Chart for visual churn risk with traffic-light zones."""
    prob_pct = churn_prob * 100
    thresh_pct = threshold * 100

    # Determine needle/bar color based on risk level
    if prob_pct < thresh_pct:
        bar_color = "#34d399"  # Emerald Green (Low)
    elif prob_pct < 60:
        bar_color = "#fbbf24"  # Amber Yellow (Moderate)
    else:
        bar_color = "#f87171"  # Coral Red (High)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob_pct,
            number={
                "suffix": "%",
                "font": {"size": 28, "color": "#f8fafc", "family": "Plus Jakarta Sans"},
            },
            title={
                "text": "<b>Indikator Risiko Churn</b><br><span style='font-size:0.75rem;color:#94a3b8;'>Tingkat Probabilitas Pelanggan</span>",
                "font": {"size": 13, "color": "#cbd5e1"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#64748b",
                    "tickfont": {"color": "#94a3b8", "size": 10},
                },
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": "rgba(15, 23, 42, 0.4)",
                "borderwidth": 1,
                "bordercolor": "rgba(51, 65, 85, 0.4)",
                "steps": [
                    {"range": [0, thresh_pct], "color": "rgba(16, 185, 129, 0.22)"},
                    {"range": [thresh_pct, 60], "color": "rgba(245, 158, 11, 0.22)"},
                    {"range": [60, 100], "color": "rgba(239, 68, 68, 0.32)"},
                ],
                "threshold": {
                    "line": {"color": "#ef4444", "width": 3},
                    "thickness": 0.8,
                    "value": thresh_pct,
                },
            },
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        margin=dict(l=20, r=20, t=35, b=10),
        height=190,
    )
    st.plotly_chart(fig, width="stretch", alt="Visual Gauge Indikator Risiko Churn")


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
