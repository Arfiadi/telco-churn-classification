"""Custom CSS for Modern Glassmorphism & High-End Aesthetic Styling in Streamlit."""

import streamlit as st

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"], [class*="st-"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Glassmorphism KPI Metric Cards */
.metric-card {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 12px;
    box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.35);
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    border-color: rgba(56, 189, 248, 0.4);
    box-shadow: 0 12px 28px -4px rgba(56, 189, 248, 0.15);
}

.metric-title {
    font-size: 0.78rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 6px;
}

.metric-value-danger {
    font-size: 2.1rem;
    font-weight: 800;
    color: #f87171;
    line-height: 1.1;
    font-feature-settings: "tnum";
}

.metric-value-warning {
    font-size: 2.1rem;
    font-weight: 800;
    color: #fbbf24;
    line-height: 1.1;
    font-feature-settings: "tnum";
}

.metric-value-success {
    font-size: 2.1rem;
    font-weight: 800;
    color: #34d399;
    line-height: 1.1;
    font-feature-settings: "tnum";
}

.metric-value-info {
    font-size: 2.1rem;
    font-weight: 800;
    color: #38bdf8;
    line-height: 1.1;
    font-feature-settings: "tnum";
}

.metric-subtext {
    font-size: 0.78rem;
    color: #64748b;
    margin-top: 6px;
    display: block;
}

/* Status Badges */
.badge-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 14px;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.025em;
}

.badge-high {
    background-color: rgba(239, 68, 68, 0.18);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.5);
}

.badge-mod {
    background-color: rgba(245, 158, 11, 0.18);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.5);
}

.badge-low {
    background-color: rgba(16, 185, 129, 0.18);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.5);
}

/* Explainability Factor Card */
.factor-card {
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(51, 65, 85, 0.6);
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 8px;
    display: flex;
    flex-direction: column;
    gap: 2px;
}

/* Strategic Action / Copilot Card */
.copilot-action-card {
    background: linear-gradient(145deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.85));
    border: 1px solid rgba(59, 130, 246, 0.35);
    border-radius: 16px;
    padding: 22px 24px;
    margin-top: 12px;
    box-shadow: 0 12px 32px -8px rgba(37, 99, 235, 0.2);
}

.copilot-script-bubble {
    background: #090d16;
    border: 1px solid #1e293b;
    border-left: 4px solid #38bdf8;
    border-radius: 12px;
    padding: 16px 18px;
    font-size: 0.92rem;
    line-height: 1.65;
    color: #e2e8f0;
    margin-top: 10px;
}

/* Global button styling enhancement */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    border: 1px solid rgba(147, 197, 253, 0.35);
    border-radius: 10px;
    padding: 8px 22px;
    font-weight: 700;
    font-size: 0.95rem;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
    transition: all 0.2s ease-in-out;
}

div.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.55);
}
</style>
"""

def apply_custom_css():
    """Inject custom glassmorphism styles into the Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
