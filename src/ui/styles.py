"""Custom CSS for Modern Glassmorphism & High-End Aesthetic Styling in Streamlit."""

import streamlit as st

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');
@import url('https://fonts.googleapis.com/icon?family=Material+Icons');

/* Base typography for application shell and content */
html, body, .stApp, [data-testid="stAppViewContainer"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Ensure code elements use monospaced typography */
code, pre, .stCodeBlock {
    font-family: 'JetBrains Mono', monospace !important;
}

/* CRITICAL FIX: Strictly protect and enforce Material Symbols / Icons font */
/* Never allow custom typography to override Streamlit internal ligature icons */
[data-testid="stIconMaterial"],
[data-testid*="Icon"],
[data-testid="stSidebarCollapseButton"] span,
[data-testid="stExpanderStepChevron"] span,
[data-testid="stExpanderIcon"] span,
.material-symbols-rounded,
.material-symbols-outlined,
.material-symbols-sharp,
.material-icons,
[class*="material-symbols"],
[class*="material-icons"],
i[class*="material"],
span[class*="material"] {
    font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
    font-weight: normal !important;
    font-style: normal !important;
    font-size: 1.25rem !important;
    line-height: 1 !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    display: inline-block !important;
    white-space: nowrap !important;
    word-wrap: normal !important;
    direction: ltr !important;
    -webkit-font-feature-settings: 'liga' 1 !important;
    font-feature-settings: 'liga' 1 !important;
    -webkit-font-smoothing: antialiased !important;
}

/* Fix expander chevron spacing and alignment */
[data-testid="stExpander"] summary {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
}

[data-testid="stExpanderStepChevron"] {
    display: inline-flex !important;
    align-items: center !important;
    margin-right: 6px !important;
}

/* Column equal height container alignment */
div[data-testid="stColumn"] {
    display: flex !important;
    flex-direction: column !important;
}

div[data-testid="stColumn"] > div[data-testid="stVerticalBlock"] {
    display: flex !important;
    flex-direction: column !important;
    flex: 1 1 auto !important;
    height: 100% !important;
}

div[data-testid="stColumn"] > div[data-testid="stVerticalBlock"] > div.element-container {
    display: flex !important;
    flex-direction: column !important;
    flex: 1 1 auto !important;
}

div[data-testid="stColumn"] .stMarkdown {
    display: flex !important;
    flex-direction: column !important;
    flex: 1 1 auto !important;
    height: 100% !important;
}

/* Glassmorphism KPI Metric Cards */
.metric-card {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 14px;
    padding: 16px 18px;
    margin-bottom: 12px;
    box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.35);
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    min-height: 185px !important;
    height: 185px !important;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-sizing: border-box;
}

.metric-card:hover {
    transform: translateY(-2px);
    border-color: rgba(56, 189, 248, 0.4);
    box-shadow: 0 12px 28px -4px rgba(56, 189, 248, 0.15);
}

.metric-title {
    font-size: 0.76rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 2px;
    display: flex;
    align-items: center;
    gap: 6px;
    min-height: 22px;
    line-height: 1.25;
}

.metric-body {
    flex: 1 1 auto;
    display: flex;
    align-items: center;
    justify-content: flex-start;
    margin: 4px 0;
    width: 100%;
}

.metric-body-center {
    justify-content: center !important;
}

.metric-value-danger {
    font-size: clamp(1.2rem, 1.55vw, 1.7rem);
    font-weight: 800;
    color: #f87171;
    line-height: 1.15;
    font-feature-settings: "tnum";
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    letter-spacing: -0.01em;
}

.metric-value-warning {
    font-size: clamp(1.2rem, 1.55vw, 1.7rem);
    font-weight: 800;
    color: #fbbf24;
    line-height: 1.15;
    font-feature-settings: "tnum";
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    letter-spacing: -0.01em;
}

.metric-value-success {
    font-size: clamp(1.2rem, 1.55vw, 1.7rem);
    font-weight: 800;
    color: #34d399;
    line-height: 1.15;
    font-feature-settings: "tnum";
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    letter-spacing: -0.01em;
}

.metric-value-info {
    font-size: clamp(1.2rem, 1.55vw, 1.7rem);
    font-weight: 800;
    color: #38bdf8;
    line-height: 1.15;
    font-feature-settings: "tnum";
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    letter-spacing: -0.01em;
}

.metric-subtext {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 4px;
    display: block;
    min-height: 2.3em;
    line-height: 1.25;
    white-space: normal !important;
    word-break: break-word;
    overflow-wrap: break-word;
}

/* Status Badges */
.badge-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 16px;
    border-radius: 9999px;
    font-size: 0.95rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
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

/* Modern Glassmorphic Container for Plotly Charts */
div[data-testid="stPlotlyChart"] {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 14px;
    padding: 12px 14px;
    box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.35);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

div[data-testid="stPlotlyChart"]:hover {
    border-color: rgba(56, 189, 248, 0.3);
    box-shadow: 0 12px 28px -4px rgba(56, 189, 248, 0.12);
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
div.stButton > button[kind="primary"],
div.stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    border: 1px solid rgba(147, 197, 253, 0.35) !important;
    border-radius: 10px !important;
    padding: 8px 22px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
    transition: all 0.2s ease-in-out !important;
}

div.stButton > button[kind="primary"]:hover,
div.stButton > button[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.55) !important;
}
</style>
"""


def apply_custom_css():
    """Inject custom glassmorphism styles into the Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
