"""Telco Churn Intelligent Retention Platform (Phase 2).

Main Entrypoint and Modern Multi-Page Navigation Hub.
"""

import streamlit as st

# Global App Configuration
st.set_page_config(
    page_title="Telco Churn Intelligent Retention Platform",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Modern 2026 Multi-Page Navigation definition
pages = [
    st.Page(
        "pages/1_profiler.py",
        title="Single Customer Profiler & Copilot",
        icon="👤",
        default=True,
    ),
    st.Page(
        "pages/2_batch_queue.py",
        title="Batch Work Queue Priority",
        icon="📁",
    ),
    st.Page(
        "pages/3_governance.py",
        title="Executive Governance & Diagnostics",
        icon="📈",
    ),
]

nav = st.navigation(pages)
nav.run()
