"""Telco Churn Intelligent Retention Platform (Phase 2).

Main Entrypoint and Modern Multi-Page Navigation Hub.
"""

import warnings
import streamlit as st

# Suppress external library warnings (e.g., from SHAP)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

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
        title="Single customer profiler & copilot",
        icon=":material/person:",
        default=True,
    ),
    st.Page(
        "pages/2_batch_queue.py",
        title="Batch work queue priority",
        icon=":material/dataset:",
    ),
    st.Page(
        "pages/3_governance.py",
        title="Executive governance & diagnostics",
        icon=":material/analytics:",
    ),
]

nav = st.navigation(pages)
nav.run()
