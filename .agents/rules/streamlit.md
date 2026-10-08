---
trigger: always_on
description: Modern Streamlit 2026 Best Practices and Architectural Guidelines
---

# Streamlit 2026 Expert Rule

As an AI Assistant, whenever you write, refactor, or structure a Streamlit application, you MUST adhere to the following modern standards (as of 2026). Do NOT use legacy workarounds, outdated caching methods, or monolithic structures.

## 1. Architectural & Structure Best Practices
- **Decouple UI from Logic:** Do NOT write heavy logic (ML models, data processing) directly in `app.py`. Separate them into modules like `services/`, `models/`, or `utils/`.
- **Modern Multi-Page:** For multi-page apps, ALWAYS use `st.navigation` and `st.Page` along with the `pages/` directory instead of legacy custom routing.

## 2. State & Rerun Management (Crucial)
- **`st.session_state`:** Persist all form inputs and filter settings using `st.session_state` so the app doesn't reset awkwardly during reruns.
- **Partial Reruns:** Use the `@st.fragment` decorator to isolate functions that require frequent UI updates (preventing the whole app from rerunning).
- **Dynamic Containers:** Leverage `on_change` parameters on widgets like `st.tabs` and `st.popover` to manage reactivity effectively.
- **Live Widgets:** Use `live=True` on `st.text_input` only if you need the UI to update immediately as the user types.

## 3. Caching (Do NOT use the legacy `st.cache`)
- **`@st.cache_data`:** Use this for functions returning serializable data (Pandas DataFrames, dictionaries, API calls, CSV loads).
- **`@st.cache_resource`:** Use this EXCLUSIVELY for long-lived, non-serializable objects (e.g., PyTorch/TensorFlow Models, Database Connections).
- **Async Caching:** Streamlit now natively supports `async`/`await`. You can use async functions inside these cache decorators.

## 4. Modern UI & Visualizations
- **Drawers & Dialogs:** Use `st.dialog(..., position="left"|"right")` if you need side drawers or modal popups, avoiding hacky CSS.
- **Advanced Charts:** Use native `st.echarts_chart` for Apache ECharts when Plotly or Altair is insufficient.
- **Accessibility:** Always provide the `alt` parameter for images, media, and charts.

## 5. Data Handling for Machine Learning
- **Memory Efficiency:** Prefer efficient formats like Parquet over CSV when loading local data.
- **Avoid Heavy Uploads:** Do not use `st.file_uploader` for datasets > 500MB; simulate cloud storage or use chunking.

Always optimize for responsiveness and strictly follow this guideline to ensure the ML applications remain scalable and maintainable.
