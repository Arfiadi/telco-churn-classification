# Telco Customer Churn: Predictive Pipeline & Profit-Driven Decisioning

> **Proyek AI & Data Science:** Arsitektur *Intelligent Retention* & Penerapan Machine Learning End-to-End dengan Data IBM Telco Customer Churn.
> **Dokumentasi Lengkap:** Detail arsitektur data science, kamus variabel, audit pencegahan *data leakage*, dan evaluasi teknis tersedia di [docs/methodologi.md](docs/methodologi.md).
> **Live Demo (Web App):** 🚀 [https://telco-churn-agent-support.streamlit.app/](https://telco-churn-agent-support.streamlit.app/)
---

## 1. Problem Statement & Business Objective
Pada industri telekomunikasi berbasis langganan, mempertahankan pelanggan lama (*retention*) membutuhkan biaya yang jauh lebih rendah daripada mengakuisisi pelanggan baru (*acquisition*). 

Proyek ini mengatasi kelemahan pemodelan churn konvensional:
1. **Pencegahan Data Leakage:** Mengisolasi seluruh proses pembersihan, imputasi, dan penskalaan data ke dalam `Pipeline` Scikit-Learn.
2. **Rekayasa Fitur Perilaku:** Menambahkan indikator fluktuasi tagihan (*price shock*) dan tingkat keterikatan produk (*stickiness*).
3. **Thresholding Berbasis Nilai Bisnis:** Menentukan ambang batas keputusan klasifikasi menggunakan matriks biaya-manfaat (*Cost-Benefit Matrix*) pada prediksi Out-Of-Fold, bukan sekadar metrik akurasi dengan threshold default 0.50.

---

## 2. Alur Metodologi & Arsitektur Solusi

```
Raw Data (IBM Telco)
       │
       ▼
[Stateless Cleaning] (Data Types, Null Handling)
       │
       ▼
[Stratified Train-Test Split] (80% Train, 20% Unseen Holdout)
       │
       ├───────────────────────────────────────┐
       ▼ (Training Pipeline)                   ▼ (Holdout Evaluation)
[Custom Feature Engineering]              [Pipeline Inference]
  - TotalServices                           (Zero Data Leakage)
  - AverageMonthlyCost                         │
  - MonthlyChargesRatio                        ▼
  - TenureGroup                           [Model Evaluation]
       │                                    - ROC-AUC: 0.835
       ▼                                    - Bias-Variance Audited
[ColumnTransformer Preprocessing]              │
  - Median Imputer + StandardScaler (Numeric)  ▼
  - OneHotEncoder (Categorical)           [Holdout Business Impact]
       │                                    - Threshold: 0.39
       ▼                                    - Net Profit: $3,790
[LightGBM Classifier]                       - 243 Churners Saved
       │
       ▼
[Optuna Bayesian Optimization] (5-Fold Stratified CV on Train)
       │
       ▼
[Learning Curve Diagnostics] (Plateau ~2,800 samples, Gap ~0.02)
       │
       ▼
[OOF Profit Threshold Optimization] (Optimal Tau* = 0.39)
       │
       ▼
[Model Interpretability] (TreeSHAP Global & Local Waterfall)
       │
       ▼
[Pipeline Serialization] (telco_churn_lgbm_pipeline.joblib)
```

---

## 3. Rekayasa Fitur (Domain Feature Engineering)
Dibentuk 4 variabel baru melalui custom transformer `TelcoFeatureEngineer`:
* **`TotalServices`:** Akumulasi layanan digital aktif (keamanan, proteksi, dukungan teknis). Mengukur switching barrier pelanggan.
* **`AverageMonthlyCost`:** Rata-rata pengeluaran bulanan historis ($\text{TotalCharges} / \text{tenure}$).
* **`MonthlyChargesRatio`:** Rasio tagihan berjalan terhadap rata-rata historis. Nilai $> 1.0$ mengindikasikan kenaikan tarif atau masa promo diskon yang berakhir.
* **`TenureGroup`:** Kohort masa langganan (`New`: $\le 12$ bulan, `Growing`: $13-24$, `Mature`: $25-48$, `Loyal`: $> 48$).

---

## 4. Evaluasi Model & Dampak Bisnis

### A. Metrik Teknis Pemodelan
* **Baseline CV ROC-AUC (PyCaret Benchmark):** ~0.8468 (sinyal mentah data tabular).
* **Tuned LightGBM CV ROC-AUC (Optuna TPE):** 0.8488.
* **Holdout Test Set ROC-AUC (Unseen Data):** 0.8354.
* **Learning Curve Gap:** ~0.0198 antara skor latih dan validasi pada kapasitas penuh (menunjukkan model seimbang dan tidak mengalami overfitting).

### B. Evaluasi Finansial (Cost-Benefit Framework)
*Catatan Transparansi Skenario: Parameter biaya voucher ($20), margin CLV terselamatkan ($100), dan efektivitas intervensi (50%) merupakan asumsi skenario simulasi bisnis (unit economics modeling) untuk mendemonstrasikan kerangka kerja Expected Value, bukan variabel observasional dari dataset mentah IBM.*

*Asumsi Unit Economics Retensi:*
* Biaya Voucher/Intervensi Retensi ($C$): **$20**
* Margin Nilai Pelanggan Terselamatkan ($V$): **$100**
* Efektivitas Intervensi Retensi ($p_{\text{save}}$): **50%**
* Net Gain True Positive: **+$30** | Net Loss False Positive: **-$20**

**Hasil Uji pada Holdout Test Set (1.409 Pelanggan):**
| Strategi Keputusan | Threshold | Expected Net Profit | Pelanggan Terselamatkan (TP) | False Positives (FP) | Churn Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Tanpa Intervensi (Baseline) | N/A | $0 | 0 | 0 | 0.0% |
| Default Model ML | 0.50 | $3,720 | 198 | 111 | 52.9% |
| **Profit-Optimized ML ($\tau^*$)** | **0.39** | **$3,790** | **243** | **175** | **65.0%** |

*Optimasi threshold pada Out-Of-Fold meningkatkan profit bersih kampanye sebesar +$70 dan menangkap 45 calon churner tambahan pada holdout test set dibandingkan threshold default 0.50.*

---

## 5. Interpretasi Model (SHAP)
* **Faktor Risiko Utama (+ Churn):** Ketiadaan kontrak jangka panjang (Month-to-month), penggunaan internet *fiber optic* (churn 41.9%), metode pembayaran *electronic check* (churn 45.3%), dan rata-rata tagihan bulanan tinggi (*AverageMonthlyCost*).
* **Faktor Pelindung Utama (- Churn):** Kontrak dua tahun (churn hanya 2.8%), masa langganan panjang (*tenure*), dan kepemilikan layanan proteksi (*OnlineSecurity* & *TechSupport*).
* **Temuan Terhadap Fitur Baru:** *AverageMonthlyCost* terbukti menjadi salah satu pendorong signifikan (Rank #8 SHAP), sedangkan *MonthlyChargesRatio* memiliki kontribusi marjinal (+0.012 korelasi linier) karena pola tagihan sebagian besar pelanggan bersifat flat.


---

## 6. Struktur File Repository
```
├── app.py                                 <- Entrypoint aplikasi web Streamlit (Modern st.navigation)
├── pages/
│   ├── 1_profiler.py                      <- Profil risiko pelanggan tunggal, What-If sandbox, & AI Copilot
│   ├── 2_batch_queue.py                   <- Skoring batch CSV & antrean kerja prioritas (Expected Loss)
│   └── 3_governance.py                    <- Tata kelola model, SHAP global, & kalibrasi threshold finansial
├── src/
│   ├── transformers.py                    <- Custom Scikit-Learn transformer (TelcoFeatureEngineer)
│   ├── services/
│   │   ├── agent_service.py               <- PydanticAI Agent Orchestrator (Multi-Provider ladder & failover)
│   │   ├── agent_tools.py                 <- Deterministic tools (What-if ML simulation & ROI calculation)
│   │   ├── fallback_engine.py             <- Zero-downtime offline heuristic retention engine
│   │   ├── inference_service.py           <- LightGBM pipeline serving & batch inference
│   │   ├── schemas.py                     <- Validasi skema input & output Pydantic v2 dengan guardrails
│   │   └── shap_service.py                <- TreeSHAP explainer (lokal waterfall & narasi bisnis)
│   └── ui/
│       └── common.py                      <- Komponen UI bersama, status ML badges, & API key configurator
├── tests/
│   ├── benchmark_personas.json            <- Kumpulan persona pelanggan sintetis untuk benchmarking
│   ├── test_agent_harness.py              <- Pengujian guardrails anggaran, personas, & offline fallback
│   ├── test_agent_tools.py                <- Pengujian deterministik alat inferensi & kalkulasi ROI
│   ├── test_inference_service.py          <- Pengujian validasi skema Pydantic & prediksi pipeline
│   ├── test_shap_service.py               <- Pengujian TreeSHAP waterfall & ekstraksi faktor risiko
│   └── test_streamlit_app.py              <- Pengujian otomasi UI via Streamlit AppTest (Navigation & UX)
├── docs/
│   ├── methodologi.md                     <- Dokumentasi teknis mendalam (arsitektur, data leakage audit)
│   ├── phase2_system_design_and_architecture.md <- Desain arsitektur sistem Fase 2
│   ├── agentic_ai_implementation_plan.md  <- Rencana implementasi & evaluasi PydanticAI
│   └── agentic_rag_upgrade_plan.md        <- Roadmap upgrade Agentic RAG
├── models/
│   └── telco_churn_lgbm_pipeline.joblib   <- Artefak model pipeline terenkapsulasi siap deployment
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv <- Dataset mentah lokal IBM Telco
├── notebooks/
│   ├── telco_churn_advanced_pipeline.ipynb <- Notebook utama (Custom Pipeline, Optuna, Profit Threshold, SHAP)
│   └── telco_churn_pycaret_baseline.ipynb  <- Notebook eksperimen AutoML (PyCaret Benchmark & Tuning)
├── pyproject.toml                         <- Konfigurasi dependensi modern (UV / PEP 621)
├── requirements.txt                       <- Daftar dependensi pip kompatibel
└── README.md                              <- Dokumentasi utama proyek
```

---

## 7. Penggunaan Model untuk Inferensi (Serving)
Model tersimpan sebagai pipeline utuh yang siap menerima payload dictionary/JSON mentah:

```python
import joblib
import pandas as pd

# Muat artefak
pack = joblib.load('models/telco_churn_lgbm_pipeline.joblib')
pipeline = pack['pipeline']
threshold = pack['optimal_threshold']

# Payload pelanggan baru
new_customer = pd.DataFrame([{
    'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
    'tenure': 2, 'PhoneService': 'Yes', 'MultipleLines': 'No',
    'InternetService': 'Fiber optic', 'OnlineSecurity': 'No', 'OnlineBackup': 'No',
    'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'Yes',
    'StreamingMovies': 'No', 'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check', 'MonthlyCharges': 85.50, 'TotalCharges': 171.00
}])

# Prediksi risiko
prob = pipeline.predict_proba(new_customer)[:, 1][0]
is_churn_risk = prob >= threshold

print(f"Probabilitas Churn: {prob*100:.1f}%")
print(f"Target Retensi: {'Ya' if is_churn_risk else 'Tidak'}")
```

---

## 8. Phase 2: Agentic AI Intelligent Retention Platform

Fase 2 mentransformasi pipeline prediktif menjadi platform retensi presisi yang interaktif dengan *business guardrails*, *multi-provider resilience*, dan *zero-downtime fallback*:

### Fitur Utama:
1. **Interactive Single Customer Profiler:** Visualisasi gauge probabilitas vs ambang batas $\tau^* = 0.39$ secara real-time.
2. **Explainable AI (TreeSHAP):** Penguraian faktor lokal pendorong risiko (*risk drivers*) dan penahan (*retention anchors*) ke dalam narasi bisnis manusiawi.
3. **What-If Counterfactual Sandbox:** Eksperimen langsung perubahan fitur (misal: migrasi kontrak 1 tahun atau penambahan proteksi) dengan estimasi penurunan risiko instan.
4. **Agentic AI Retention Copilot (PydanticAI):** Orkestrasi LLM multi-tier dengan *deterministic tool-calling loop* untuk merumuskan penawaran retensi presisi:
   - **Primary Engine:** Google Gemini aktif (`gemini-3.5-flash-lite`, `gemini-3.5-flash`).
   - **Secondary Failover:** OpenRouter API (`openrouter/free`).
   - **Guardrails Ketat:** Pagu biaya penawaran wajib $\le \$20.00$ per pelanggan.
5. **Zero-Downtime Deterministic Fallback:** `HeuristicRetentionEngine` lokal otomatis mengambil alih secara transparan saat API limit (429/503) atau ketiadaan API key.
6. **In-App API Key Configurator:** Pengaturan API Key fleksibel melalui `.env`, Streamlit Cloud Secrets (`st.secrets`), maupun input langsung via sidebar pada sesi browser.
7. **Batch Work Queue Table:** Analisis batch CSV pelanggan dan pengurutan prioritas otomatis berdasarkan *Expected Loss* ($P(\text{churn}) \times \text{CLV}$).
8. **Executive Governance & Calibration:** Evaluasi kurva ROC-AUC holdout, kalibrasi ambang batas finansial, dan audit *bias-variance*.

### Menjalankan Aplikasi Web (Streamlit):
```bash
# Menggunakan uv (Rekomendasi - Cepat & Terisolasi)
uv run streamlit run app.py

# Atau menggunakan pip / virtual environment aktif
.\.venv\Scripts\activate
streamlit run app.py
```

### Konfigurasi API Key (Opsional untuk AI Copilot):
Aplikasi tetap berjalan normal (100% fungsional) menggunakan Heuristic Engine tanpa API Key. Untuk mengaktifkan mode AI Copilot:
- **Lokal:** Tambahkan `GEMINI_API_KEY="kunci_anda"` pada file `.env`.
- **Streamlit Community Cloud:** Tambahkan `GEMINI_API_KEY = "kunci_anda"` pada menu **App Settings > Secrets**.
- **Via Antarmuka Web:** Masukkan kunci pada expander **"🔑 Konfigurasi API Key LLM"** di sidebar.

### Menjalankan Automated Test Harness:
```bash
# Menjalankan seluruh test suite (21 automated test cases)
uv run pytest -v
```

