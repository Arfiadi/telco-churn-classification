# Telco Customer Churn: Predictive Pipeline & Profit-Driven Decisioning

> **Mata Kuliah:** Machine Learning (Semester 3) — Proyek Review & Penerapan Machine Learning dengan Data IBM Telco Customer Churn.
> **Dokumentasi Lengkap:** Detail arsitektur data science, kamus variabel, audit pencegahan *data leakage*, dan evaluasi teknis tersedia di [docs/methodologi.md](docs/methodologi.md).

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
├── README.md                              <- Dokumentasi utama proyek
├── requirements.txt                       <- Daftar dependensi Python proyek
├── .gitignore                             <- Konfigurasi git ignore (termasuk draf notebook)
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv <- Dataset mentah lokal
├── docs/
│   └── methodologi.md                     <- Dokumentasi teknis mendalam (arsitektur, data leakage audit)
├── models/
│   └── telco_churn_lgbm_pipeline.joblib   <- Artefak model pipeline terenkapsulasi siap deployment
└── notebooks/
    ├── telco_churn_advanced_pipeline.ipynb <- Notebook utama (Custom Pipeline, Optuna, Profit Threshold, SHAP)
    └── telco_churn_pycaret_baseline.ipynb  <- Notebook eksperimen AutoML (PyCaret Benchmark & Tuning)
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

Fase 2 mentransformasi pipeline prediktif menjadi platform retensi presisi yang interaktif dengan *business guardrails* dan *zero-downtime fallback*:

### Fitur Utama:
1. **Interactive Single Customer Profiler:** Visualisasi gauge probabilitas vs ambang batas $\tau^* = 0.39$ secara real-time.
2. **Explainable AI (TreeSHAP):** Penguraian faktor lokal pendorong risiko (*risk drivers*) dan penahan (*retention anchors*) ke dalam narasi bisnis manusiawi.
3. **What-If Counterfactual Sandbox:** Eksperimen langsung perubahan fitur (misal: migrasi kontrak 1 tahun atau penambahan proteksi) dengan estimasi penurunan risiko instan.
4. **Agentic AI Retention Copilot:** Orkestrasi LLM via OpenRouter dengan *deterministic tool-calling loop* untuk merumuskan penawaran retensi presisi, batasan pagu biaya ($\le \$20.00$), dan naskah komunikasi empati.
5. **Zero-Downtime Fallback:** `HeuristicRetentionEngine` lokal otomatis mengambil alih secara transparan saat API rate limit (429) atau koneksi offline.
6. **Batch Work Queue Table:** Analisis batch CSV pelanggan dan pengurutan prioritas otomatis berdasarkan *Expected Loss* ($P(\text{churn}) \times \text{CLV}$).

### Menjalankan Aplikasi Web (Streamlit):
```bash
# 1. Pastikan virtual environment aktif
.\.venv\Scripts\activate

# 2. Jalankan dashboard Streamlit
streamlit run app.py
```

### Menjalankan Automated Test Harness:
```bash
# Jalankan seluruh test suite (14 test cases)
pytest -v
```

