# Rencana Upgrade Proyek: Agentic RAG untuk Telco Churn Intelligent Retention Platform

## 1. Ringkasan Eksekutif
Proyek ini bertransformasi dari sekadar **Sistem Prediksi *Churn* (Predictive ML)** menjadi **Platform Retensi Cerdas (Decision Support System)**. 
Upgrade utama melibatkan pengintegrasian teknologi **Agentic Retrieval-Augmented Generation (RAG)**. Tujuan utamanya adalah menjembatani celah antara hasil prediksi (probabilitas pelanggan akan *churn*) dengan aksi operasional, dengan cara membekali AI Copilot dengan pengetahuan kontekstual perusahaan (SOP, Katalog Promo, Skrip Komunikasi) untuk menghasilkan rekomendasi retensi yang sangat personal, *actionable*, dan sesuai standar operasi perusahaan.

## 2. Latar Belakang Masalah & Solusi
*   **Masalah Saat Ini:** Model *Machine Learning* (XGBoost/LightGBM) dapat memprediksi *churn* dengan akurat, namun staf *Customer Service* (CS) masih harus menebak-nebak tindakan mitigasi atau promo apa yang paling tepat untuk pelanggan tersebut.
*   **Solusi Agentic RAG:** LLM (Gemini/OpenAI) bertindak sebagai *Copilot* yang tidak hanya membaca profil pelanggan dan alasan *churn* (SHAP values), tetapi juga **secara dinamis mencari (Retrieve)** SOP perusahaan dan katalog promo terkini, lalu **menghasilkan (Generate)** rencana retensi langkah-demi-langkah yang spesifik untuk pelanggan tersebut.

## 3. Arsitektur Sistem Agentic RAG

Sistem ini menggunakan arsitektur **Hybrid Agentic Workflow**, yang menggabungkan kecerdasan generatif dengan fallback heuristik deterministik untuk keandalan tingkat produksi (*zero-downtime*).

*   **Knowledge Base (Vector/Document Store):** Penyimpanan dokumen fiktif perusahaan dalam format Markdown/Teks (SOP CS, Katalog Paket, FAQ).
*   **Retriever:** Modul yang bertugas melakukan pencarian dokumen yang relevan berdasarkan profil pelanggan (misalnya: mencari promo khusus pelanggan prabayar jika pelanggan tersebut adalah prabayar).
*   **LLM Copilot (Agent):** Menggunakan model LLM (Gemini 3.1 Pro/GPT-4o) dengan kemampuan *Function Calling* (Tool Use) untuk merangkai profil data terstruktur dengan pengetahuan tak terstruktur (dokumen).
*   **Heuristic Fallback Engine:** Jika API LLM mengalami *timeout*, *rate-limit*, atau kegagalan jaringan, sistem secara otomatis beralih ke mesin aturan statis (If-Else / Mapping berbasis *threshold*) sehingga UI/UX tidak pernah terganggu.

## 4. Tech Stack & Tools (Standar 2026)

Upgrade ini mengimplementasikan *tech stack* modern yang sangat relevan dengan kebutuhan industri saat ini:

| Kategori | Teknologi / Library | Peran dalam Proyek |
| :--- | :--- | :--- |
| **Generative AI & LLM** | `openai` (OpenRouter API), Gemini 3.1 Pro | *Engine* utama untuk penalaran (Reasoning) dan generasi teks respons Copilot. |
| **Agentic Framework** | *Custom Python Agent* dengan *Function Calling* | Mengorkestrasi pemanggilan model, injeksi *prompt*, dan mekanisme *fallback* tanpa bergantung pada *framework* gemuk (langchain-free) demi performa. |
| **Predictive ML** | `pycaret`, `scikit-learn`, `shap` | *Pipeline* latih model otomatis, pemrosesan data, dan *Explainable AI* (XAI). |
| **Frontend / UI** | `streamlit` (dengan fitur modern 2026: `@st.fragment`, `st.dialog`) | Membangun antarmuka web interaktif, responsif, dan bernuansa premium. |
| **Testing & QA** | `pytest`, `st.testing.v1.AppTest` | Pengujian unit otomatis (TDD) untuk memvalidasi alur UI Streamlit dan keandalan *backend* LLM. |
| **Dependency Management**| `uv` (atau `poetry`) | Resolusi dan instalasi paket Python ultra-cepat dan deterministik. |

## 5. Keterampilan Utama yang Didemonstrasikan (Nilai Portofolio)

Proyek ini dirancang untuk menunjukkan bahwa kandidat adalah seorang **Full-Stack AI Engineer** yang memahami siklus hidup AI secara menyeluruh, tidak hanya pada fase eksplorasi data:

1.  **AI System Design & Architecture:** Mampu merancang sistem hibrida (*LLM + Heuristik*) yang *fault-tolerant* dan siap masuk ke lingkungan produksi.
2.  **Prompt Engineering & Context Management (RAG):** Keahlian meramu sistem RAG untuk mengurangi halusinasi LLM (*grounding*) menggunakan dokumen lokal/perusahaan.
3.  **Modern UI/UX Development:** Menguasai paradigma pengembangan web reaktif di Python (Streamlit) dengan penekanan pada estetika (*Material Symbols*, tata letak dinamis) dan performa manajemen *state* (`st.session_state`).
4.  **Software Engineering Best Practices:** Menerapkan arsitektur kode termodularisasi (pemisahan logika dan UI), *version control* (Git), dan *Test-Driven Development* (TDD) menggunakan *monkeypatching* untuk simulasi respons API eksternal.

## 6. Roadmap Implementasi (Tahap Selanjutnya)

- [ ] **Fase 1: Persiapan Knowledge Base**
  - Membuat direktori `data/knowledge_base/`.
  - Menyusun dokumen Markdown fiktif (SOP Retensi Telco, Katalog Produk/Promo, Panduan Tone-of-Voice CS).
- [ ] **Fase 2: Integrasi RAG ke AgentService**
  - Menulis modul pembaca dokumen (`document_loader`).
  - Mengupdate *system prompt* di `agent_service.py` untuk menerima injeksi teks konteks (*Context Injection*).
- [ ] **Fase 3: Pengujian & Validasi**
  - Membuat skenario pengujian di mana Copilot merekomendasikan promo spesifik yang HANYA ADA di dokumen *knowledge base* (pembuktian bahwa RAG berfungsi dan tidak halusinasi).
- [ ] **Fase 4: Finalisasi UI & Dokumentasi**
  - Memastikan *dashboard* menampilkan sumber dokumen (sitasi) di UI Copilot untuk meningkatkan kepercayaan pengguna (CS).
