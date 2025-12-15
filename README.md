# KRR-XAI: Analisis Sentimen dengan Explainability

Proyek ini mengimplementasikan model analisis sentimen menggunakan RoBERTa dan menyediakan fitur explainability menggunakan LIME, SHAP, dan Integrated Gradients (IG). Proyek ini memungkinkan pengguna untuk tidak hanya mengklasifikasikan sentimen teks tetapi juga memahami *mengapa* model membuat prediksi tertentu.

## Fitur

*   **Klasifikasi Sentimen yang Robust**: Model RoBERTa yang telah di-finetune untuk analisis sentimen dengan akurasi tinggi.
*   **Explainable AI (XAI)**:
    *   **LIME** (Local Interpretable Model-agnostic Explanations): Penjelasan berbasis perturbasi.
    *   **SHAP** (SHapley Additive exPlanations): Kepentingan fitur berbasis teori permainan.
    *   **IG** (Integrated Gradients): Metode atribusi berbasis gradien dengan integrasi jalur untuk stabilitas yang lebih baik.
*   **Antarmuka CLI**: Antarmuka command-line yang mudah digunakan untuk menganalisis teks.
*   **Mode Interaktif**: Loop analisis berkelanjutan untuk menguji beberapa input.

## Instalasi

Proyek ini menggunakan `pyproject.toml` untuk manajemen dependensi.

### Metode 1: Menggunakan pip

```bash
pip install -e .
```

### Metode 2: Menggunakan uv (Direkomendasikan untuk kecepatan)

Jika Anda telah menginstal [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

## Cara Penggunaan

### 1. Unduh Model yang Sudah Dilatih

Sebelum menggunakan CLI, Anda perlu mengunduh model RoBERTa yang sudah dilatih:

1.  Unduh file model dari [Google Drive](https://drive.google.com/drive/folders/1uqECfzIyH-bQaLbTNPogf-TxYBo_ej09?usp=sharing)
2.  Buat folder bernama `best_roberta_model` di direktori root proyek
3.  Ekstrak semua file model yang diunduh ke dalam folder `best_roberta_model`

Struktur folder harus terlihat seperti ini:
```
KRR-XAI/
├── best_roberta_model/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   ├── vocab.json
│   ├── merges.txt
│   ├── special_tokens_map.json
│   └── label_mappings.json
├── sentiment_cli.py
└── ...
```

### 2. (Opsional) Melatih Model Sendiri

Jika Anda ingin melatih model dari awal:

1.  Pastikan Anda memiliki dataset di `dataset/train.csv`.
2.  Jalankan notebook pelatihan `train_roberta_optimized.ipynb`.
3.  Ini akan menyimpan model yang telah di-finetune ke `./best_roberta_model`.

### 3. Command-Line Interface (CLI)

Gunakan `sentiment_cli.py` untuk menganalisis teks.

**Penggunaan Dasar:**

```bash
python sentiment_cli.py "This movie is absolutely fantastic!"
```

**Tentukan Metode Explainability:**

Evaluasi menggunakan metode tertentu (default adalah `all`):

```bash
python sentiment_cli.py "The service was terrible but the food was okay." --method lime
python sentiment_cli.py "The service was terrible but the food was okay." --method shap --top 5
python sentiment_cli.py "Amazing product!" --method ig
```

Metode yang tersedia: `lime`, `shap`, `ig`, `all`

**Mode Interaktif:**

```bash
python sentiment_cli.py --interactive
```

## Struktur Proyek

*   `sentiment_cli.py`: Tool CLI utama untuk inferensi dan penjelasan.
*   `train_roberta_optimized.ipynb`: Notebook untuk melatih model RoBERTa.
*   `XAI_Sentiment_Analysis_lib.ipynb` & `_scratch.ipynb`: Notebook penelitian dan eksperimen.
*   `pyproject.toml`: Dependensi dan konfigurasi proyek.

## Persyaratan

*   Python >= 3.9
*   Torch >= 2.0.0
*   Transformers >= 4.30.0
