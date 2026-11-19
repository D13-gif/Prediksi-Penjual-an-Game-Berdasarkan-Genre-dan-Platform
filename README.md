# Penerapan Algoritma Decision Tree untuk Klasifikasi Tingkat Penjualan Video Game Berdasarkan Genre dan Platform

**Pemilik:** Adi Hadianto (NIM 301240006, Kelas 5B)
**Mata praktikum:** Pembelajaran Mesin — Modul 05 (Decision Tree)
**Algoritma:** `DecisionTreeClassifier` (Gini) dalam satu sklearn `Pipeline`
**Dataset:** [Kaggle - Video Game Sales (VGChartz, gregorut)](https://www.kaggle.com/datasets/gregorut/videogamesales)

## Ringkasan
Aplikasi web Flask mengklasifikasikan `Sales_Category` (**Rendah / Sedang / Tinggi**) dari fitur **Genre, Platform, Year, Publisher**. `Global_Sales` hanya dipakai untuk membentuk target; **Global/NA/EU/JP/Other Sales tidak digunakan sebagai fitur** (pencegahan data leakage). Aplikasi memuat **pipeline tersimpan** (`decision_tree_pipeline.joblib`), bukan melatih ulang saat halaman dibuka.

**Aturan target (kategori turunan, bukan label asli Kaggle):**
- **Rendah:** `Global_Sales < 0.5` juta kopi
- **Sedang:** `0.5 ≤ Global_Sales ≤ 2.0` juta kopi
- **Tinggi:** `Global_Sales > 2.0` juta kopi

## Struktur Proyek
```
.
├── main.py                      # Entry Flask (load app)
├── train_model.py               # Training: audit→target→pipeline→split→baseline→pruning→eval→save
├── requirements.txt             # Runtime dependencies (pinned for saved model)
├── requirements-training.txt    # Extra plotting dependencies for retraining
├── README.md
├── vgsales.csv                  # Dataset lokal (sumber Kaggle)
├── src/
│   ├── data_loader.py           # Load + audit (shape/info/head/missing/duplikat)
│   ├── preprocessing.py         # Target, cleaning, ColumnTransformer+Pipeline (fit on train)
│   └── evaluation.py            # Metrik (acc/P/R/F1 macro+weighted), CM, report, overfit check
├── app/
│   ├── __init__.py              # create_app + template filters
│   ├── routes.py                # 6 halaman + /api/health
│   ├── services.py              # Load pipeline/metadata, validasi, prediksi + probabilitas
│   ├── templates/               # base, dashboard, predict, dataset, evaluation, tree, about, error
│   └── static/css, js, images/
├── data/models/                 # decision_tree_pipeline.joblib, model_metadata.json, decision_tree_viz.png
├── vercel.json                  # Include templates, static files, dataset, model artifacts
├── docs/screenshots/            # Tempat screenshot NYATA (lihat petunjuk di folder)
├── notebooks/                   # EDA script/figures (aktual)
└── archive/                     # Artefak legacy (tidak dipakai)
```

## Instalasi & Cara Menjalankan
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-training.txt

# (opsional) latih ulang dari data lokal — menghasilkan artefak + metrik aktual
python train_model.py

# jalankan aplikasi (memuat pipeline tersimpan)
python main.py
```
Buka **http://127.0.0.1:5000**.

## Deploy ke Vercel

Aplikasi memakai entry point Flask `main.py` (`app` WSGI) dan runtime Python 3.14. Dependensi runtime dipin agar konsisten dengan versi scikit-learn yang digunakan untuk menyimpan pipeline. `vercel.json` menyertakan template, aset statis, dataset, dan artefak model ke function. Visualisasi pohon dibaca langsung dari artefak PNG sehingga tidak perlu menulis ke filesystem serverless yang read-only.

### Deploy otomatis dari GitHub
1. Push repository ini ke GitHub.
2. Masuk ke [Vercel](https://vercel.com/new), pilih **Add New → Project**, lalu import `D13-gif/Prediksi-Penjual-an-Game-Berdasarkan-Genre-dan-Platform`.
3. Biarkan Framework Preset terdeteksi sebagai Flask/Python. Root Directory gunakan `./`; Build Command dan Output Directory biarkan kosong/default.
4. Klik **Deploy**. Tidak perlu environment variable untuk aplikasi saat ini; opsional set `SECRET_KEY` pada Project Settings → Environment Variables agar stabil antar cold start.
5. Setelah deploy, verifikasi `/api/health` mengembalikan `status: "ok"` dan buka halaman Dashboard, Prediction, serta Decision Tree.

### Deploy lewat CLI
```powershell
npx vercel login
npx vercel link
npx vercel
npx vercel --prod
```

Jalankan `npx vercel` dari root proyek. Pastikan dataset `vgsales.csv`, `data/models/decision_tree_pipeline.joblib`, `data/models/model_metadata.json`, dan visualisasi pohon ikut ter-commit. Vercel Functions bersifat serverless dan filesystem-nya tidak untuk penyimpanan persisten; aplikasi hanya membaca dataset/model dan tidak menyimpan unggahan pengguna. Jika perubahan versi training dilakukan, latih ulang model lalu deploy ulang dengan requirements runtime yang selaras.

## Halaman Aplikasi
1. **Dashboard** — KPI (data/genre/platform + metrik model aktual), distribusi target & grafik ringkasan.
2. **Prediction** — input wajib (Platform, Genre, Publisher, Year), hasil kategori + badge + probabilitas tiap kelas + disclaimer estimasi; validasi & error handling.
3. **Dataset Exploration** — audit, ringkasan missing/duplikat, 6 visualisasi EDA (target, genre, platform, tahun, kategori vs genre/platform) + interpretasi aktual, tabel paginasi.
4. **Model Evaluation** — baseline vs final, metrik (acc/P/R/F1 macro & weighted), confusion matrix, classification report, tabel eksperimen pruning.
5. **Decision Tree Visualization** — pohon (depth tampil dibatasi) + zoom/unduh + feature importance + catatan.
6. **About Project** — identitas, dataset/sumber, fitur/target/ambang, anti-leakage, metodologi, batasan.

## Metodologi & Keputusan Teknis
- **EDA & audit:** load `vgsales.csv`, `head/shape`, missing, duplikasi, distribusi & perbandingan kategori (aktual).
- **Preprocessing (konsisten, anti-leakage):** `Year` imputasi median, `Publisher` imputasi konstanta `Unknown`, `OneHotEncoder(handle_unknown="ignore")`, `ColumnTransformer`+`Pipeline`. Encoder **di-fit pada data training saja** (setelah split). Input kategori baru aman (tidak crash).
- **Fitur:** `Platform, Genre, Publisher, Year`. Dikecualikan: `Global_Sales` (hanya target), `NA/EU/JP/Other_Sales` (komponen penjualan → leakage), `Rank`, `Name`.
- **Split:** 80:20, `random_state=42`, `stratify=y`.
- **Model & pruning:** baseline Gini (unpruned) vs kandidat pruning (`max_depth`, `min_samples_split`, `min_samples_leaf`, `ccp_alpha`) dibandingkan pada train vs test; model final dipilih deterministik untuk keseimbangan performa/generalisasi/keterbacaan pada fitur leakage-free.
- **Evaluasi:** accuracy, precision, recall, F1 **macro & weighted** (kelas imbalanced), confusion matrix, classification report.

## Hasil Evaluasi (AKTUAL — dari `data/models/model_metadata.json`)
Dataset: **16.598 baris, 11 kolom, duplikat 0**. Missing: Year 271, Publisher 58 (ditangani pipeline). Target (turunan): Rendah 12.580 (75,79%), Sedang 3.172 (19,11%), Tinggi 846 (5,10%). Split: train 13.278 / test 3.320 (stratified).

| Model | Test Acc | Prec (w) | Recall (w) | F1 (w) | F1 (macro) | Train Acc | Gap Acc | Overfit |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline (unpruned) | 0.7380 | 0.7106 | 0.7380 | 0.7219 | 0.4626 | 0.9069 | 0.1690 | Ya |
| **Final (pruned, `depth_8`)** `max_depth=8, min_samples_split=2, min_samples_leaf=1, ccp_alpha=0.0` | **0.7623** | **0.6957** | **0.7623** | **0.7057** | **0.3854** | 0.7816 | 0.0192 | Ya |

Detail (confusion matrix, classification report, seluruh eksperimen pruning) tersimpan di `data/models/model_metadata.json` dan ditampilkan di halaman **Model Evaluation**. Angka dapat berubah jika dilatih ulang.

## Screenshot (bukti nyata dari aplikasi yang berjalan)
> Aturan proyek: **dilarang screenshot/grafik/hasil palsu**. Screenshot berikut diambil dari aplikasi lokal yang berjalan dan disimpan pada `docs/screenshots/`:
- `docs/screenshots/dashboard.png`
- `docs/screenshots/prediction.png` (sertakan hasil + probabilitas + disclaimer)
- `docs/screenshots/dataset.png`
- `docs/screenshots/evaluation.png`
- `docs/screenshots/tree.png`
- `docs/screenshots/about.png`
- `docs/screenshots/dashboard-mobile.png` (uji tampilan ponsel)

## Batasan & Kejujuran Akademik
- Hasil adalah **estimasi model** berbasis data historis, **bukan** jaminan keberhasilan komersial.
- Diskretisasi `Global_Sales` mengurangi informasi; kelas **imbalanced** (Tinggi minoritas) dan fitur terbatas tanpa sinyal penjualan → performa kelas minoritas terbatas (lihat macro-F1 & confusion matrix). Model **tidak diklaim “terbaik”** tanpa bukti.
- Tidak ada metrik/screenshot/hasil pengujian/deployment palsu.

## Lisensi & Sumber Data
Data: Kaggle VGChartz (gregorut). Periksa ketentuan penggunaan/redistribusi pada [halaman sumber](https://www.kaggle.com/datasets/gregorut/videogamesales) sebelum mempublikasikan dataset mentah.

## Identitas
Adi Hadianto — NIM 301240006 — Kelas 5B.
