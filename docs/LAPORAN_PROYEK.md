# Laporan Proyek

## Penerapan Algoritma Decision Tree untuk Klasifikasi Tingkat Penjualan Video Game Berdasarkan Genre dan Platform

**Nama:** Adi Hadianto  
**NIM:** 301240006  
**Kelas:** 5B  
**Mata Praktikum:** Pembelajaran Mesin — Modul 05 (Decision Tree)  
**Sumber Dataset:** Kaggle — Video Game Sales (VGChartz, gregorut)  
**Jenis Aplikasi:** Dashboard analitik dan aplikasi prediksi berbasis web

> Dokumen ini merupakan draf bahan laporan berdasarkan source code, dataset, metadata model, pengujian, dan hasil training yang tersimpan di proyek. Sesuaikan format halaman, gaya sitasi, dan ketentuan institusi sebelum dikumpulkan.

## Abstrak

Proyek ini membangun aplikasi web untuk mengklasifikasikan tingkat penjualan global video game ke dalam tiga kategori, yaitu Rendah, Sedang, dan Tinggi. Dataset yang digunakan adalah Video Game Sales dari Kaggle yang berisi 16.598 baris dan 11 kolom sumber. Kategori target `Sales_Category` dibentuk dari `Global_Sales` menggunakan batas kurang dari 0,5 juta kopi untuk Rendah, 0,5 sampai dengan 2,0 juta kopi untuk Sedang, dan lebih dari 2,0 juta kopi untuk Tinggi. Prediksi menggunakan fitur `Platform`, `Genre`, `Publisher`, dan `Year`. Kolom penjualan global maupun regional sengaja tidak digunakan sebagai fitur untuk menghindari data leakage. Preprocessing, yang mencakup imputasi dan One-Hot Encoding, ditempatkan dalam satu pipeline bersama `DecisionTreeClassifier` dengan kriteria Gini. Data dibagi secara stratified menjadi 80% training dan 20% testing. Model final memiliki test accuracy 0,7623 dan macro-F1 0,3854. Walaupun accuracy cukup tinggi, recall untuk kelas Tinggi hanya 0,0296; karena itu hasil perlu ditafsirkan dengan memperhatikan ketidakseimbangan kelas. Aplikasi Flask menyediakan dashboard, eksplorasi dataset, evaluasi model, visualisasi pohon, dan form prediksi.

**Kata kunci:** klasifikasi, Decision Tree, Gini, penjualan video game, Flask, data leakage.

## BAB I — Pendahuluan

### 1.1 Latar Belakang

Penjualan video game dapat berbeda menurut platform, genre, publisher, dan waktu rilis. Dataset VGChartz menyimpan informasi penjualan historis beserta karakteristik game. Proyek ini menggunakan karakteristik tersebut untuk membangun model klasifikasi yang mengestimasi kelompok penjualan global.

Tujuan aplikasi bukan menjamin penjualan sebuah game di masa depan. Model belajar pola dari data historis, sementara banyak faktor lain seperti pemasaran, kualitas produk, franchise, wilayah, tren, dan kondisi pasar tidak tercakup sebagai fitur.

### 1.2 Rumusan Masalah

1. Bagaimana mengubah nilai penjualan global menjadi kelas yang dapat dipelajari oleh algoritma klasifikasi?
2. Bagaimana memprediksi kelas penjualan menggunakan karakteristik game tanpa memasukkan informasi penjualan yang dapat menyebabkan kebocoran target?
3. Bagaimana menampilkan dataset, proses evaluasi, struktur model, dan hasil prediksi melalui antarmuka web?

### 1.3 Tujuan

1. Menyusun target turunan `Sales_Category` dari `Global_Sales`.
2. Melatih dan mengevaluasi `DecisionTreeClassifier` menggunakan preprocessing yang konsisten.
3. Menyimpan pipeline model dan metadata evaluasi agar dapat digunakan aplikasi tanpa melatih ulang pada setiap request.
4. Menyediakan antarmuka untuk dashboard, EDA, evaluasi, visualisasi pohon, dan prediksi interaktif.

### 1.4 Batasan

- Dataset berasal dari Kaggle VGChartz dan disimpan sebagai `vgsales.csv`.
- Kelas target hanya terdiri dari Rendah, Sedang, dan Tinggi sesuai ambang proyek.
- Fitur prediksi adalah Platform, Genre, Publisher, dan Year.
- Model yang digunakan adalah Decision Tree Classifier dengan kriteria Gini.
- Prediksi merupakan estimasi dari pola historis, bukan hubungan kausal atau jaminan penjualan.

## BAB II — Dataset dan Analisis Data

### 2.1 Sumber dan Isi Dataset

Dataset yang digunakan adalah [Video Game Sales — VGChartz](https://www.kaggle.com/datasets/gregorut/videogamesales). Dataset aktual pada proyek memiliki:

- 16.598 baris dan 11 kolom sumber.
- Rentang tahun rilis teramati 1980–2020.
- 0 duplikat baris.
- 271 nilai Year kosong dan 58 nilai Publisher kosong.
- Tidak terdapat nilai kosong pada `Global_Sales` dalam dataset yang diaudit.

Kolom sumber mencakup `Rank`, `Name`, `Platform`, `Year`, `Genre`, `Publisher`, `NA_Sales`, `EU_Sales`, `JP_Sales`, `Other_Sales`, dan `Global_Sales`.

### 2.2 Pembentukan Target

Kolom `Sales_Category` bukan label bawaan dataset; kolom ini dibuat oleh proyek berdasarkan `Global_Sales`, dengan satuan juta kopi:

| Kategori | Aturan |
|---|---|
| Rendah | `Global_Sales < 0.5` |
| Sedang | `0.5 ≤ Global_Sales ≤ 2.0` |
| Tinggi | `Global_Sales > 2.0` |

Nilai tepat 0,5 dan 2,0 termasuk kategori Sedang. Pada audit awal, implementasi memasukkan nilai tepat 0,5 ke Rendah. Kesalahan tersebut telah diperbaiki; terdapat 71 baris dengan nilai tepat 0,5 yang sekarang dilabeli Sedang. Distribusi final setelah koreksi:

| Kategori | Jumlah | Persentase |
|---|---:|---:|
| Rendah | 12.580 | 75,79% |
| Sedang | 3.172 | 19,11% |
| Tinggi | 846 | 5,10% |
| **Total** | **16.598** | **100%** |

Distribusi tersebut menunjukkan ketidakseimbangan kelas, terutama karena jumlah kelas Tinggi jauh lebih sedikit daripada kelas Rendah.

### 2.3 Fitur dan Pencegahan Data Leakage

Fitur yang digunakan:

- `Platform`
- `Genre`
- `Publisher`
- `Year`

`Global_Sales` hanya dipakai untuk membentuk label target dan tidak masuk sebagai fitur model. `NA_Sales`, `EU_Sales`, `JP_Sales`, dan `Other_Sales` juga dikeluarkan karena merupakan komponen penjualan yang berkaitan langsung dengan total penjualan. `Rank` dan `Name` tidak dipakai sebagai fitur.

Pemisahan ini mencegah model menggunakan informasi hasil penjualan untuk memprediksi kelas yang dibentuk dari hasil penjualan itu sendiri.

## BAB III — Metodologi

### 3.1 Alur Pengolahan Data

1. Membaca dan mengaudit `vgsales.csv`.
2. Membentuk target `Sales_Category`.
3. Mengubah `Year` menjadi nilai numerik dan mempertahankan missing value untuk ditangani di pipeline.
4. Memisahkan fitur dan target.
5. Membagi data menjadi training dan testing secara stratified.
6. Melatih baseline Decision Tree serta beberapa konfigurasi pembatasan kompleksitas pohon.
7. Mengukur metrik training dan testing.
8. Menyimpan pipeline final, metadata metrik, matriks konfusi, laporan klasifikasi, tabel eksperimen, dan gambar pohon.
9. Memuat artefak tersimpan dari aplikasi Flask untuk melayani prediksi.

### 3.2 Pembagian Data

Pembagian data menggunakan `train_test_split` dengan:

- 80% data training: 13.278 baris.
- 20% data testing: 3.320 baris.
- `random_state=42` agar hasil pembagian dapat direproduksi.
- `stratify=y` agar proporsi kelas pada data train dan test tetap mendekati distribusi keseluruhan.

### 3.3 Preprocessing

Preprocessing dimasukkan ke dalam `sklearn.Pipeline` dan `ColumnTransformer`, sehingga proses yang sama digunakan saat training dan prediksi:

- Kolom kategori Platform, Genre, dan Publisher: missing Publisher diisi dengan konstanta `Unknown`, lalu dikonversi menggunakan One-Hot Encoding.
- `OneHotEncoder(handle_unknown="ignore")` memungkinkan kategori baru diproses tanpa menyebabkan error karena kategori tersebut tidak dikenal saat training.
- Kolom Year diisi menggunakan median yang dipelajari dari data training.
- Imputer dan encoder di-fit ketika pipeline dilatih dengan data training, bukan seluruh data sebelum pembagian.

### 3.4 Algoritma Decision Tree

Model utama adalah `DecisionTreeClassifier(criterion="gini", random_state=42)`. Gini impurity mengukur ketidakmurnian kelas dalam sebuah node:

`Gini = 1 − Σ(pᵢ²)`

di mana `pᵢ` adalah proporsi kelas ke-i pada node. Pemisahan dipilih berdasarkan penurunan impurity tertimbang. Pohon membentuk aturan keputusan berdasarkan nilai fitur hasil preprocessing.

Baseline menggunakan pohon tanpa batas kedalaman. Eksperimen juga menguji nilai `max_depth`, `min_samples_split`, `min_samples_leaf`, dan `ccp_alpha`. Model final yang tersimpan adalah konfigurasi:

```text
max_depth=8
min_samples_split=2
min_samples_leaf=1
ccp_alpha=0.0
```

Catatan metodologis: konfigurasi final dibatasi menggunakan `max_depth=8`. Karena `ccp_alpha=0.0`, final ini tidak menggunakan cost-complexity pruning aktif; lebih akurat menyebutnya pohon yang dibatasi kedalamannya (pre-pruning). Kode training menetapkan `depth_8` sebagai pilihan final secara deterministik. Tabel eksperimen menyediakan pembanding, tetapi hasil ini tidak membuktikan bahwa depth 8 adalah konfigurasi optimal untuk semua penggunaan.

Visualisasi pohon dipotong pada kedalaman tampilan agar lebih terbaca. Gambar terbatas tersebut bukan berarti pipeline prediksi memakai pohon dengan kedalaman tampilan yang lebih pendek; artefak model final memiliki kedalaman 8.

### 3.5 Metrik Evaluasi

- **Accuracy:** proporsi seluruh prediksi yang benar.
- **Precision:** dari prediksi untuk suatu kelas, proporsi yang benar.
- **Recall:** dari data aktual pada suatu kelas, proporsi yang berhasil ditemukan model.
- **F1-score:** harmonic mean precision dan recall.
- **Macro average:** rata-rata metrik per kelas dengan bobot kelas sama.
- **Weighted average:** rata-rata per kelas dengan bobot berdasarkan jumlah kasus.
- **Confusion matrix:** jumlah prediksi benar/salah untuk setiap pasangan kelas aktual dan prediksi.

Macro-F1 dan metrik per kelas penting karena distribusi dataset tidak seimbang. Accuracy saja dapat didominasi performa pada kelas Rendah.

## BAB IV — Rancangan dan Implementasi Aplikasi

### 4.1 Teknologi

- Python
- Flask untuk web application dan routing.
- pandas/NumPy untuk manipulasi data.
- scikit-learn untuk preprocessing, training, dan klasifikasi.
- joblib untuk menyimpan/memuat pipeline.
- Matplotlib untuk menghasilkan gambar pohon saat training.
- Chart.js untuk visualisasi grafik di browser.
- HTML, CSS, dan JavaScript untuk antarmuka.

### 4.2 Arsitektur Proyek

| Berkas/folder | Fungsi |
|---|---|
| `main.py` | Entry point Flask untuk local run dan WSGI. |
| `app/__init__.py` | Application factory, konfigurasi, dan template filter. |
| `app/routes.py` | Halaman web, health check, dan endpoint gambar pohon. |
| `app/services.py` | Memuat model/metadata, mengaudit dataset, membangun data dashboard, memvalidasi input, dan membuat prediksi. |
| `app/templates/` | Template Dashboard, Prediction, Dataset, Evaluation, Tree, About, dan Error. |
| `app/static/` | CSS, JavaScript, dan aset statis. |
| `src/data_loader.py` | Membaca dataset dan melakukan audit dasar. |
| `src/preprocessing.py` | Aturan target dan preprocessing pipeline. |
| `src/evaluation.py` | Metrik, confusion matrix, classification report, dan perbandingan train-test. |
| `train_model.py` | Training, eksperimen pruning, evaluasi, serta penyimpanan artefak. |
| `data/models/` | Pipeline `.joblib`, metadata JSON, metrik, dan visualisasi pohon. |
| `tests/` | Regression tests untuk target, validasi input, kategori tak dikenal, dan penyajian gambar pohon. |

### 4.3 Halaman dan Alur Pengguna

1. **Dashboard** — KPI jumlah data/genre/platform, metrik model, distribusi kelas, genre, platform, dan tahun rilis.
2. **Prediction** — pengguna mengisi Platform, Genre, Publisher, dan Year. Hasil menampilkan kategori, probabilitas per kelas, input yang diproses, serta disclaimer.
3. **Dataset Exploration** — ringkasan missing value/duplikasi, visualisasi EDA, interpretasi distribusi, serta tabel data yang dipaginasi.
4. **Model Evaluation** — perbandingan baseline/final, metrik, confusion matrix, classification report, dan konfigurasi eksperimen.
5. **Decision Tree** — gambar pohon, feature importance, zoom, dan unduh PNG.
6. **About Project** — identitas proyek, sumber data, ambang target, fitur, metodologi, dan batasan.

Endpoint tambahan:

- `/api/health` melaporkan kesiapan model dan dataset.
- `/tree/image` menyajikan gambar pohon dari artefak tanpa menulis ke filesystem. Ini mendukung filesystem serverless yang tidak persisten/read-only.

### 4.4 Validasi dan Penanganan Error

Form prediksi memeriksa field wajib, format Year, bilangan bulat, serta rentang input 1970–2035. Tahun pecahan tidak dibulatkan diam-diam. Kategori baru tetap dapat dimasukkan dan diproses encoder. Ketika model atau dataset tidak tersedia, aplikasi menampilkan keterangan dan health check melaporkan status `degraded`; detail exception internal dicatat ke log dan tidak ditampilkan sebagai hasil sukses.

## BAB V — Hasil dan Pembahasan

### 5.1 Hasil Evaluasi

Hasil berikut berasal dari `data/models/model_metadata.json` setelah koreksi target, dengan test set 3.320 baris.

| Model | Train accuracy | Test accuracy | Test precision weighted | Test recall weighted | Test F1 weighted | Test F1 macro | Gap accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline, tanpa batas kedalaman | 0,9069 | 0,7380 | 0,7106 | 0,7380 | 0,7219 | 0,4626 | 0,1690 |
| Final, depth 8 | 0,7816 | 0,7623 | 0,6957 | 0,7623 | 0,7057 | 0,3854 | 0,0192 |

Final mengurangi gap accuracy train-test dibanding baseline dan meningkatkan accuracy test. Namun macro-F1 final lebih rendah daripada baseline, dan hasil per kelas menunjukkan model lebih baik mengenali kelas mayoritas daripada kelas minoritas. Karena itu tidak tepat menyimpulkan bahwa final lebih baik pada semua aspek hanya dari accuracy.

### 5.2 Classification Report Model Final

| Kelas | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Rendah | 0,7903 | 0,9634 | 0,8684 | 2.516 |
| Sedang | 0,4359 | 0,1606 | 0,2348 | 635 |
| Tinggi | 0,2632 | 0,0296 | 0,0532 | 169 |
| Macro average F1 | — | — | 0,3854 | 3.320 |
| Weighted average F1 | — | — | 0,7057 | 3.320 |

### 5.3 Confusion Matrix

Urutan label pada baris aktual dan kolom prediksi adalah Rendah, Sedang, Tinggi:

| Aktual \ Prediksi | Rendah | Sedang | Tinggi |
|---|---:|---:|---:|
| Rendah | 2.424 | 85 | 7 |
| Sedang | 526 | 102 | 7 |
| Tinggi | 117 | 47 | 5 |

Sebanyak 5 dari 169 data aktual kelas Tinggi berhasil dikenali sebagai Tinggi pada test set. Banyak data Sedang dan Tinggi diprediksi sebagai Rendah. Hal ini konsisten dengan ketidakseimbangan kelas dan terbatasnya sinyal prediktif dari fitur karakteristik game.

### 5.4 Koreksi Target dan Perubahan Hasil

Audit awal menemukan 71 baris dengan `Global_Sales=0.5` salah masuk kelas Rendah. Setelah batas diperbaiki sesuai definisi target dan model dilatih ulang, jumlah kelas berubah:

- Rendah: 12.651 → 12.580.
- Sedang: 3.101 → 3.172.
- Tinggi: tetap 846.

Metrik final sebelum koreksi (accuracy 0,76898; macro-F1 0,39795) bukan hasil model terbaru. Hasil resmi proyek setelah koreksi adalah accuracy 0,76235 dan macro-F1 0,38543. README dan artefak metadata sudah diperbarui agar konsisten.

### 5.5 Interpretasi Probabilitas Prediksi

`predict_proba` pada Decision Tree menghitung proporsi kelas training di leaf yang dicapai oleh input. Nilai ini ditampilkan sebagai probabilitas estimasi untuk tiap kelas, tetapi tidak dikalibrasi sebagai peluang keberhasilan penjualan di dunia nyata dan bukan jaminan komersial. Interpretasi harus tetap mempertimbangkan performa test per kelas.

## BAB VI — Pengujian Kualitas

- 9 regression tests terakhir lulus saat konfigurasi deployment diperbarui.
- Skenario yang diuji meliputi batas target, target missing, tahun kosong/non-numerik/NaN/pecahan/di luar rentang, kategori baru, serta endpoint gambar pohon.
- Halaman utama diuji: Dashboard, Prediction, Dataset, Evaluation, Tree, dan About.
- Paginasi dataset, reset form, navigasi, zoom dan unduh gambar pohon diuji di browser.
- Responsivitas diperiksa pada viewport 1440×900, 900×1024, dan 390×844. Dashboard dan form prediksi tidak memiliki horizontal overflow; form menjadi satu kolom pada viewport ponsel.
- Pengujian lokal mencatat median sekitar 15,372 ms untuk service prediction pada 50 pemanggilan, dengan P95 18,499 ms. Angka ini hanya hasil workstation/interpreter pengujian dan bukan benchmark production Vercel.
- Uji input markup memastikan input di-escape; payload grafik diserialisasi dengan filter Jinja `tojson`.
- Pengujian kehilangan model/dataset dilakukan dengan direktori sementara; halaman memberikan penjelasan dan health check menjadi `degraded`.

## BAB VII — Deploy dan Status

Repository GitHub: [D13-gif/Prediksi-Penjual-an-Game-Berdasarkan-Genre-dan-Platform](https://github.com/D13-gif/Prediksi-Penjual-an-Game-Berdasarkan-Genre-dan-Platform).

Proyek memiliki konfigurasi `.python-version`, `vercel.json`, dan requirements runtime yang dipin. Setelah build Vercel melaporkan function bundle 811,91 MB, pola file tracing dibatasi dari glob seluruh proyek ke `{app/**,data/models/**,vgsales.csv}`. File runtime yang dihitung secara lokal berjumlah sekitar 2,34 MB. Perubahan config telah dipush, tetapi laporan ini **tidak menyatakan deployment Vercel berhasil** karena build ulang setelah perubahan tersebut belum diverifikasi. Keberhasilan deployment harus dikonfirmasi melalui hasil build terbaru dan endpoint `/api/health`.

## BAB VIII — Kesimpulan dan Saran

### 8.1 Kesimpulan

Proyek menghasilkan aplikasi Flask yang menyatukan eksplorasi data, evaluasi, visualisasi pohon, dan prediksi kategori penjualan video game. Target dibuat berdasarkan aturan proyek; pipeline memakai empat fitur non-penjualan dan preprocessing yang dibungkus bersama Decision Tree. Setelah koreksi label batas 0,5 dan retraining, final model memperoleh accuracy 0,7623 dan macro-F1 0,3854. Kinerja kelas Rendah jauh lebih baik dibanding Sedang dan Tinggi; recall kelas Tinggi 0,0296 membatasi kegunaan model untuk mengidentifikasi game berpenjualan tinggi.

### 8.2 Saran Pengembangan

1. Bahas dengan dosen/pemilik metodologi sebelum mengubah penanganan imbalance, misalnya class weight atau strategi sampling.
2. Untuk penelitian lanjutan, gunakan validation set atau cross-validation untuk pemilihan parameter, lalu pertahankan test set untuk evaluasi final sekali saja.
3. Tambahkan pembahasan biaya salah klasifikasi dan ukuran ketidakpastian, tanpa mengklaim probabilitas sebagai jaminan.
4. Jika aplikasi dipublikasikan, lakukan verifikasi deployment Vercel setelah glob file dipersempit, termasuk status health dan akses gambar/dataset/model.
5. Periksa syarat penggunaan dataset Kaggle sebelum redistribusi dataset mentah.

## Lampiran A — Cara Menjalankan

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-training.txt
python train_model.py
python main.py
```

Buka `http://127.0.0.1:5000`. Training dijalankan secara eksplisit melalui CLI; aplikasi web memuat artefak pipeline yang telah disimpan.

## Lampiran B — Bukti Visual

Screenshot aktual aplikasi tersimpan pada `docs/screenshots/`:

- `dashboard.png`
- `dashboard-mobile.png`
- `prediction.png`
- `dataset.png`
- `evaluation.png`
- `tree.png`
- `about.png`

Rincian kondisi awal, pengujian QA, dan retest tersedia pada [laporan evaluasi](./EVALUATION_REPORT.md) dan [log pengujian](./evaluation-test-log.txt).
