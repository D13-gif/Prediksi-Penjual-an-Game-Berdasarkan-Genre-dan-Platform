# ATURAN VIBE CODING

## Proyek Portofolio: Prediksi Penjualan Video Game Berdasarkan Genre dan Platform

**Pemilik proyek:** Adi Hadianto\
**NIM:** 301240006\
**Kelas:** 5B\
**Mata praktikum:** Pembelajaran Mesin\
**Algoritma utama:** Decision Tree Classifier\
**Dataset:** Video Game Sales (VGChartz) --- Kaggle\
**Sumber dataset:**
https://www.kaggle.com/datasets/gregorut/videogamesales

------------------------------------------------------------------------

## 1. Tujuan Dokumen

Dokumen ini menjadi *project instruction* untuk AI coding assistant saat
membangun aplikasi menggunakan pendekatan **vibe coding**. AI harus
bekerja berdasarkan instruksi, batasan teknis, dan kriteria penerimaan
di bawah ini.

Hasil akhir harus memenuhi dua tujuan sekaligus:

1.  Memenuhi tahapan pembelajaran pada Modul 05: Algoritma Decision
    Tree.
2.  Menjadi aplikasi portofolio yang rapi, profesional, mudah digunakan,
    responsif, dan dapat dipresentasikan kepada dosen maupun calon
    perekrut.

AI tidak boleh hanya membuat tampilan demo. Seluruh alur
utama---pemuatan data, preprocessing, pelatihan model, evaluasi,
penyimpanan model, dan prediksi melalui web---harus benar-benar
berfungsi.

## 2. Identitas dan Ruang Lingkup Proyek

### Judul

**Penerapan Algoritma Decision Tree untuk Klasifikasi Tingkat Penjualan
Video Game Berdasarkan Genre dan Platform.**

### Permasalahan

Aplikasi mengklasifikasikan tingkat penjualan global sebuah video game
berdasarkan informasi yang tersedia, terutama genre dan platform. Hasil
klasifikasi ditampilkan sebagai kategori **Rendah**, **Sedang**, atau
**Tinggi**.

### Dataset

Gunakan dataset Kaggle Video Game Sales (VGChartz) dari tautan yang
tercantum di atas. Jangan mengganti sumber data dengan dataset bawaan
library atau data buatan untuk menggantikan hasil eksperimen.

File dataset yang umum digunakan bernama `vgsales.csv`. Jika nama atau
struktur file berbeda, periksa isi file terlebih dahulu dan
dokumentasikan penyesuaiannya.

### Definisi target

Kolom asli `Global_Sales` berisi angka penjualan global dalam satuan
juta kopi. Untuk menggunakan `DecisionTreeClassifier`, buat kolom target
turunan bernama `Sales_Category` dengan aturan awal berikut:

  Kategori   Aturan
  ---------- ----------------------------
  Rendah     `Global_Sales` \< 0.5
  Sedang     0.5 ≤ `Global_Sales` ≤ 2.0
  Tinggi     `Global_Sales` \> 2.0

Batas tersebut adalah aturan kategorisasi proyek, bukan label asli
Kaggle. Tampilkan aturan ini pada halaman metodologi agar pengguna
memahami cara label dibuat. Jangan mengubah batas kategori diam-diam.

### Fitur awal

Gunakan fitur yang tersedia dan relevan, misalnya: - `Genre` -
`Platform` - `Year` - `Publisher`

Kolom yang tidak tersedia atau kosong dalam dataset harus ditangani
dengan aman. `Name` boleh ditampilkan sebagai informasi, tetapi jangan
otomatis dijadikan fitur prediksi karena bukan karakteristik umum yang
ingin dianalisis.

**Pencegahan data leakage:** `Global_Sales` hanya digunakan untuk
membentuk target, bukan sebagai fitur input model. Kolom penjualan
regional (`NA_Sales`, `EU_Sales`, `JP_Sales`, `Other_Sales`) juga jangan
digunakan sebagai fitur utama jika tujuan aplikasi adalah memprediksi
tingkat penjualan dari genre/platform sebelum penjualan terjadi.
Jelaskan keputusan fitur tersebut.

## 3. Ketentuan Wajib Mengikuti Modul Decision Tree

Implementasi harus mencakup seluruh tahapan berikut.

### 3.1 Persiapan data

-   Memuat dataset menggunakan Pandas.
-   Menampilkan lima baris pertama (`head()`).
-   Menampilkan ukuran data (`shape`) dan ringkasan tipe data
    (`info()`).
-   Memeriksa nilai kosong, duplikasi, dan nilai tidak valid.
-   Menjelaskan kolom yang dipakai, kolom yang dihapus, dan alasannya.

### 3.2 Exploratory Data Analysis (EDA)

Sediakan analisis dan visualisasi yang relevan: - Distribusi kategori
target `Sales_Category`. - Jumlah game berdasarkan genre. - Jumlah game
berdasarkan platform. - Distribusi tahun rilis. - Perbandingan kategori
penjualan menurut genre. - Perbandingan kategori penjualan menurut
platform.

Gunakan grafik dengan judul, label sumbu, legenda bila diperlukan, dan
format angka yang mudah dibaca. Kesimpulan EDA harus berasal dari hasil
dataset yang benar-benar dihitung, bukan angka rekaan.

### 3.3 Preprocessing

-   Tangani missing values dengan strategi yang sesuai tipe data.
-   Tangani data duplikat secara terukur dan dokumentasikan.
-   Ubah fitur kategorikal menjadi numerik menggunakan pipeline encoding
    yang tepat, misalnya `OneHotEncoder(handle_unknown="ignore")`.
-   Gunakan `ColumnTransformer` dan `Pipeline` dari Scikit-learn agar
    preprocessing konsisten saat training dan prediksi.
-   Jangan melakukan fitting encoder pada seluruh dataset sebelum
    train-test split.
-   Pastikan input kategori baru saat prediksi tidak menyebabkan
    aplikasi crash.

### 3.4 Pembagian data

-   Pisahkan data training dan testing menggunakan `train_test_split`.
-   Gunakan proporsi awal 80:20.
-   Gunakan `random_state` tetap agar eksperimen dapat direproduksi.
-   Gunakan `stratify=y` jika distribusi kelas memungkinkan.
-   Tampilkan jumlah data pada masing-masing bagian dan distribusi
    kelasnya.

### 3.5 Pelatihan Decision Tree

-   Gunakan `DecisionTreeClassifier` dari Scikit-learn.
-   Gunakan `criterion="gini"` sebagai konfigurasi awal sesuai materi
    Gini Impurity.
-   Latih minimal satu model baseline.
-   Catat parameter model yang digunakan.
-   Jangan mengklaim model terbaik sebelum melakukan evaluasi.

### 3.6 Overfitting dan pruning

Bandingkan pohon baseline dengan pohon yang dibatasi kompleksitasnya.
Uji parameter yang relevan, antara lain: - `max_depth` -
`min_samples_split` - `min_samples_leaf` - `ccp_alpha` untuk
cost-complexity pruning bila sesuai

Tampilkan perbandingan skor training dan testing. Pilih model akhir
berdasarkan evaluasi pada data validasi/testing, bukan hanya karena skor
training tinggi. Jelaskan indikasi overfitting dan alasan pemilihan
parameter.

### 3.7 Evaluasi

Minimal tampilkan: - Accuracy - Precision - Recall - F1-score -
Confusion matrix - Classification report

Untuk klasifikasi multi-kelas, jelaskan penggunaan rata-rata `macro`
atau `weighted` jika diperlukan. Jangan hanya menampilkan accuracy.
Berikan interpretasi singkat yang sesuai hasil aktual.

### 3.8 Visualisasi pohon

-   Visualisasikan Decision Tree yang telah dilatih menggunakan
    `plot_tree` atau visualisasi yang setara.
-   Tampilkan nama fitur dan kelas bila memungkinkan.
-   Batasi kedalaman visualisasi agar tetap terbaca.
-   Sediakan keterangan bahwa visualisasi yang dipotong hanya untuk
    keterbacaan; model tersimpan tetap menggunakan konfigurasi hasil
    training.

### 3.9 Penyimpanan model

-   Simpan pipeline preprocessing dan model final dalam satu artefak,
    misalnya `joblib`.
-   Simpan metadata eksperimen seperti nama dataset, fitur, target,
    parameter, tanggal training, dan metrik.
-   Aplikasi harus memuat model tersimpan, bukan melatih ulang setiap
    kali pengguna membuka halaman.
-   Berikan pesan error yang jelas jika file model belum tersedia.

### 3.10 Aplikasi web Flask

Bangun aplikasi web menggunakan Flask yang memiliki: 1. Dashboard
ringkasan proyek. 2. Halaman prediksi. 3. Halaman eksplorasi
dataset/EDA. 4. Halaman evaluasi model. 5. Halaman tentang metodologi
dan dataset.

Form prediksi minimal menerima Genre, Platform, Year, dan Publisher
sesuai fitur final model. Field wajib harus divalidasi. Tampilkan hasil
kategori prediksi, probabilitas tiap kelas jika tersedia, dan penjelasan
bahwa hasil merupakan estimasi model, bukan kepastian penjualan.

## 4. Aturan Desain UI/UX: Modern, Menarik, dan Profesional

Tampilan merupakan bagian penting dari portofolio. Terapkan prinsip
berikut.

### 4.1 Arah visual

-   Gunakan gaya **modern data analytics dashboard** dengan nuansa
    industri game.
-   Gunakan warna dasar gelap elegan atau terang bersih secara
    konsisten. Pilih satu tema utama, jangan mencampur terlalu banyak
    gaya.
-   Rekomendasi palet: navy/charcoal sebagai warna dasar, aksen ungu
    atau biru elektrik, serta warna status yang memiliki kontras baik.
-   Hindari neon berlebihan, gradien berlebihan, animasi yang
    mengganggu, dan tampilan seperti template mentah.
-   Gunakan tipografi modern yang mudah dibaca, hierarki judul jelas,
    dan jarak antarelemen yang konsisten.

### 4.2 Layout

-   Buat sidebar navigasi yang rapi pada desktop dan navigasi yang
    nyaman pada layar kecil.
-   Dashboard memiliki header, ringkasan metrik, grafik utama, dan
    aktivitas/insight dataset.
-   Gunakan card dengan radius dan shadow halus; jangan menumpuk terlalu
    banyak card.
-   Beri ruang kosong yang cukup agar halaman tidak terasa padat.
-   Gunakan tabel dengan header jelas, zebra/hover seperlunya, dan
    tampilan responsif.
-   Form prediksi harus sederhana, memiliki label, placeholder,
    petunjuk, dan validasi yang jelas.

### 4.3 Komponen visual

-   KPI cards: jumlah data, jumlah genre, jumlah platform, dan metrik
    model yang benar-benar dihitung.
-   Grafik interaktif atau statis berkualitas tinggi dengan tooltip dan
    label yang informatif.
-   Badge kategori untuk Rendah, Sedang, dan Tinggi dengan warna
    konsisten.
-   Confusion matrix yang mudah dibaca.
-   Pohon keputusan ditampilkan dalam panel khusus yang dapat diperbesar
    atau diunduh bila memungkinkan.
-   Sediakan loading state, empty state, success state, dan error state.

### 4.4 Responsivitas dan aksesibilitas

-   Pastikan tampilan nyaman pada desktop, tablet, dan ponsel.
-   Kontras teks dan latar harus memadai.
-   Jangan mengandalkan warna saja untuk menyampaikan status; sertakan
    teks.
-   Semua tombol memiliki label yang jelas dan area klik memadai.
-   Form dapat digunakan dengan keyboard dan memiliki pesan validasi
    yang mudah dipahami.

### 4.5 Micro-interaction

Gunakan animasi ringan untuk hover, transisi halaman, dan feedback aksi.
Hindari animasi berlebihan atau efek yang memperlambat aplikasi. Fitur
utama harus tetap dapat digunakan tanpa animasi.

## 5. Arsitektur dan Kualitas Kode

Gunakan struktur proyek yang mudah dipahami dan dirawat. Contoh
struktur:

``` text
game-sales-prediction/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── vgsales.csv
├── models/
│   ├── decision_tree_pipeline.joblib
│   └── model_metadata.json
├── notebooks/
│   └── eda_experiment.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   └── evaluation.py
├── templates/
│   ├── base.html
│   ├── dashboard.html
│   ├── predict.html
│   ├── dataset.html
│   ├── evaluation.html
│   └── about.html
└── static/
    ├── css/
    │   └── style.css
    ├── js/
    └── images/
```

Struktur tersebut adalah rekomendasi profesional. Jika dosen atau
lingkungan praktikum mensyaratkan struktur berbeda, ikuti ketentuan
praktikum dan dokumentasikan penyesuaiannya.

Aturan kode: - Gunakan Python dan library yang sesuai: Pandas, NumPy,
Matplotlib/Seaborn atau Plotly, Scikit-learn, Flask, dan Joblib. -
Pisahkan logika pemrosesan data, training, evaluasi, dan web. - Gunakan
nama variabel dan fungsi yang jelas. - Tambahkan komentar pada bagian
penting, bukan komentar berlebihan. - Hindari duplikasi kode. - Jangan
menaruh kredensial, token, atau path lokal pribadi di source code. -
Gunakan `requirements.txt` dengan dependensi yang benar-benar dipakai. -
Tangani error file tidak ditemukan, kolom tidak sesuai, input tidak
valid, dan model belum tersedia. - Jangan mengarang hasil eksperimen,
screenshot, metrik, atau status deployment.

## 6. Aturan Kerja Vibe Coding untuk AI Assistant

AI coding assistant wajib mengikuti alur kerja berikut.

1.  **Pahami proyek terlebih dahulu.** Periksa file, struktur folder,
    dataset, dan instruksi ini sebelum mengubah apa pun.
2.  **Buat rencana singkat.** Jelaskan fitur yang akan dibuat dan file
    yang akan ditambahkan/diubah.
3.  **Kerjakan bertahap.** Prioritaskan data pipeline dan model yang
    benar sebelum mempercantik UI.
4.  **Jangan menghapus fitur atau file yang sudah ada** tanpa alasan
    teknis dan persetujuan pengguna.
5.  **Jangan mengganti dataset atau target** tanpa persetujuan.
6.  **Jangan membuat data atau metrik palsu** demi membuat dashboard
    terlihat penuh.
7.  **Jalankan pengujian setelah perubahan.** Perbaiki error sebelum
    melanjutkan.
8.  **Jaga konsistensi desain.** Gunakan komponen, warna, spacing, dan
    tipografi yang konsisten.
9.  **Jelaskan perubahan.** Setelah setiap tahap, sebutkan file yang
    dibuat/diubah, cara menjalankan, dan hasil pengujian.
10. **Jangan menyatakan selesai jika belum diuji.** Bedakan dengan jelas
    antara fitur yang sudah berjalan, belum diuji, dan masih berupa
    placeholder.
11. **Minta klarifikasi** jika ada instruksi yang bertentangan atau
    keputusan yang berdampak pada target, fitur, atau metodologi.
12. **Jaga keamanan.** Validasi input, batasi ukuran upload jika ada,
    jangan mengeksekusi file pengguna, dan jangan menampilkan informasi
    sensitif pada log.

### Format respons AI setiap tahap

Setiap kali menyelesaikan tahap, AI harus memberikan: - Ringkasan
pekerjaan. - Daftar file yang dibuat/diubah. - Cara menjalankan atau
menguji. - Hasil pengujian aktual. - Kendala yang masih tersisa. -
Langkah berikutnya yang disarankan.

## 7. Pengujian dan Kriteria Penerimaan

Proyek dianggap memenuhi instruksi apabila:

-   [ ] Dataset berasal dari tautan Kaggle yang ditentukan.
-   [ ] Dataset berhasil dimuat dan diperiksa.
-   [ ] EDA memiliki grafik dan interpretasi berdasarkan data aktual.
-   [ ] Target `Sales_Category` dibuat sesuai batas yang
    didokumentasikan.
-   [ ] `Global_Sales` dan fitur yang menyebabkan data leakage tidak
    masuk sebagai fitur prediksi.
-   [ ] Preprocessing kategorikal berjalan konsisten melalui pipeline.
-   [ ] Data dibagi menjadi training dan testing.
-   [ ] Decision Tree baseline berhasil dilatih.
-   [ ] Eksperimen pruning dilakukan dan dibandingkan.
-   [ ] Evaluasi menampilkan accuracy, precision, recall, F1-score, dan
    confusion matrix.
-   [ ] Pohon keputusan dapat divisualisasikan.
-   [ ] Pipeline final tersimpan dan dapat dimuat kembali.
-   [ ] Flask dapat dijalankan dan halaman utama dapat diakses.
-   [ ] Form prediksi menghasilkan output dari model yang tersimpan.
-   [ ] Input tidak valid ditangani dengan pesan yang jelas.
-   [ ] UI modern, konsisten, responsif, dan nyaman digunakan.
-   [ ] README menjelaskan instalasi, struktur, dataset, metodologi,
    training, dan cara menjalankan aplikasi.
-   [ ] Screenshot yang dimasukkan merupakan hasil aplikasi yang
    benar-benar dijalankan.

## 8. Dokumentasi dan Portofolio

README harus memuat: - Nama dan deskripsi proyek. - Tujuan dan
permasalahan. - Sumber serta lisensi/ketentuan penggunaan dataset sesuai
halaman Kaggle. - Penjelasan fitur dan target. - Alur preprocessing dan
training. - Strategi pruning. - Hasil evaluasi aktual. - Screenshot
dashboard, halaman prediksi, EDA, evaluasi, dan visualisasi pohon. -
Instruksi instalasi dan menjalankan aplikasi. - Batasan model dan ide
pengembangan berikutnya.

Jika proyek dipublikasikan ke GitHub: - Jangan unggah file sensitif atau
kredensial. - Periksa ukuran dataset dan ketentuan redistribusi Kaggle
sebelum menyertakan dataset mentah. - Sertakan instruksi agar pengguna
mengunduh dataset dari sumber resmi bila perlu. - Sertakan
`requirements.txt` dan README yang dapat diikuti dari awal.

## 9. Batasan dan Kejujuran Akademik

-   Aplikasi adalah proyek pembelajaran dan portofolio, bukan alat untuk
    menjamin keberhasilan komersial sebuah game.
-   Kategori penjualan merupakan hasil diskretisasi dari `Global_Sales`;
    jelaskan konsekuensi informasi yang hilang akibat kategorisasi.
-   Hasil model bergantung pada kualitas dataset, pemilihan fitur, dan
    pembagian data.
-   Jangan menyebut model akurat/terbaik tanpa bukti evaluasi.
-   Jangan membuat screenshot, hasil training, grafik, atau tautan
    deployment palsu.
-   Cantumkan sumber dataset dan dokumentasi library yang digunakan.

## 10. Prompt Awal untuk AI Vibe Coding

Salin prompt berikut saat memulai sesi coding:

> Baca seluruh file proyek dan dokumen `VIBE_CODING_RULES.md` terlebih
> dahulu. Bangun aplikasi portofolio berjudul "Penerapan Algoritma
> Decision Tree untuk Klasifikasi Tingkat Penjualan Video Game
> Berdasarkan Genre dan Platform". Ikuti seluruh ketentuan Modul 05
> Decision Tree dan aturan pada dokumen ini. Gunakan dataset Kaggle
> VGChartz yang memiliki kolom `Global_Sales`, bentuk target
> `Sales_Category` sesuai ambang yang ditetapkan, dan cegah data
> leakage. Implementasikan EDA, preprocessing pipeline, train-test
> split, Decision Tree baseline, eksperimen pruning, evaluasi lengkap,
> visualisasi pohon, penyimpanan pipeline, serta aplikasi web Flask.
> Buat UI dashboard modern, profesional, menarik, dan responsif dengan
> pengalaman pengguna yang baik. Jangan mengarang data, metrik,
> screenshot, atau hasil pengujian. Kerjakan secara bertahap: mulai dari
> audit dataset dan rencana implementasi, lalu tunggu persetujuan
> sebelum melakukan perubahan besar. Setelah setiap tahap, laporkan file
> yang berubah, cara menjalankan, dan hasil pengujian aktual.

------------------------------------------------------------------------

**Catatan:** Instruksi ini menggabungkan tahapan teknis Modul 05
Decision Tree---EDA, preprocessing, training, evaluasi, pruning,
visualisasi pohon, penyimpanan model, dan integrasi Flask---dengan
standar tambahan untuk kualitas UI/UX dan portofolio. Standar desain
modern merupakan persyaratan pengembangan tambahan, bukan klaim bahwa
seluruh detail desain tersebut diwajibkan oleh modul praktikum.
