# Klasifikasi Dropout dan Keberhasilan Akademik Mahasiswa dengan Machine Learning

Halo salam kenal, saya **Juan Luthfi Ryandi**, mahasiswa Teknik Informatika Universitas Pamulang.

Di repositori ini saya membangun model untuk **memprediksi status akademik seorang mahasiswa**, apakah ia akan *Dropout*, masih *Enrolled* (aktif kuliah), atau *Graduate* (lulus), berdasarkan data pendaftaran, kondisi sosial-ekonomi, dan kinerja akademiknya di semester 1 dan 2. Seluruh prosesnya saya kerjakan dari nol sampai model jadi, mengikuti alur yang diajarkan dosen saya: mulai dari mengumpulkan data, membersihkan, menganalisis, membagi data, melatih model, sampai mengevaluasi dan menyimpannya.


---

## Kenapa saya memilih klasifikasi?

Ada dua jenis masalah utama di *supervised learning*. Kalau yang diprediksi berupa **angka** (misalnya biaya atau harga), itu **regresi**. Kalau yang diprediksi berupa **kategori atau label**, itu **klasifikasi**.

Pertanyaan yang ingin saya jawab di proyek ini adalah: *"Mahasiswa ini akhirnya akan dropout, tetap aktif, atau lulus?"* Jawabannya bukan angka, melainkan salah satu dari tiga label. Karena itu pendekatan yang tepat adalah **klasifikasi**, dan karena labelnya ada tiga, jenisnya disebut **klasifikasi multikelas**.

Klasifikasi juga cocok dengan tujuan praktisnya. Kampus tidak butuh tebakan "nilai dropout seorang mahasiswa adalah 0,37". Kampus butuh jawaban tegas: **mahasiswa ini berisiko atau tidak**, sehingga dosen wali bisa segera mendampinginya.

## Kenapa saya memakai dataset ini?

Saya memakai dataset **Predict Students' Dropout and Academic Success** dari UCI Machine Learning Repository. Alasannya:

- **Masalahnya nyata dan bermanfaat.** Dataset ini dibuat untuk membantu institusi pendidikan tinggi mengenali mahasiswa yang berisiko dropout sejak dini, supaya bantuan bisa diberikan sebelum terlambat. Topik ini dekat dengan dunia kampus saya sendiri.
- **Cocok untuk klasifikasi multikelas.** Targetnya punya tiga kelas (Dropout, Enrolled, Graduate), jadi lebih menantang dan lebih menarik daripada klasifikasi dua kelas biasa.
- **Ukurannya pas untuk belajar.** Ada 4.424 mahasiswa dan 36 fitur, cukup banyak untuk model yang bermakna tetapi masih ringan dijalankan di laptop atau Google Colab.
- **Fiturnya beragam.** Ada data demografis, latar belakang sosial-ekonomi, jalur pendaftaran, kinerja akademik, sampai indikator makro-ekonomi. Saya jadi bisa berlatih seluruh tahap pra-pemrosesan dan EDA.
- **Datanya relatif bersih, tetapi tidak sempurna.** Tidak ada data kosong dan tidak ada baris duplikat, tetapi ada nama kolom yang bermasalah dan banyak pencilan (*outlier*). Jadi tahap *data cleaning* tetap bermakna.
- **Kelasnya tidak seimbang.** Ini justru alasan yang paling menarik buat saya: pada data seperti ini, accuracy saja bisa menipu, jadi saya bisa belajar memilih metrik dan penanganan yang tepat. Penjelasannya ada di bagian berikut.

---

## Apakah datanya imbalance?

**Ya, datanya tidak seimbang (*imbalanced*), meski tingkatnya sedang.** Ini hasil hitungan langsung pada kolom `Target`:

| Kelas      | Jumlah mahasiswa | Proporsi |
| ---------- | ---------------- | -------- |
| Graduate   | 2.209            | 49,9%    |
| Dropout    | 1.421            | 32,1%    |
| Enrolled   | 794              | 17,9%    |
| **Total**  | **4.424**        | 100%     |

Kelas terbesar (Graduate) hampir **2,8 kali lipat** kelas terkecil (Enrolled).

**Kenapa ini jadi masalah?** Bayangkan model yang malas dan selalu menebak "Graduate" untuk semua mahasiswa. Accuracy-nya sudah sekitar 50%, padahal model itu sama sekali tidak mengenali mahasiswa yang berisiko dropout. Model juga cenderung lebih "rajin" belajar dari kelas yang banyak, sehingga kelas kecil seperti Enrolled sering terabaikan.

### Bagaimana saya menanganinya?

Saya tidak menambah atau membuang baris data. Penanganannya ada tiga lapis:

1. **Pembagian data berstrata (`stratify=y`).** Saat membagi data latih dan data uji, proporsi tiap kelas dijaga tetap sama. Hasilnya, proporsi kelas hampir identik di data total, data latih, dan data uji (sekitar 49,9% / 32,1% / 17,9%). Tanpa ini, data uji bisa kebetulan hanya berisi sedikit mahasiswa Enrolled sehingga evaluasinya tidak adil.
2. **Metrik yang tepat.** Saya tidak hanya melihat *accuracy*, tetapi juga **F1-score macro** (memperlakukan setiap kelas secara setara), **precision**, **recall per kelas**, dan **confusion matrix**.
3. **Pembobotan kelas (`class_weight='balanced'`).** Pada Random Forest, kesalahan pada kelas kecil diberi "hukuman" lebih besar, sehingga model terdorong lebih memperhatikan kelas Enrolled.

**Hasilnya** (Random Forest, data uji 885 mahasiswa):

| Recall per kelas | Sebelum (baseline) | Sesudah (`balanced`) |
| ---------------- | ------------------ | -------------------- |
| Dropout          | 0,761              | 0,694                |
| Enrolled         | 0,365              | **0,560**            |
| Graduate         | 0,937              | 0,855                |

| Metrik     | Sebelum | Sesudah |
| ---------- | ------- | ------- |
| Accuracy   | 0,7774  | 0,7503  |
| F1 macro   | 0,6991  | **0,7016** |

**Cara membacanya:** Recall kelas Enrolled naik tajam dari 0,365 menjadi 0,560, artinya model jauh lebih sering berhasil menemukan mahasiswa Enrolled. Harganya, recall Dropout dan Graduate turun sedikit dan accuracy turun sekitar 2,7 poin. Ini *trade-off* yang saya anggap wajar, karena model yang unggul di accuracy tetapi buta terhadap kelas minoritas kurang berguna untuk sistem peringatan dini. Itu sebabnya saya memilih model berdasarkan F1 macro, bukan accuracy.

---

## Dataset yang dipakai

Berkas `mahasiswa.csv` berisi **4.424 baris dan 37 kolom** (36 fitur + 1 kolom target), dengan pemisah kolom titik koma (`;`). Data berasal dari sebuah institusi pendidikan tinggi (Polytechnic Institute of Portalegre, Portugal) dan menggabungkan beberapa sumber data. Kolom-kolomnya saya kelompokkan seperti ini:

| Kelompok                  | Contoh kolom                                                                                                 |
| ------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Demografis                | `Marital status`, `Gender`, `Age at enrollment`, `Nationality`, `International`                               |
| Latar belakang keluarga   | `Mother's qualification`, `Father's qualification`, `Mother's occupation`, `Father's occupation`              |
| Sosial-ekonomi            | `Scholarship holder`, `Debtor`, `Tuition fees up to date`, `Displaced`, `Educational special needs`           |
| Jalur pendaftaran         | `Application mode`, `Application order`, `Course`, `Daytime/evening attendance`, `Previous qualification`, `Admission grade` |
| Kinerja akademik semester 1 dan 2 | `Curricular units 1st/2nd sem` (credited, enrolled, evaluations, approved, grade, without evaluations) |
| Makro-ekonomi             | `Unemployment rate`, `Inflation rate`, `GDP`                                                                  |
| **Target**                | **`Target`: Dropout, Enrolled, atau Graduate**                                                                |

Sebagian besar fitur kategorikal sudah berbentuk **kode numerik** dari sumber data (misalnya kode program studi di kolom `Course`).

---

## Proses yang saya kerjakan

Berikut urutan yang saya kerjakan pada Google Collab. Di setiap tahap saya jelaskan **apa yang dilakukan** dan **kenapa harus dilakukan**.

### 1. Data Collecting dan Data Loading

Saya mengunduh dataset dari UCI Machine Learning Repository, lalu memuatnya dengan `pandas` memakai `sep=';'` karena pemisah kolomnya titik koma, dan `encoding='utf-8-sig'` untuk menangani penanda urutan byte (BOM) di awal berkas. Dimensi data mentah: **(4424, 37)**. Setelah itu saya membuat salinan `df` untuk dikerjakan, dan nama kolom dirapikan: karakter *tab* tersembunyi di kolom `Daytime/evening attendance` dihapus dan typo `Nacionality` diperbaiki menjadi `Nationality`.

**Kenapa begitu?** Semua proyek machine learning dimulai dari data, dan data harus terbaca dengan benar dulu. Salinan dibuat supaya data asli tetap utuh kalau ada langkah pembersihan yang perlu diulang.

### 2. Data Cleaning (membersihkan data)

Model yang dilatih dari data kotor akan memberi hasil yang kurang bisa dipercaya. Ada empat hal yang saya periksa:

- **Informasi dan statistik dataset** (`info()` dan `describe()`): memastikan tipe data benar dan melihat sebaran awal. Semua kolom terisi penuh, dan kolom `Course` punya kode yang rentangnya sangat lebar (33 sampai 9.991).
- **Missing value** (data kosong): hasilnya **tidak ada data kosong**. Kode penanganannya (median untuk numerik, modus untuk kategorikal) tetap saya jalankan sebagai langkah pengaman, supaya alurnya lengkap dan bisa dipakai ulang di dataset lain.
- **Data duplikat**: ada **0 baris duplikat**, jadi tidak ada baris yang dihapus.
- **Outlier** (nilai ekstrem): saya mendeteksinya dengan metode **IQR** (nilai di bawah Q1 − 1,5×IQR atau di atas Q3 + 1,5×IQR) pada 18 fitur numerik, lengkap dengan box plot. Pencilan terbanyak ada di nilai rata-rata semester 2 (877 baris, 19,82%) dan semester 1 (726 baris, 16,41%), yang umumnya berupa nilai 0 dari mahasiswa yang tidak punya nilai pada semester itu.

  Sebelum memutuskan, saya cek dulu apakah pencilan itu membawa informasi. Hasilnya, **proporsi Dropout pada mahasiswa dengan usia di atas batas IQR mencapai 53,7%**, jauh di atas proporsi Dropout di seluruh data (32,1%). Artinya pencilan usia bukan kesalahan input, melainkan sinyal yang penting. Maka keputusan saya:
  - **Tidak ada baris yang dihapus.** Ukuran data tetap (4424, 37).
  - **Capping (winsorizing)** pada tiga fitur kontinu: `Age at enrollment`, `Admission grade`, dan `Previous qualification (grade)`. Nilainya dibatasi ke batas IQR.
  - Fitur hitungan mata kuliah dan nilai rata-rata semester **tidak diubah**, karena pencilannya adalah nilai valid yang memang berpola nol.

### 3. Data Transformation: Encoding

- **Target** (`Dropout`, `Enrolled`, `Graduate`) diubah menjadi angka dengan **Label Encoding**: Dropout = 0, Enrolled = 1, Graduate = 2.
- **Fitur kategorikal** sudah berbentuk kode numerik dari sumber data, jadi tetap dipakai dalam bentuk itu. Sebagai pembanding, saya mencoba **One-Hot Encoding** pada `Marital status` dan `Course`, dan jumlah kolomnya melonjak dari 37 menjadi 58.

**Kenapa begitu?** Komputer hanya mengerti angka, jadi target harus dikodekan. Untuk fitur kode, Label Encoding menjaga dimensi data tetap ringkas, sedangkan One-Hot membuat kolom membengkak. Catatan: kode seperti `Course` sebenarnya kategori tanpa urutan, jadi pendekatan ini punya keterbatasan (lihat bagian catatan di bawah).

### 4. EDA dan ExDA (mengenali data lewat grafik)

Sebelum melatih model, saya "berkenalan" dulu dengan data lewat visualisasi: distribusi target, histogram fitur numerik, box plot dan fitur biner terhadap target, komposisi target per program studi, matriks korelasi, dan korelasi terhadap Dropout.

**Kenapa penting?** Dari tahap ini saya sudah bisa menebak jawabannya sebelum model dibuat:

- **Kinerja akademik adalah pembeda paling kuat.** Rata-rata mata kuliah lulus semester 1 adalah 2,55 (Dropout), 4,32 (Enrolled), dan 6,23 (Graduate). Nilai rata-rata semester 2 hanya 5,90 untuk Dropout dibanding 12,70 untuk Graduate.
- **Kondisi keuangan sangat berpengaruh.** Mahasiswa yang biaya kuliahnya tidak up to date punya persentase Dropout 86,6%, sedangkan yang up to date hanya 24,7%. Penerima beasiswa juga jauh lebih jarang dropout (12,2% dibanding 38,7%).
- **Usia ikut berperan.** Rata-rata usia saat mendaftar pada kelas Dropout 24,74 tahun, sedangkan Enrolled dan Graduate sekitar 21 tahun. Sebaliknya, `Admission grade` relatif mirip antarkelas, jadi nilai masuk bukan pembeda yang kuat.
- **Program studi berpengaruh.** Persentase Dropout tertinggi ada di Biofuel Production Technologies (66,7%), disusul Equinculture (55,3%) dan Informatics Engineering (54,1%).
- **Korelasi terhadap Dropout** terkuat dimiliki nilai rata-rata semester 2 (−0,572) dan jumlah mata kuliah lulus semester 2 (−0,570).
- **Ada multikolinearitas.** Lima pasangan fitur punya korelasi di atas 0,80, misalnya `Mother's occupation` dengan `Father's occupation` (0,910) dan semester 1 dengan semester 2 pada beberapa indikator akademik.

### 5. Data Splitting (membagi data)

Pertama, data dipisah menjadi matriks fitur **X** (36 kolom) dan target **y**. Lalu dibagi **80% untuk latih (3.539 baris)** dan **20% untuk uji (885 baris)** dengan `train_test_split(test_size=0.2, random_state=42, stratify=y)`.

**Kenapa harus dibagi?** Ibarat belajar untuk ujian: kalau soal ujian sama persis dengan soal latihan, nilai bagus belum tentu berarti paham. Data uji adalah "soal ujian" yang belum pernah dilihat model, sehingga kemampuan model yang sebenarnya bisa dinilai secara jujur. Pembagiannya **berstrata** supaya proporsi kelas tetap terjaga, seperti dijelaskan di bagian imbalance.

### 6. Data Transformation: Standardisasi

Seluruh 36 fitur distandarisasi dengan `StandardScaler` (rata-rata 0 dan simpangan baku 1). **Scaler hanya dipelajari (`fit`) dari data latih**, sedangkan data uji hanya di-`transform` dengan parameter yang sama.

**Kenapa begitu?** Skala antarfitur berbeda jauh (kode `Course` sampai 9.991, sedangkan fitur lain hanya satuan). Model berbasis jarak atau margin seperti KNN, SVM, dan Logistic Regression bisa didominasi fitur berskala besar. Scaler hanya belajar dari data latih agar tidak terjadi **kebocoran data (*data leakage*)** dari data uji.

### 7. Modelling (membangun model)

Saya melatih dan membandingkan **enam algoritma klasifikasi**:

| Model               | Gambaran singkat                                                  |
| ------------------- | ----------------------------------------------------------------- |
| Logistic Regression | Model linier, berperan sebagai pembanding dasar                   |
| KNN (k = 7)         | Memprediksi dari tetangga terdekat (berbasis jarak)               |
| Decision Tree       | Pohon keputusan tunggal (`max_depth=6`)                           |
| SVM (RBF)           | Mencari batas pemisah kelas (berbasis margin)                     |
| Random Forest       | Kumpulan banyak pohon yang "memilih bersama" (300 pohon)          |
| Gradient Boosting   | Pohon dibangun berurutan, tiap pohon memperbaiki kesalahan sebelumnya |

**Kenapa enam model?** Dengan membandingkan model sederhana sampai kompleks, kita bisa melihat apakah kerumitan tambahan benar-benar memberi hasil yang lebih baik.

### 8. Evaluasi Model

Model diuji pada 885 data uji dengan accuracy, precision macro, recall macro, F1 macro, F1 weighted, serta **cross-validation 5-fold** (`StratifiedKFold`) pada data latih. Hasilnya diurutkan berdasarkan F1 macro:

| Model                 | Accuracy | F1 (macro) | F1 (weighted) | CV Acc (5-fold) |
| --------------------- | -------- | ---------- | ------------- | --------------- |
| **Random Forest**     | 0,7774   | **0,6991** | 0,7626        | 0,7748          |
| Gradient Boosting     | 0,7627   | 0,6896     | 0,7522        | 0,7739          |
| Logistic Regression   | 0,7695   | 0,6850     | 0,7544        | 0,7674          |
| SVM (RBF)             | 0,7537   | 0,6700     | 0,7400        | 0,7598          |
| Decision Tree         | 0,7322   | 0,6530     | 0,7207        | 0,7468          |
| KNN                   | 0,6791   | 0,5796     | 0,6628        | 0,6977          |

**Cara membacanya:** Random Forest terbaik di accuracy maupun F1 macro, sehingga dipilih untuk dilanjutkan. Skor cross-validation-nya (0,7748) dekat dengan accuracy data uji (0,7774), jadi model tidak mengalami *overfitting* yang berarti. KNN paling lemah, kemungkinan karena 36 fitur dengan banyak fitur kode kategori kurang cocok dengan ukuran jarak Euclidean.

### 9. Random Forest dengan Penyeimbangan Kelas

Random Forest terpilih lalu dilatih ulang dengan `class_weight='balanced'`, `max_depth=20`, `max_features='log2'`, `min_samples_leaf=4`, dan `n_estimators=300`. Perbandingan sebelum dan sesudahnya sudah saya tampilkan di bagian [imbalance](#apakah-datanya-imbalance).

Rincian kinerja akhir per kelas (classification report pada data uji):

| Kelas      | Precision | Recall | F1-score | Jumlah data uji |
| ---------- | --------- | ------ | -------- | --------------- |
| Dropout    | 0,824     | 0,694  | 0,753    | 284             |
| Enrolled   | 0,466     | 0,560  | 0,509    | 159             |
| Graduate   | 0,831     | 0,855  | 0,843    | 442             |
| **Accuracy** |         |        | **0,750** | 885            |

Dari 284 mahasiswa Dropout, sebanyak 197 diprediksi benar. Kelas **Enrolled tetap yang paling sulit** (F1 hanya 0,509), karena posisinya memang berada "di tengah" antara Dropout dan Graduate: dari 159 mahasiswa Enrolled, 89 diprediksi benar dan sisanya tertukar dengan kelas lain.

### 10. Feature Importance (fitur mana yang paling berpengaruh)

Empat fitur terpenting menurut Random Forest seluruhnya berasal dari kinerja akademik:

| Peringkat | Fitur                                   | Importance |
| --------- | --------------------------------------- | ---------- |
| 1         | Curricular units 2nd sem (approved)     | 0,1505     |
| 2         | Curricular units 2nd sem (grade)        | 0,1108     |
| 3         | Curricular units 1st sem (approved)     | 0,0944     |
| 4         | Curricular units 1st sem (grade)        | 0,0671     |
| 5         | Tuition fees up to date                 | 0,0497     |

Hasil ini konsisten dengan temuan di tahap EDA: **performa akademik tahun pertama dan kondisi pembayaran biaya kuliah adalah penentu utama status akademik mahasiswa.**

### 11. Penyimpanan Model

Model terbaik disimpan memakai `joblib` ke berkas `rf_dropout_model.joblib`. Isinya satu paket: **model**, **scaler**, **label encoder**, dan **daftar urutan fitur**. Semuanya dibawa bersama karena data baru harus diproses dengan cara yang **persis sama** seperti saat model dilatih. Saya juga menguji muat ulang dan memprediksi 5 mahasiswa dari data uji: 4 dari 5 prediksi sesuai dengan kondisi aktual.

---

## Kesimpulan

Status akademik mahasiswa pada data ini paling dipengaruhi oleh **kinerja akademik di semester 1 dan 2** (jumlah mata kuliah lulus dan nilai rata-rata), disusul **status pembayaran biaya kuliah**, **beasiswa**, dan **usia saat mendaftar**. Dari enam model yang dibandingkan, **Random Forest** menjadi yang terbaik, dengan accuracy **0,7774** dan F1 macro **0,6991**. Setelah penyeimbangan kelas, accuracy menjadi 0,7503 dan F1 macro 0,7016, dengan recall kelas Enrolled naik dari 0,365 menjadi 0,560.

Pelajaran terpentingnya: **pada data yang kelasnya tidak seimbang, accuracy saja tidak cukup untuk memilih model.** Saya juga belajar bahwa pencilan tidak selalu harus dibuang, karena pencilan usia justru membawa sinyal tentang risiko dropout.

---

## Isi repositori

```
.
├── Praktikum_ML_Dropout_Mahasiswa_Fixed.ipynb   # notebook utama (kode + penjelasan tiap langkah)
├── mahasiswa.csv                                 # dataset mentah (pemisah titik koma)
├── README.md                                     # dokumen ini
└── (hasil setelah notebook dijalankan)
    └── rf_dropout_model.joblib                   # paket model: model + scaler + label encoder + daftar fitur
```

## Cara menjalankan

**Opsi 1: Google Colab (paling mudah).** Buka [Google Colab](https://colab.research.google.com), unggah berkas `Praktikum_ML_Dropout_Mahasiswa_Fixed.ipynb` **dan** `mahasiswa.csv` ke sesi yang sama (lewat ikon folder di sisi kiri), lalu pilih *Runtime → Run all*.

**Opsi 2: di komputer sendiri.**

```
# 1. Salin repositori
git clone <URL-REPOSITORI-INI>
cd <NAMA-FOLDER>

# 2. Pasang pustaka yang dibutuhkan
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter

# 3. Buka notebook
jupyter notebook Praktikum_ML_Dropout_Mahasiswa_Fixed.ipynb
```

Pastikan `mahasiswa.csv` berada di folder yang sama dengan notebook, karena notebook membacanya langsung dari folder tersebut.

## Teknologi yang dipakai

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, dan joblib.

## Catatan dan rencana pengembangan

Saya menulis ini apa adanya supaya jujur soal batasannya:

- **Kelas Enrolled masih sulit diprediksi.** F1-nya hanya 0,509 dan precision-nya 0,466. Penyeimbangan kelas membantu, tetapi belum menuntaskan masalahnya.
- **Hyperparameter belum terdokumentasi.** Nilai hyperparameter Random Forest di notebook adalah hasil pencarian sebelumnya yang dimasukkan langsung. Ke depan saya ingin mendokumentasikan pencariannya secara sistematis (misalnya `GridSearchCV` atau `RandomizedSearchCV` dengan cross-validation) agar bisa direproduksi.
- **Teknik penyeimbangan lain belum dicoba.** Saya baru memakai `class_weight`. Ke depan ingin mencoba SMOTE atau penyesuaian ambang keputusan.
- **Fitur kode kategorikal diperlakukan sebagai angka.** Fitur seperti `Course` dan `Application mode` sebenarnya kategori tanpa urutan. Alternatif yang lebih tepat adalah One-Hot Encoding atau target encoding.
- **Multikolinearitas belum ditangani.** Ada lima pasangan fitur dengan korelasi di atas 0,80. Model berbasis pohon relatif tahan terhadap hal ini, tetapi untuk Logistic Regression bisa memengaruhi koefisien.
- **Evaluasi memakai satu kali pembagian data**, ditambah cross-validation pada data latih. Hasil di data uji bisa sedikit berubah kalau pembagiannya berbeda.
- **Algoritma lain belum dibandingkan**, misalnya XGBoost atau LightGBM, dan analisis kesalahan prediksi per program studi belum dilakukan.

## Sumber data

Dataset **Predict Students' Dropout and Academic Success** diambil dari UCI Machine Learning Repository:
<https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success>

- **Dibuat oleh:** Valentim Realinho, Mónica Vieira Martins, Jorge Machado, dan Luís Baptista, dari **Instituto Politécnico de Portalegre** (Portugal).
- **Didonasikan ke UCI:** 12 Desember 2021.
- **Pendanaan:** program SATDAP - Capacitação da Administração Pública (hibah POCI-05-5762-FSE-000191), Portugal.
- **Lisensi:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode). Data boleh dibagikan dan diadaptasi untuk tujuan apa pun selama mencantumkan kredit.
- **Makalah pengantar:** Martins, M. V., Tolledo, D., Machado, J., Baptista, L. M. T., & Realinho, V. (2021). *Early prediction of student's performance in higher education: a case study*. Trends and Applications in Information Systems and Technologies.

---

Terima kasih sudah membaca!

**Juan Luthfi Ryandi**
Teknik Informatika, Universitas Pamulang
