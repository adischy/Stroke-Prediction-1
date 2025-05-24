# Laporan Proyek Machine Learning - Adis Cahya Aprilian Nabilah

## Domain Proyek
Domain yang dipilih untuk proyek *machine learning* ini adalah **Kesehatan**, yang berjudul **Predictive Analytics: Stroke Prediction Using Patient Health Records and Lifestyle Factors**

**Latar Belakang**

![Infographics: Pancreatic Cancer, medindia.net](https://www.stroke.org/-/media/Stroke-Images/Stroke-Professionals/CS-2021-Infographic-2021.jpg?h=479&w=370&sc_lang=en)

Stroke merupakan penyebab kematian kedua di dunia dengan 11% total kematian global ([1](https://www.who.int/news-room/fact-sheets/detail/the-top-10-causes-of-death)).  
Di Indonesia, prevalensi stroke mencapai 10.9‰ dengan mortalitas 19.9% ([2](https://www.litbang.kemkes.go.id/laporan-riset-kesehatan-dasar-riskesdas-2018/)).  
Deteksi dini melalui prediksi risiko stroke menjadi krusial karena:
- 80% stroke dapat dicegah melalui manajemen faktor risiko ([3](https://www.cdc.gov/stroke/facts.htm))
- Biaya perawatan stroke akut mencapai Rp 25-50 juta/hari ([4](https://www.kemkes.go.id/article/view/21060800001/biaya-penanganan-pasien-stroke.html))

Proyek ini mengembangkan model prediktif untuk mengidentifikasi pasien berisiko stroke berdasarkan rekam medis dan gaya hidup, menggunakan dataset dari 5110 pasien.

**Mengapa masalah ini harus diselesaikan?**
- Stroke menyebabkan beban ekonomi dan sosial yang sangat besar bagi pasien dan keluarga.
- Deteksi dini risiko stroke memungkinkan intervensi lebih awal, sehingga dapat menurunkan angka kematian dan kecacatan.
- Model prediksi berbasis data dapat membantu tenaga medis dalam pengambilan keputusan dan prioritas penanganan.

**Referensi Terkait**
- [A Machine Learning Approach for Predicting Stroke Risk in Patients Using Health Records (Springer, 2022)](https://link.springer.com/article/10.1007/s00500-022-07579-7)
- [World Health Organization - The top 10 causes of death (2022)](https://www.who.int/news-room/fact-sheets/detail/the-top-10-causes-of-death)
- [Laporan Riskesdas 2018 - Kementerian Kesehatan RI](https://www.litbang.kemkes.go.id/laporan-riset-kesehatan-dasar-riskesdas-2018/)
- [CDC - Stroke Facts (2023)](https://www.cdc.gov/stroke/facts.htm)
- [Kementerian Kesehatan RI - Biaya Penanganan Pasien Stroke](https://www.kemkes.go.id/article/view/21060800001/biaya-penanganan-pasien-stroke.html)

## Business Understanding

### Problem Statements
Berdasarkan latar belakang, berikut ini merupakan rincian masalah yang diselesaikan pada proyek ini:
- Bagaimana merancang model machine learning yang efektif dalam memprediksi risiko stroke berdasarkan variabel-variabel kesehatan pasien?
- Model algoritma machine learning apa yang mampu memberikan performa prediksi terbaik terhadap risiko stroke?

### Goals
Tujuan dari proyek ini adalah:
- Membangun model machine learning yang dapat memprediksi apakah seorang pasien berisiko terkena stroke atau tidak, berdasarkan data kesehatan dan gaya hidupnya.
- Membandingkan performa beberapa algoritma machine learning untuk menemukan model yang memiliki akurasi terbaik dalam memprediksi risiko stroke.

### Solution statements
Untuk mencapai tujuan tersebut, dalam proyek ini akan dibuat beberapa model yang berbeda untuk dibandingkan, diantaranya adalah menggunakan:
- **K-Nearest Neighbor (KNN)** adalah algoritma klasifikasi yang menentukan kelas suatu data baru berdasarkan mayoritas kelas dari sejumlah tetangga terdekatnya di data latih. Algoritma ini mudah diimplementasikan dan efektif untuk dataset kecil hingga menengah, namun sensitif terhadap skala data dan outlier [[5](https://scikit-learn.org/stable/modules/neighbors.html#classification)].
- **Random Forest** adalah algoritma ensemble yang membangun banyak pohon keputusan (decision tree) dan menggabungkan hasilnya untuk meningkatkan akurasi prediksi dan mengurangi overfitting. Algoritma ini sangat baik dalam menangani data dengan banyak fitur dan dapat memberikan estimasi pentingnya setiap fitur [[6](https://scikit-learn.org/stable/modules/ensemble.html#random-forests)].
- **Support Vector Classifier (SVC)** adalah implementasi dari Support Vector Machine (SVM) untuk klasifikasi. Algoritma ini mencari hyperplane optimal yang memisahkan kelas-kelas data dengan margin terbesar. SVC efektif untuk data berdimensi tinggi dan dapat menggunakan kernel untuk menangani data non-linear [[7](https://scikit-learn.org/stable/modules/svm.html#svm-classification)].
- **Naive Bayes** adalah algoritma klasifikasi probabilistik yang didasarkan pada Teorema Bayes dengan asumsi independensi antar fitur. Algoritma ini sangat cepat, sederhana, dan sering digunakan sebagai baseline untuk klasifikasi teks atau data kategorikal. [[8](https://scikit-learn.org/stable/modules/naive_bayes.html)].

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah data rekam medis pasien yang terdiri dari **5110 sampel**. Dataset ini tersedia secara publik dan dapat diunduh melalui [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset). Pada dataset ini terdapat **12 kolom fitur**, yang merepresentasikan informasi demografis, gaya hidup, dan kondisi kesehatan pasien. Berikut adalah daftar kolom pada dataset:

| #   | Column             | Dtype   |
|-----|--------------------|---------|
| 0   | id                 | int64   |
| 1   | gender             | object  |
| 2   | age                | float64 |
| 3   | hypertension       | int64   |
| 4   | heart_disease      | int64   |
| 5   | ever_married       | object  |
| 6   | work_type          | object  |
| 7   | Residence_type     | object  |
| 8   | avg_glucose_level  | float64 |
| 9   | bmi                | float64 |
| 10  | smoking_status     | object  |
| 11  | stroke             | int64   |

Terdapat missing value pada salah satu fitur di dataset.

| Column | Tipe Data | Jumlah Missing Value |
|--------|-----------|----------------------|
| bmi    | float64   | 201                  |

Dataset mempunyai nilai outliers pada fitur-fitur numerical. Fitur avg_glucose_leve dan bmi memiliki jumlah outliers paling banyak.

![bp1](https://drive.google.com/uc?export=view&id=1LeqGjA_Cu45vapT7IqNwCmoHb9Y_TUQE)
![bp2](https://drive.google.com/uc?export=view&id=1o5JxZsQT7ReT8t3Lc2RPGKt9qdORCKYZ)

### Variabel-variabel pada dataset adalah sebagai berikut:
1. `id` – ID unik untuk setiap pasien
2. `gender` – Jenis kelamin pasien
3. `age` – Usia pasien
4. `hypertension` – Riwayat hipertensi (0 = tidak, 1 = ya)
5. `heart_disease` – Riwayat penyakit jantung (0 = tidak, 1 = ya)
6. `ever_married` – Status pernah menikah (Yes/No)
7. `work_type` – Jenis pekerjaan pasien
8. `Residence_type` – Tipe tempat tinggal (Urban/Rural)
9. `avg_glucose_level` – Rata-rata kadar glukosa dalam darah
10. `bmi` – Indeks massa tubuh pasien
11. `smoking_status` – Status merokok pasien (pernah, tidak pernah, dll.)
12. `stroke` – Target (1 = pasien pernah mengalami stroke, 0 = tidak pernah)

## Data Preparation

### Teknik Data Preparation

* **Menghapus fitur yang tidak perlu**: Menghapus kolom yang tidak relevan seperti `id`.
* **Menangani Missing Value**: Mengimputasi nilai yang hilang pada kolom `bmi` menggunakan median berdasarkan kelompok usia atau gender. Jika tidak tersedia, digunakan median global sebagai cadangan (fallback).
* **One Hot Encoding**: Mengubah variabel kategorikal menjadi variabel numerik menggunakan teknik one-hot encoding.
* **Menangani Outliers**: Menghapus data yang memiliki nilai outliers pada kolom `avg_glucose_level`, dan `bmi` dengan metode IQR.
* **Data Split (Train-Test-Split)**: Membagi data menjadi 80% untuk pelatihan dan 20% untuk pengujian, dengan `stroke` sebagai target.
* **SMOTE (Synthetic Minority Over-sampling Technique)**: Melakukan oversampling pada data training untuk menyeimbangkan jumlah kelas mayoritas dan minoritas (stroke), sehingga model dapat belajar lebih baik pada kasus stroke yang jumlahnya sedikit.
* **Normalisasi**: Melakukan normalisasi menggunakan MinMaxScaler agar semua fitur numerik berada dalam skala 0 hingga 1.

---

### Proses Data Preparation

* Kolom `id` dihapus karena tidak memiliki nilai prediktif dalam diagnosis stroke.

* Kolom `bmi` memiliki 201 missing value. Dilakukan imputasi dengan median berdasarkan gender atau usia, dan fallback imputasi dengan median global.

* Outliers pada `avg_glucose_level`, dan `bmi` diidentifikasi menggunakan metode IQR dan dihapus. Visualisasi dilakukan menggunakan boxplot.

* Variabel kategorikal (`gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status`) diencoding menggunakan teknik one-hot encoding.

* Dataset dibagi menjadi:

  | Dataset       | Jumlah |
  | ------------- | ------ |
  | Whole Dataset | 4393   |
  | Train         | 3514   |
  | Test          | 879    |

* **SMOTE** diterapkan pada data training untuk menyeimbangkan jumlah kasus stroke dan non-stroke, sehingga model tidak bias terhadap kelas mayoritas.
  Sebelum SMOTE: [3382  132]
  Setelah SMOTE: [3382 3382]
  
* Semua fitur numerik dinormalisasi menggunakan MinMaxScaler untuk menghindari dominasi fitur dengan skala besar dan membantu model belajar lebih optimal.

### Alasan Tahapan Data Preparation Dilakukan

* **Drop kolom `id`** dilakukan karena tidak memberikan kontribusi informasi terhadap target dan hanya berperan sebagai penanda unik.
* **Menangani Missing Value** penting untuk menjaga integritas data dan mencegah error saat training model.
* **Encoding categorical variables** memungkinkan algoritma machine learning memproses informasi kategorikal dalam bentuk numerik.
* **Removing outliers** membantu meningkatkan kualitas data dan performa model, karena data ekstrem bisa mempengaruhi hasil pelatihan.
* **Data splitting 80:20** memberikan keseimbangan antara pelatihan dan pengujian untuk mengevaluasi generalisasi model secara adil.
* **SMOTE** digunakan untuk mengatasi ketidakseimbangan kelas, sehingga model tidak hanya belajar dari kelas mayoritas, tetapi juga mampu mengenali kasus stroke yang jumlahnya jauh lebih sedikit.
* **Feature scaling dengan MinMaxScaler** membuat semua fitur berada dalam skala seragam, mempercepat konvergensi, dan meningkatkan performa algoritma yang sensitif terhadap skala fitur.

## Modeling

**Tahap Modeling**

Pada tahap ini, empat algoritma klasifikasi digunakan untuk memprediksi risiko stroke, yaitu:

* K-Nearest Neighbors (KNN)
* Random Forest
* Support Vector Machine (SVM)
* Bernoulli Naive Bayes

Seluruh model dilatih menggunakan data training hasil SMOTE (untuk mengatasi ketidakseimbangan kelas) dan telah dinormalisasi menggunakan MinMaxScaler. Untuk Random Forest dan SVM, digunakan `class_weight='balanced'` agar model lebih memperhatikan kelas minoritas (stroke).

- Melatih model KNN
    ```python
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_res_scaled, y_train_res)
    ```
    Model KNN dilatih menggunakan KNeighborsClassifier dengan parameter n_neighbors=5, yang berarti model akan memprediksi berdasarkan 5 tetangga terdekat. Model dilatih pada data latih X_train_res_scaled dan y_train_res.
- Melatih model Random Forest
    ```python
    RF = RandomForestClassifier(class_weight='balanced')
    RF.fit(X_train_res_scaled, y_train_res)
    ```
    Model Random Forest dilatih menggunakan RandomForestClassifier(class_weight='balanced'), algoritma ensemble berbasis pohon keputusan yang bekerja dengan membuat banyak pohon keputusan dan menggabungkannya untuk hasil prediksi yang lebih stabil dan akurat. Model dilatih pada data latih yang sama.
- Melatih model Support Vector Classifier
    ```python
    svc = SVC(probability=True, class_weight='balanced')
    svc.fit(X_train_res_scaled, y_train_res)
    ```
    Model Support Vector Classifier dilatih menggunakan SVC(probability=True, class_weight='balanced'), yang bekerja dengan memisahkan kelas menggunakan hyperplane terbaik. Tidak ada parameter khusus yang diatur, sehingga menggunakan parameter default.
- Melatih model Bernoulli Naive Bayes
    ```python
    NB = BernoulliNB()
    NB.fit(X_train_res_scaled, y_train_res)
    ```
    Model Bernoulli Naive Bayes dilatih menggunakan BernoulliNB, yang cocok untuk fitur biner. Model ini mengasumsikan bahwa fitur-fitur independen satu sama lain dalam setiap kelas. Cocok digunakan jika data telah dibinerisasi.

**Tahapan dan Parameter yang Digunakan**

Pada tahap modeling, empat algoritma berbeda digunakan untuk memprediksi apakah pasien mengalami stroke:

- K-Nearest Neighbors (KNN):
    - Parameter: `n_neighbors=5`
    - Deskripsi: Algoritma KNN mencari 5 tetangga terdekat untuk memprediksi label berdasarkan mayoritas kelas dari tetangga-tetangga tersebut.
- Random Forest:
    - Parameter: `class_weight='balanced'`
    - Deskripsi: Random Forest membangun banyak pohon keputusan dan menggabungkan hasilnya. Dengan `class_weight='balanced'`, model menjadi lebih sensitif terhadap kelas minoritas.
- Support Vector Classifier (SVC):
    - Parameter: `probability=True` dan `class_weight='balanced'`
    - Deskripsi: SVM mencari hyperplane terbaik untuk memisahkan kelas. Dengan class_weight='balanced', model lebih peka terhadap kelas stroke yang jumlahnya sedikit.
- Bernoulli Naive Bayes:
    - Parameter: default (`BernoulliNB()`)
    - Deskripsi: Model ini mengasumsikan fitur bersifat biner dan saling independen. Cocok untuk data hasil one-hot encoding.

**Kelebihan dan Kekurangan Setiap Algoritma**
- K-Nearest Neighbors (KNN):
    - Kelebihan:
        - Sederhana dan mudah diimplementasikan.
        - Tidak membutuhkan proses pelatihan yang kompleks.
    - Kekurangan:
        - Sensitif terhadap outliers dan noise.
        - Tidak efisien untuk dataset besar karena proses prediksi memerlukan pencarian tetangga.
- Random Forest:
    - Kelebihan:
        - Dapat menangani data kompleks dan non-linear.
        - Mengurangi risiko overfitting dengan teknik ensemble.
        - Dapat menangani data tidak seimbang dengan `class_weight`.
    - Kekurangan:
        - Model sulit untuk diinterpretasikan (kurang transparan).
        - Membutuhkan lebih banyak memori dan waktu komputasi.
- Support Vector Classifier (SVC):
    - Kelebihan:
        - Akurasi tinggi untuk data dengan margin yang jelas.
        - Efektif pada dataset berdimensi tinggi.
        - Dapat menangani data tidak seimbang dengan `class_weight`.
    - Kekurangan:
        - Waktu komputasi lebih tinggi terutama pada dataset besar.
        - Parameter kernel dan C perlu tuning khusus.
- Bernoulli Naive Bayes:
    - Kelebihan:
        - Cepat dan efisien, terutama untuk data teks atau biner.
        - Mudah diinterpretasikan.
    - Kekurangan:
        - Asumsi independensi antar fitur sering tidak realistis.
        - Kinerja menurun jika fitur tidak biner atau distribusinya tidak cocok.

## Evaluation

**Memilih Model Terbaik Sebagai Solusi**

Setelah pelatihan dan prediksi dilakukan, performa dari masing-masing model dievaluasi menggunakan lima metrik utama:

* Accuracy: Seberapa sering model membuat prediksi yang benar.
* Precision: Proporsi prediksi stroke yang benar-benar stroke.
* Recall: Kemampuan model dalam mendeteksi kasus stroke.
* F1-Score: Harmoni antara precision dan recall, penting untuk data tidak seimbang.
* ROC-AUC: Mengukur kemampuan model membedakan antara stroke dan bukan stroke.

Tabel berikut menunjukkan hasil evaluasi lengkap dari keempat model:
| Metrik        | KNN       | Random Forest | SVM       | Naive Bayes |
| ------------- | --------- | ------------- | --------- | ----------- |
| **Accuracy**  | 0.859     | **0.918**     | 0.861     | 0.761       |
| **Precision** | 0.090     | 0.047         | **0.092** | 0.032       |
| **Recall**    | **0.303** | 0.061         | **0.303** | 0.182       |
| **F1-Score**  | 0.139     | 0.053         | **0.141** | 0.054       |
| **ROC-AUC**   | 0.631     | **0.777**     | 0.739     | 0.573       |

Diagram batang di bawah menunjukkan perbandingan visual dari accuracy, precision, recall, f1-score, dan ROC-AUC tiap model:
**Perbandingan accuracy berbagai model**
![Akurasi](https://drive.google.com/uc?export=view&id=1A9bK9-WNzW1nYkYx0DM4uBSOFanC4TRt)

**Perbandingan precision berbagai model**
![Precision](https://drive.google.com/uc?export=view&id=1Ef-8rQLmujRh_zUEfklgG9tSqw-S29RP)

**Perbandingan recall berbagai model**
![Recall](https://drive.google.com/uc?export=view&id=1xFOBLV2yG6IcAwVD7AFxPUqNR79U0WHG)

**Perbandingan f1-score berbagai model**
![F1](https://drive.google.com/uc?export=view&id=1onlnanbB4lIEDK7rPv2LfFruGsffEkLz)

**Perbandingan ROC-AUC berbagai model**
![ROC AUC](https://drive.google.com/uc?export=view&id=1IqOFfIUhrb7OTMg-1RQE9f_m2NF0xpl5)

- Meskipun Random Forest menunjukkan akurasi tertinggi (0.918), model ini hanya mendeteksi sekitar 6% kasus stroke (recall = 0.061).
- Support Vector Machine (SVM) memberikan performa paling seimbang, dengan recall 30.3%, f1-score tertinggi (0.141), dan ROC-AUC yang tinggi (0.739).
- Oleh karena itu, SVM dipilih sebagai model terbaik untuk mendeteksi stroke pada data yang tidak seimbang.
- Namun, jika dibutuhkan kecepatan dan interpretasi yang sederhana, maka Naive Bayes atau KNN dapat dipertimbangkan sebagai alternatif.
- Model yang dibangun menyelesaikan kasus klasifikasi biner, yaitu memprediksi apakah seorang pasien berisiko terkena stroke atau tidak.

Pada kasus prediksi stroke, recall dan f1-score lebih penting daripada akurasi. Recall mengukur seberapa banyak kasus stroke yang berhasil dideteksi, sedangkan f1-score menyeimbangkan antara recall dan precision. Model dengan recall tinggi akan membantu deteksi dini risiko stroke dan mendukung pengambilan keputusan medis secara lebih tepat.

Melalui proses pemodelan dan evaluasi, telah berhasil dibangun model klasifikasi yang mampu memprediksi risiko stroke berdasarkan data karakteristik pasien. Model SVM terbukti memberikan performa terbaik secara keseluruhan. Setiap tahapan data preparation—mulai dari penanganan missing value, outlier, encoding, SMOTE, hingga normalisasi—berkontribusi signifikan dalam meningkatkan performa model. Implementasi solusi ini diharapkan dapat membantu deteksi dini risiko stroke dan mendukung pengambilan keputusan di bidang kesehatan.

## Referensi

[[1](https://www.who.int/news-room/fact-sheets/detail/the-top-10-causes-of-death)] World Health Organization. (2020). *The top 10 causes of death*. WHO Fact Sheets. [https://www.who.int/news-room/fact-sheets/detail/the-top-10-causes-of-death](https://www.who.int/news-room/fact-sheets/detail/the-top-10-causes-of-death)

[[2](https://www.litbang.kemkes.go.id/laporan-riset-kesehatan-dasar-riskesdas-2018/)] Badan Penelitian dan Pengembangan Kesehatan. (2018). *Laporan Riset Kesehatan Dasar (Riskesdas) 2018*. Kementerian Kesehatan Republik Indonesia. [https://www.litbang.kemkes.go.id/laporan-riset-kesehatan-dasar-riskesdas-2018/](https://www.litbang.kemkes.go.id/laporan-riset-kesehatan-dasar-riskesdas-2018/)

[[3](https://www.cdc.gov/stroke/facts.htm)] Centers for Disease Control and Prevention. (2022). *Stroke facts*. CDC. [https://www.cdc.gov/stroke/facts.htm](https://www.cdc.gov/stroke/facts.htm)

[[4](https://www.kemkes.go.id/article/view/21060800001/biaya-penanganan-pasien-stroke.html)] Kementerian Kesehatan Republik Indonesia. (2021). *Biaya Penanganan Pasien Stroke*. [https://www.kemkes.go.id/article/view/21060800001/biaya-penanganan-pasien-stroke.html](https://www.kemkes.go.id/article/view/21060800001/biaya-penanganan-pasien-stroke.html)

[[5](https://scikit-learn.org/stable/modules/neighbors.html#classification)] Pedregosa, F., et al. (2023). *K-Nearest Neighbors classification*. In *scikit-learn documentation*. [https://scikit-learn.org/stable/modules/neighbors.html#classification](https://scikit-learn.org/stable/modules/neighbors.html#classification)

[[6](https://scikit-learn.org/stable/modules/ensemble.html#random-forests)] Pedregosa, F., et al. (2023). *Random Forests*. In *scikit-learn documentation*. [https://scikit-learn.org/stable/modules/ensemble.html#random-forests](https://scikit-learn.org/stable/modules/ensemble.html#random-forests)

[[7](https://scikit-learn.org/stable/modules/svm.html#svm-classification)] Pedregosa, F., et al. (2023). *Support Vector Machines classification*. In *scikit-learn documentation*. [https://scikit-learn.org/stable/modules/svm.html#svm-classification](https://scikit-learn.org/stable/modules/svm.html#svm-classification)

[[8](https://scikit-learn.org/stable/modules/naive_bayes.html)] Pedregosa, F., et al. (2023). *Naive Bayes*. In *scikit-learn documentation*. [https://scikit-learn.org/stable/modules/naive\_bayes.html](https://scikit-learn.org/stable/modules/naive_bayes.html)

