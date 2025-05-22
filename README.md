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

![Boxplot Outliers 1](https://drive.google.com/uc?export=view&id=1JfbQ9f_i0OKtl2zMMv9pPyZG2gxyUQJe)
![Boxplot Outliers 2](https://drive.google.com/uc?export=view&id=18C4Whluwe3YdiLS9tYWRmW8xXjbDwP8y)

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

* Semua fitur numerik dinormalisasi menggunakan MinMaxScaler untuk menghindari dominasi fitur dengan skala besar dan membantu model belajar lebih optimal.

### Alasan Tahapan Data Preparation Dilakukan

* **Drop kolom `id`** dilakukan karena tidak memberikan kontribusi informasi terhadap target dan hanya berperan sebagai penanda unik.
* **Menangani Missing Value** penting untuk menjaga integritas data dan mencegah error saat training model.
* **Encoding categorical variables** memungkinkan algoritma machine learning memproses informasi kategorikal dalam bentuk numerik.
* **Removing outliers** membantu meningkatkan kualitas data dan performa model, karena data ekstrem bisa mempengaruhi hasil pelatihan.
* **Data splitting 80:20** memberikan keseimbangan antara pelatihan dan pengujian untuk mengevaluasi generalisasi model secara adil.
* **Feature scaling dengan MinMaxScaler** membuat semua fitur berada dalam skala seragam, mempercepat konvergensi, dan meningkatkan performa algoritma yang sensitif terhadap skala fitur.

## Modeling

**Tahap Modeling**

- Menyiapkan DataFrame untuk Analisis Masing-Masing Model
    ```python
    models = pd.DataFrame(index=['accuracy_score'],
                      columns=['KNN', 'RandomForest', 'SVM', 'Naive Bayes'])
    ```
    Memulai dengan menyiapkan DataFrame bernama models untuk menyimpan nilai Mean Squared Error (MSE) pada data latih dan uji untuk setiap model yang akan diuji, yaitu KNN, Random Forest, SVC, dan Naive Bayes.
- Melatih model KNN
    ```python
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    ```
    Model KNN dilatih menggunakan KNeighborsClassifier dengan parameter n_neighbors=5, yang berarti model akan memprediksi berdasarkan 5 tetangga terdekat. Model dilatih pada data latih X_train dan y_train.
- Melatih model Random Forest
    ```python
    RF = RandomForestClassifier()
    RF.fit(X_train, y_train)
    ```
    Model Random Forest dilatih menggunakan RandomForestClassifier, algoritma ensemble berbasis pohon keputusan yang bekerja dengan membuat banyak pohon keputusan dan menggabungkannya untuk hasil prediksi yang lebih stabil dan akurat. Model dilatih pada data latih yang sama.
- Melatih model Support Vector Classifier
    ```python
    svc = SVC()
    svc.fit(X_train, y_train)
    ```
    Model Support Vector Classifier dilatih menggunakan SVC, yang bekerja dengan memisahkan kelas menggunakan hyperplane terbaik. Tidak ada parameter khusus yang diatur, sehingga menggunakan parameter default.
- Melatih model Bernoulli Naive Bayes
    ```python
    NB = BernoulliNB()
    NB.fit(X_train, y_train)
    ```
    Model Bernoulli Naive Bayes dilatih menggunakan BernoulliNB, yang cocok untuk fitur biner. Model ini mengasumsikan bahwa fitur-fitur independen satu sama lain dalam setiap kelas. Cocok digunakan jika data telah dibinerisasi.

**Tahapan dan Parameter yang Digunakan**

Pada tahap modeling, empat algoritma berbeda digunakan untuk memprediksi apakah pasien mengalami stroke:

- K-Nearest Neighbors (KNN):
    - Parameter: `n_neighbors=5`
    - Deskripsi: Algoritma KNN mencari 5 tetangga terdekat untuk memprediksi label berdasarkan mayoritas kelas dari tetangga-tetangga tersebut.
- Random Forest:
    - Parameter: default (`RandomForestClassifier()`)
    - Deskripsi: Random Forest merupakan algoritma ensemble yang terdiri dari banyak pohon keputusan. Setiap pohon dilatih pada subset data yang berbeda untuk meningkatkan generalisasi dan akurasi.
- Support Vector Classifier (SVC):
    - Parameter: default (`SVC()`)
    - Deskripsi: SVC bekerja dengan mencari hyperplane terbaik yang memisahkan data dalam ruang berdimensi tinggi. Cocok untuk data dengan margin klasifikasi yang jelas.
- Bernoulli Naive Bayes:
    - Parameter: default (`BernoulliNB()`)
    - Deskripsi: Digunakan untuk data biner, algoritma ini mengasumsikan bahwa setiap fitur bersifat independen dan mengikuti distribusi Bernoulli.

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
    - Kekurangan:
        - Model sulit untuk diinterpretasikan.
        - Membutuhkan lebih banyak memori dan waktu komputasi.
- Support Vector Classifier (SVC):
    - Kelebihan:
        - Memberikan akurasi tinggi untuk data dengan margin yang jelas.
        - Efektif pada dataset berdimensi tinggi.
    - Kekurangan:
        - Waktu komputasi lebih tinggi terutama pada dataset besar.
        - Parameter seperti kernel dan C perlu disesuaikan dengan hati-hati.
- Bernoulli Naive Bayes:
    - Kelebihan:
        - Cepat dan efisien, terutama untuk data teks atau biner.
        - Mudah diinterpretasikan.
    - Kekurangan:
        - Asumsi independensi antar fitur sering tidak realistis.
        - Kinerja menurun jika fitur tidak biner atau distribusinya tidak cocok.

**Memilih Model Terbaik Sebagai Solusi**

Setelah pelatihan dan prediksi dilakukan, akurasi dari masing-masing model dihitung dan dibandingkan:
| Model                           | Akurasi   |
| ------------------------------- | --------- |
| K-Nearest Neighbors (KNN)       | 0.961     |
| Random Forest                   | 0.961     |
| Support Vector Classifier (SVC) | **0.962** |
| Bernoulli Naive Bayes           | 0.957     |

Diagram batang di bawah menunjukkan perbandingan visual dari akurasi tiap model:
![accuracy](https://drive.google.com/uc?export=view&id=1M2ygS6XAL5KN7pgpmDBn3PmvpQcAibXG)

- Model SVC menunjukkan akurasi tertinggi (0.962) dibandingkan dengan model lainnya.
- Meskipun perbedaannya tidak terlalu besar, SVC dapat dipilih sebagai model terbaik untuk kasus ini.
- Namun, apabila waktu komputasi atau kemudahan interpretasi menjadi prioritas, maka Random Forest atau Naive Bayes bisa menjadi alternatif.

## Evaluation

Pada proyek ini, model yang dibangun digunakan untuk menyelesaikan **kasus klasifikasi**, sehingga **metrik evaluasi yang digunakan adalah akurasi**.

**Akurasi** merupakan metrik yang mengukur seberapa tepat model dalam melakukan prediksi, yaitu dengan menghitung persentase jumlah prediksi yang benar dibandingkan dengan seluruh jumlah data yang diprediksi. Nilai akurasi dapat dihitung menggunakan rumus berikut:

$$
\text{Akurasi} = \frac{\text{Jumlah Prediksi Benar}}{\text{Total Jumlah Prediksi}} \times 100\%
$$

Melalui proses pemodelan dan evaluasi, telah berhasil dibangun model klasifikasi yang akurat untuk memprediksi risiko stroke berdasarkan data karakteristik pasien. Model Support Vector Machine (SVM) terbukti memberikan akurasi tertinggi dibandingkan model lain seperti K-Nearest Neighbors, Random Forest, dan Naive Bayes. Setiap tahapan data preparation—mulai dari penanganan missing value, outlier, encoding, hingga normalisasi—berkontribusi signifikan dalam meningkatkan performa model. Implementasi solusi ini memberikan dampak positif, sejalan dengan tujuan awal untuk membantu deteksi dini risiko stroke dan mendukung pengambilan keputusan dalam bidang kesehatan.

## Referensi

[[1](https://www.who.int/news-room/fact-sheets/detail/the-top-10-causes-of-death)] World Health Organization. (2020). *The top 10 causes of death*. WHO Fact Sheets. [https://www.who.int/news-room/fact-sheets/detail/the-top-10-causes-of-death](https://www.who.int/news-room/fact-sheets/detail/the-top-10-causes-of-death)

[[2](https://www.litbang.kemkes.go.id/laporan-riset-kesehatan-dasar-riskesdas-2018/)] Badan Penelitian dan Pengembangan Kesehatan. (2018). *Laporan Riset Kesehatan Dasar (Riskesdas) 2018*. Kementerian Kesehatan Republik Indonesia. [https://www.litbang.kemkes.go.id/laporan-riset-kesehatan-dasar-riskesdas-2018/](https://www.litbang.kemkes.go.id/laporan-riset-kesehatan-dasar-riskesdas-2018/)

[[3](https://www.cdc.gov/stroke/facts.htm)] Centers for Disease Control and Prevention. (2022). *Stroke facts*. CDC. [https://www.cdc.gov/stroke/facts.htm](https://www.cdc.gov/stroke/facts.htm)

[[4](https://www.kemkes.go.id/article/view/21060800001/biaya-penanganan-pasien-stroke.html)] Kementerian Kesehatan Republik Indonesia. (2021). *Biaya Penanganan Pasien Stroke*. [https://www.kemkes.go.id/article/view/21060800001/biaya-penanganan-pasien-stroke.html](https://www.kemkes.go.id/article/view/21060800001/biaya-penanganan-pasien-stroke.html)

[[5](https://scikit-learn.org/stable/modules/neighbors.html#classification)] Pedregosa, F., et al. (2023). *K-Nearest Neighbors classification*. In *scikit-learn documentation*. [https://scikit-learn.org/stable/modules/neighbors.html#classification](https://scikit-learn.org/stable/modules/neighbors.html#classification)

[[6](https://scikit-learn.org/stable/modules/ensemble.html#random-forests)] Pedregosa, F., et al. (2023). *Random Forests*. In *scikit-learn documentation*. [https://scikit-learn.org/stable/modules/ensemble.html#random-forests](https://scikit-learn.org/stable/modules/ensemble.html#random-forests)

[[7](https://scikit-learn.org/stable/modules/svm.html#svm-classification)] Pedregosa, F., et al. (2023). *Support Vector Machines classification*. In *scikit-learn documentation*. [https://scikit-learn.org/stable/modules/svm.html#svm-classification](https://scikit-learn.org/stable/modules/svm.html#svm-classification)

[[8](https://scikit-learn.org/stable/modules/naive_bayes.html)] Pedregosa, F., et al. (2023). *Naive Bayes*. In *scikit-learn documentation*. [https://scikit-learn.org/stable/modules/naive\_bayes.html](https://scikit-learn.org/stable/modules/naive_bayes.html)

