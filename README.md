# Laporan Proyek Prediksi Harga Sewa Apartemen

## Domain Proyek
### Latar Belakang
Dalam dunia real estat dan penyewaan apartemen, menetapkan harga sewa yang akurat sangat penting baik bagi pemilik properti maupun penyewa. Penentuan harga yang tepat tidak hanya membantu pemilik properti untuk memaksimalkan pendapatan, tetapi juga membantu calon penyewa untuk menemukan tempat tinggal yang sesuai dengan anggaran mereka.

### Mengapa Masalah Ini Harus Diselesaikan?
- Menentukan harga sewa yang tepat adalah kunci untuk menghindari kerugian baik bagi pemilik properti maupun penyewa.
- Memiliki sistem prediksi harga yang andal dapat meningkatkan efisiensi pasar penyewaan properti.

### Referensi Terkait
- [Using machine learning algorithms for predicting real estate values in tourism centers](https://example.com)
- [Predicting property prices with machine learning algorithms](https://example.com)


## Business Understanding
### Problem Statements
- Bagaimana cara memprediksi harga sewa apartemen berdasarkan fitur-fitur seperti jumlah kamar mandi, kamar tidur, luas apartemen, dll?
- Fitur-fitur apa yang paling berpengaruh dalam menentukan harga sewa apartemen?
- Bagaimana meningkatkan akurasi prediksi harga sewa dengan menggunakan teknik machine learning?

### Goals
- Membangun model machine learning yang dapat memprediksi harga sewa apartemen dengan akurasi tinggi.
- Mengidentifikasi fitur-fitur yang paling berpengaruh dalam menentukan harga sewa.
- Meningkatkan akurasi model prediksi melalui hyperparameter tuning dan teknik machine learning yang tepat.

### Solution Statements
- Menggunakan beberapa algoritma machine learning seperti KNN, Random Forest, dan AdaBoost untuk memprediksi harga sewa.
- Membandingkan performa model dan memilih model terbaik berdasarkan metrik evaluasi seperti Mean Squared Error (MSE).

## Data Understanding

### Dataset
Dataset yang digunakan berasal dari UCI Machine Learning Repository. Dataset ini berisi informasi tentang apartemen yang disewakan, termasuk fitur-fitur seperti jumlah kamar mandi, kamar tidur, luas apartemen, dan harga sewa.

Dataset memiliki jumlah 10.000 baris dan 22 kolom.

| Column         | Dtype    |
|----------------|----------|
| id             | int64    |
| category       | object   |
| title          | object   |
| body           | object   |
| amenities      | object   |
| bathrooms      | float64  |
| bedrooms       | float64  |
| currency       | object   |
| fee            | object   |
| has_photo      | object   |
| pets_allowed   | object   |
| price          | int64    |
| price_display  | object   |
| price_type     | object   |
| square_feet    | int64    |
| address        | object   |
| cityname       | object   |
| state          | object   |
| latitude       | float64  |
| longitude      | float64  |
| source         | object   |
| time           | int64    |


### Missing Values
| Tipe Data      | Jumlah Missing Value |
|----------------|----------------------|
| amenities      | 3549                 |
| bathrooms      | 34                   |
| bedrooms       | 7                    |
| pets_allowed   | 4163                 |
| address        | 3327                 |
| cityname       | 77                   |
| state          | 77                   |
| latitude       | 10                   |
| longitude      | 10                   |

### Outliers
Dataset mempunyai nilai outliers pada fitur-fitur numerical, terutama pada fitur `price` dan `square_feet`.
![boxplot](./images/outputcode.png)

### Exploratory Data Analysis (EDA)
- Membuat korelasi heatmap untuk melihat nilai korelasi antar fitur numerik.
- Membuat histogram untuk melihat persebaran data numerik.

#### Korelasi Heatmap
- **Luas Bangunan (square_feet)** memiliki hubungan kuat dengan **bathrooms** dan **bedrooms**.
- **Harga Sewa (price)** memiliki korelasi yang rendah dengan **bathrooms** dan **bedrooms**, menandakan faktor lain seperti lokasi atau fasilitas yang lebih mempengaruhi harga.

#### Histogram
- **Bathrooms**: Mayoritas memiliki 1 kamar mandi.
- **Bedrooms**: Mayoritas memiliki 1 atau 2 kamar tidur.
- **Price**: Harga sewa apartemen tersebar dengan puncak sekitar 700-1000 USD, dengan sedikit skew ke kanan.
- **Square_feet**: Luas apartemen bervariasi, dengan puncak sekitar 500-800 kaki persegi.

## Data Preparation

### Teknik Data Preparation
- **Handling Missing Values**: Mengimputasi atau menghapus nilai yang hilang pada dataset.
- **Removing Outliers**: Menghapus data yang memiliki nilai outliers.
- **Encoding Categorical Variables**: Mengubah variabel kategorikal menjadi numerik menggunakan teknik one-hot encoding.
- **Pembagian Dataset**: Pembagian data train-test dengan rasio 80:20.
- **Feature Scaling**: Melakukan standarisasi pada fitur numerik.

### Proses Data Preparation
1. Fitur dengan missing value < 100 akan di-drop.
2. Fitur dengan missing value > 1000 akan diimputasi.
3. Outlier diatasi menggunakan metode IQR.
4. Fitur seperti `id`, `latitude`, `longitude`, dan `time` di-drop.
5. Kategorikal fitur di-encode menggunakan one-hot encoding.
6. Pembagian data dengan skema 80:20 untuk training dan testing.
7. Feature scaling menggunakan StandardScaler.

### Dataset Pembagian
- **Whole Dataset**: 8136
- **Train**: 6508
- **Test**: 1628

## Modeling

### Tahap Modeling
Algoritma yang digunakan:
- **K-Nearest Neighbors (KNN)**: Parameter `n_neighbors=10`.
- **Random Forest**: Parameter `n_estimators=50, max_depth=16`.
- **AdaBoost**: Parameter `learning_rate=0.05`.

### Kelebihan dan Kekurangan
- **KNN**:
  - Kelebihan: Sederhana, tidak ada asumsi distribusi data.
  - Kekurangan: Sensitif terhadap outliers dan noise.
- **Random Forest**:
  - Kelebihan: Dapat menangani data yang kompleks, robust terhadap overfitting.
  - Kekurangan: Interpretasi model lebih sulit, memerlukan lebih banyak sumber daya.
- **AdaBoost**:
  - Kelebihan: Dapat meningkatkan akurasi dengan menggabungkan beberapa model.
  - Kekurangan: Rentan terhadap outliers.

### Model Terbaik
Berdasarkan evaluasi, **Random Forest** dipilih sebagai model terbaik karena memiliki nilai MSE terendah pada data uji.

## Evaluation

### Metrik Evaluasi
- **Mean Squared Error (MSE)**: Mengukur rata-rata kuadrat dari kesalahan prediksi. MSE yang lebih rendah menunjukkan performa model yang lebih baik.

### Hasil Proyek

| Model          | Train MSE | Test MSE |
|----------------|-----------|----------|
| KNN            | 78.1      | 97.8     |
| Random Forest  | 59.2      | 93.7     |
| AdaBoost       | 148.3     | 156.2    |

### Kesimpulan
- Model Random Forest terbukti menjadi model terbaik dengan MSE terendah.
- Hyperparameter tuning memainkan peran penting dalam meningkatkan performa model.
- Solusi yang diimplementasikan berhasil memenuhi problem statement dan goals yang ditetapkan.

